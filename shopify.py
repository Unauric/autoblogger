import openai
import pandas as pd
import requests
import os
import json
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

# === Load environment variables and validate ===
openai.api_key = os.getenv("YOUR_OPEN_AI_KEY")
api_key = os.getenv("YOUR_API_KEY")
password = os.getenv("YOUR_SHOPIFY_PASSWORD")
store_address = os.getenv("YOUR_STORE_ID")  # should be full URL like: https://your-store.myshopify.com/admin/api/2023-10
blog_id = os.getenv("YOUR_BLOG_ID")
author = "Wine expert"

required_env_vars = {
    "OpenAI Key": openai.api_key,
    "Shopify API Key": api_key,
    "Shopify Password": password,
    "Shopify Store Address": store_address,
    "Shopify Blog ID": blog_id
}
for name, val in required_env_vars.items():
    if not val:
        raise EnvironmentError(f"❌ Missing required environment variable: {name}")

headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
}

# === Output DataFrame ===
output_df = pd.DataFrame(columns=['URL Slug', 'Meta Title', 'Description', 'Blog Content'])
output_lock = threading.Lock()

# === Retryable OpenAI call ===
@retry(wait=wait_random_exponential(min=4, max=10), stop=stop_after_attempt(10))
def completion_with_backoff(**kwargs):
    try:
        return openai.ChatCompletion.create(**kwargs)
    except openai.error.InvalidRequestError as e:
        print(f"❌ Invalid request: {e}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise

# === Retryable Shopify POST ===
@retry(wait=wait_random_exponential(min=4, max=10), stop=stop_after_attempt(10), retry=retry_if_exception_type(requests.exceptions.RequestException))
def create_shopify_post(payload):
    url = f"{store_address}/blogs/{blog_id}/articles.json"
    response = requests.post(url, headers=headers, data=json.dumps(payload), auth=(api_key, password))

    if response.status_code == 201:
        print(f"✅ Created post ID: {response.json()['article']['id']}")
    else:
        print(f"❌ Shopify error [{response.status_code}]: {response.text}")
        response.raise_for_status()

# === Generate a single blog post ===
@retry(wait=wait_random_exponential(min=4, max=10), stop=stop_after_attempt(10))
def generate_blog_post(row):
    url_slug = row['URL Slug']
    meta_title = row['Meta Title']
    description = row['Description of Page']

    print(f"🧠 Generating outline for: {url_slug}")
    outline_prompt = [
        {"role": "system", "content": "You are an essay-writing assistant who creates detailed outlines for essays. Always write at least 15 points."},
        {"role": "user", "content": f"Create an outline for an essay about {meta_title} with at least 15 titles."}
    ]
    outline_response = completion_with_backoff(model="gpt-4", messages=outline_prompt, max_tokens=1024, temperature=0.2)
    essay_outline = outline_response['choices'][0]['message']['content']

    blog_prompt = [
        {"role": "system", "content": f"Internal links are VITAL for SEO. Use a MAXIMUM of 5 internal links per article. Write full articles only (no placeholders). Use outline: {essay_outline}. Output in HTML. Each heading must have 3 paragraphs and include a table or list with borders. Use a fun tone. After article, add an FAQ section and <script> FAQPage schema."},
        {"role": "user", "content": "Never leave an article incomplete. Only use contextually relevant internal links from this list: /suit-basics/, /suit-fit/, /how-to-wear-a-suit/, /how-to-measure/, /30-suit-basics/, /button-rules/, /suit-styles/, /how-to-clean/, /dress-pants-fit/, /suit-cuts/, /differences-in-suit-cuts/, /classic-fit-suit/, /slim-fit-suit/, /modern-fit-suit/, /three-piece-suit/, /double-breasted-suit/, /suit-vs-tuxedo/, /how-to-wear-a-tuxedo/, /blue-tuxedo/, /tuxedo-shirt/, /best-affordable-tuxedos/, /formal-attire/, /wedding-attire/, /black-tie/, /semi-formal/, /cocktail-attire/, /business-professional/, /job-interview/, /smart-casual/, /business-casual/, /funeral-attire/, /suit-color/, /color-combinations/, /blazer-trousers/, /dress-shirt-fit/, /how-to-wear-a-dress-shirt/, /dress-shirt-sizes/, /shirt-colors/, /best-dress-shirts/, /shirt-and-tie/, /ties-guide/, /bow-ties/, /match-the-watch/, /dress-shoes-styles/, /pocket-square/, /belts-guide/, /how-to-wear-a-belt/, /cufflinks/, /tie-clip/, /suspenders/, /sunglasses/, /suit-fabrics/, /wool/, /cotton/, /cashmere/, /velvet/, /linen/, /seersucker/, /tweed/, /polyester/, /sharkskin/"}
    ]

    print(f"✍️ Generating content for: {url_slug}")
    content_response = completion_with_backoff(model="gpt-4", messages=blog_prompt, max_tokens=4000, temperature=0.2)
    blog_content = content_response['choices'][0]['message']['content']

    print(f"📦 Finished blog for: {url_slug}")
    result = {'URL Slug': url_slug, 'Meta Title': meta_title, 'Description': description, 'Blog Content': blog_content}

    with output_lock:
        global output_df
        output_df = pd.concat([output_df, pd.DataFrame([result])], ignore_index=True)
        output_df.to_csv('output.csv', index=False)
        print(f"💾 Saved blog post to output.csv for: {url_slug}")

    # Send to Shopify
    shopify_payload = {
        "article": {
            "title": meta_title,
            "author": author,
            "tags": "Blog Post, OpenAI",
            "body_html": blog_content
        }
    }

    print(f"🚀 Creating Shopify article for: {url_slug}")
    create_shopify_post(shopify_payload)

# === Main entry point ===
def main():
    if not os.path.exists("input.csv"):
        raise FileNotFoundError("❌ 'input.csv' not found.")

    df = pd.read_csv("input.csv")
    if df.empty:
        raise ValueError("❌ 'input.csv' is empty.")

    records = df.to_dict(orient='records')

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(generate_blog_post, row) for row in records]
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"❌ Error during blog generation: {e}")

if __name__ == "__main__":
    main()
