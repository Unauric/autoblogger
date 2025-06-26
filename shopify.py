import openai
import pandas as pd
import requests
import os
import base64
import time
from tqdm import tqdm
import concurrent.futures
import threading
import backoff
import json
from concurrent.futures import ThreadPoolExecutor

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

# ==============================
# 🔐 API Keys & Sanity Checks
# ==============================

openai.api_key = os.getenv("YOUR_OPEN_AI_KEY")
if not openai.api_key:
    raise EnvironmentError("❌ Missing OpenAI API key. Please set YOUR_OPEN_AI_KEY in environment.")
print(f"🔑 OpenAI API key loaded: {openai.api_key[:5]}...")

api_key = os.getenv("YOUR_API_KEY")
password = os.getenv("YOUR_SHOPIFY_PASSWORD")
store_address = os.getenv("YOUR_STORE_ID")
blog_id = os.getenv("YOUR_BLOG_ID")
author = "Wine expert"

for var, val in {
    "YOUR_API_KEY": api_key,
    "YOUR_SHOPIFY_PASSWORD": password,
    "YOUR_STORE_ID": store_address,
    "YOUR_BLOG_ID": blog_id
}.items():
    if not val:
        raise EnvironmentError(f"❌ Missing {var}. Check your environment variables.")

headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
}

output_df = pd.DataFrame(columns=['URL Slug', 'Meta Title', 'Description', 'Blog Content', 'Featured Image'])
output_lock = threading.Lock()

# ==============================
# 📡 OpenAI Retry Wrapper
# ==============================

@retry(wait=wait_random_exponential(min=4, max=10), stop=stop_after_attempt(10))
def completion_with_backoff(**kwargs):
    print("📡 Calling OpenAI ChatCompletion...")
    try:
        return openai.ChatCompletion.create(request_timeout=60, **kwargs)
    except openai.error.InvalidRequestError as e:
        print(f"❌ InvalidRequestError: {e}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise

# ==============================
# 📝 Shopify Blog Post Creation
# ==============================

@retry(wait=wait_random_exponential(min=4, max=10), stop=stop_after_attempt(10),
       retry=retry_if_exception_type(requests.exceptions.RequestException))
def create_shopify_post(payload):
    print("🚀 Sending request to Shopify...")
    response = requests.post(
        f'{store_address}/blogs/{blog_id}/articles.json',
        headers=headers,
        data=json.dumps(payload),
        auth=(api_key, password)
    )
    if response.status_code == 201:
        print(f"✅ Created post with ID: {response.json()['article']['id']}")
    else:
        print(f"❌ Shopify error: {response.status_code} - {response.content}")
        response.raise_for_status()

# ==============================
# 🧠 Blog Generator
# ==============================

def generate_blog_post(row):
    try:
        url_slug = row['URL Slug']
        meta_title = row['Meta Title']
        description = row['Description of Page']

        print(f"🧠 Generating outline for: {url_slug}")
        outline_prompt = [
            {
                "role": "system",
                "content": "You are an essay-writing assistant who creates detailed outlines for essays. Always write at least 15 points."
            },
            {
                "role": "user",
                "content": f"Create an outline for an essay about {meta_title} with at least 15 titles."
            }
        ]
        print(f"🔁 Sending prompt to OpenAI for: {url_slug}")
        outline_response = completion_with_backoff(
            model="gpt-4",
            messages=outline_prompt,
            max_tokens=1024,
            temperature=0.2
        )
        print(f"✅ Received outline for: {url_slug}")
        essay_outline = outline_response['choices'][0]['message']['content']

        conversation = [
            {
                "role": "system",
                "content": (
                    f"Internal links are VITAL for SEO. Use a max of 5 internal links, contextually, throughout the article. "
                    f"NEVER USE PLACEHOLDERS. Write in HTML. Each heading should have 3 paragraphs and at least 1 list or table with borders. "
                    f"Use these internal links: /suit-basics/, /how-to-wear-a-suit/, /how-to-measure/, /suit-fit/, /blazer-trousers/..."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Use the outline: {essay_outline}. Write full article. Use fun tone. Each section should have 3 paragraphs and a table or list. "
                    f"Include FAQ at the end, and wrap structured data with <script>...</script>."
                )
            }
        ]

        print(f"✍️ Generating full blog content for: {url_slug}")
        content_response = completion_with_backoff(
            model="gpt-4",
            messages=conversation,
            max_tokens=4500,
            temperature=0.2
        )
        blog_content = content_response['choices'][0]['message']['content']
        print(f"✅ Blog generated for: {url_slug}")

        result = {
            'URL Slug': url_slug,
            'Meta Title': meta_title,
            'Description': description,
            'Blog Content': blog_content,
        }

        with output_lock:
            global output_df
            output_df = pd.concat([output_df, pd.DataFrame([result])], ignore_index=True)
            output_df.to_csv('output.csv', index=False)
            print(f"💾 Saved blog for: {url_slug}")

        payload = {
            "article": {
                "title": meta_title,
                "author": author,
                "tags": "Blog Post, OpenAI",
                "body_html": blog_content
            }
        }

        print(f"📤 Posting to Shopify: {url_slug}")
        create_shopify_post(payload)

    except Exception as e:
        print(f"❌ Error for {url_slug}: {e}")

# ==============================
# 🚀 Main Entry
# ==============================

def main():
    print("📥 Reading input.csv...")
    df = pd.read_csv('input.csv')

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(generate_blog_post, row) for _, row in df.iterrows()]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print(f"⚠️ Thread failed: {e}")

if __name__ == "__main__":
    main()
