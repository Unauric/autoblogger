import os
import openai
import pandas as pd
import requests
import base64
import time
from tqdm import tqdm
import concurrent.futures
import threading
import backoff
import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)
import random

# ==============================
# 🔐 API Keys & Sanity Checks
# ==============================

client = OpenAI(api_key=os.getenv("YOUR_OPEN_AI_KEY"))
if not client.api_key:
    raise EnvironmentError("❌ Missing OpenAI API key. Please set YOUR_OPEN_AI_KEY in environment.")
print(f"🔑 OpenAI API key loaded: {client.api_key[:5]}...")

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

output_df = pd.DataFrame(columns=['Topic', 'URL Slug', 'Meta Title', 'Meta Description', 'Keywords Used', 'Tone', 'Formality', 'Blog Content'])
output_lock = threading.Lock()

# ==============================
# 🍷 Wine Keywords & Topics Pool
# ==============================

WINE_KEYWORD_SETS = [
    ["bordeaux", "tannins", "aging", "terroir"],
    ["champagne", "celebration", "bubbles", "pairing"],
    ["tuscany", "sangiovese", "harvest", "vineyard"],
    ["sommelier", "tasting", "notes", "complexity"],
    ["vintage", "collection", "investment", "cellar"],
    ["organic", "biodynamic", "natural", "sustainable"],
    ["food pairing", "cheese", "chocolate", "cuisine"],
    ["winemaking", "fermentation", "oak", "tradition"],
    ["terroir", "climate", "soil", "geography"],
    ["decanting", "serving", "temperature", "glassware"],
    ["wine regions", "appellations", "classifications", "heritage"],
    ["grape varieties", "characteristics", "flavor profiles", "aromatics"],
    ["wine storage", "cellar management", "collecting", "preservation"],
    ["seasonal wines", "holiday pairing", "weather", "occasions"],
    ["emerging regions", "new world", "innovation", "discovery"]
]

WINE_TONES = ["sophisticated", "educational", "passionate", "accessible", "expert"]
WINE_FORMALITIES = ["professional", "conversational", "refined"]

# ==============================
# 📱 OpenAI Retry Wrapper
# ==============================

@retry(wait=wait_random_exponential(min=4, max=10), stop=stop_after_attempt(10))
def completion_with_backoff(**kwargs):
    print("📱 Calling OpenAI ChatCompletion...")
    try:
        return client.chat.completions.create(**kwargs)
    except openai.OpenAIError as e:
        print(f"❌ OpenAI error: {e}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise

# ==============================
# 📜 Shopify Blog Post Creation
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
# 🍷 High-Quality Wine Blog Generator
# ==============================

def generate_wine_blog_post():
    """
    Generate a premium, high-quality wine blog post where ChatGPT acts as a wine expert
    and freely decides the topic based on wine keywords and expertise.
    """
    try:
        # Randomly select keywords, tone, and formality for variety
        keywords = random.choice(WINE_KEYWORD_SETS)
        tone = random.choice(WINE_TONES)
        formality = random.choice(WINE_FORMALITIES)
        keywords_str = ", ".join(keywords)
        
        print(f"🍷 Generating premium wine content with keywords: {keywords_str}")
        print(f"🎭 Using tone: {tone}, formality: {formality}")
        
        # Stage 1: Expert wine topic creation and detailed planning
        topic_prompt = [
            {"role": "system", "content": (
                f"You are a world-renowned wine expert, sommelier, and wine writer with decades of experience. "
                f"You have extensive knowledge of wine regions, grape varieties, winemaking techniques, food pairings, "
                f"wine history, and the cultural significance of wine. You write for wine enthusiasts, collectors, "
                f"and anyone passionate about wine culture. Your expertise spans from technical winemaking knowledge "
                f"to the art of wine appreciation and the business of wine.\n\n"
                f"Based on the given wine-related keywords, create a sophisticated, engaging blog post concept that "
                f"showcases deep wine knowledge and provides genuine value to wine lovers.\n\n"
                f"Tone: {tone}\n"
                f"Formality: {formality}\n"
                f"Target audience: Wine enthusiasts, collectors, and educated consumers"
            )},
            {"role": "user", "content": (
                f"Using these wine-related keywords as inspiration: {keywords_str}\n\n"
                f"Please create a premium wine blog post concept with:\n"
                f"1. A sophisticated, engaging topic that demonstrates deep wine expertise\n"
                f"2. A compelling SEO-friendly title (55-60 characters)\n"
                f"3. A clean URL slug (lowercase, hyphens)\n"
                f"4. An enticing meta description (150-160 characters)\n"
                f"5. A comprehensive outline with 15-20 detailed sections that cover:\n"
                f"   - Technical wine knowledge\n"
                f"   - Practical advice for wine lovers\n"
                f"   - Historical or cultural context\n"
                f"   - Specific recommendations\n"
                f"   - Expert insights and personal experience\n\n"
                f"Format your response as:\n"
                f"TOPIC: [your expert wine topic]\n"
                f"TITLE: [your SEO title]\n"
                f"SLUG: [your-url-slug]\n"
                f"META: [your meta description]\n"
                f"OUTLINE:\n[your detailed 15-20 point outline with brief descriptions]"
            )}
        ]
        
        print(f"🧠 Creating expert wine topic and comprehensive outline...")
        topic_response = completion_with_backoff(
            model="gpt-4",
            messages=topic_prompt,
            max_tokens=2000,  # Increased for detailed outline
            temperature=0.7
        )
        
        topic_content = topic_response.choices[0].message.content
        
        # Parse the response to extract components
        lines = topic_content.split('\n')
        topic = ""
        title = ""
        slug = ""
        meta = ""
        outline = ""
        
        outline_started = False
        for line in lines:
            if line.startswith("TOPIC:"):
                topic = line.replace("TOPIC:", "").strip()
            elif line.startswith("TITLE:"):
                title = line.replace("TITLE:", "").strip()
            elif line.startswith("SLUG:"):
                slug = line.replace("SLUG:", "").strip()
            elif line.startswith("META:"):
                meta = line.replace("META:", "").strip()
            elif line.startswith("OUTLINE:"):
                outline_started = True
            elif outline_started:
                outline += line + "\n"
        
        print(f"🍷 Generated expert topic: {topic}")
        print(f"📝 Title: {title}")
        print(f"🔗 Slug: {slug}")
        
        # Stage 2: Generate premium, comprehensive wine content
        content_conversation = [
            {"role": "system", "content": (
                f"You are a master sommelier and acclaimed wine writer creating premium content for a sophisticated wine blog. "
                f"Your writing should demonstrate exceptional wine expertise while being accessible to wine enthusiasts. "
                f"Write with authority, passion, and deep knowledge. Include specific wine recommendations, technical insights, "
                f"personal anecdotes from your wine experience, and practical advice.\n\n"
                f"CONTENT REQUIREMENTS:\n"
                f"- Write in {tone} tone with {formality} formality\n"
                f"- Each major section should have 4-5 well-developed, substantial paragraphs\n"
                f"- Include specific wine recommendations with vintages, producers, and tasting notes\n"
                f"- Add technical winemaking details where relevant\n"
                f"- Include food pairing suggestions with detailed explanations\n"
                f"- Use wine terminology accurately and explain complex concepts\n"
                f"- Add tables, lists, and visual elements for engagement\n"
                f"- Include a comprehensive FAQ section with expert answers\n"
                f"- Add structured data schema for wine content\n"
                f"- Write in HTML format with proper headers (h2, h3), paragraphs, lists, tables\n"
                f"- Minimum 3000-4000 words for comprehensive coverage\n"
                f"- Internal links should be wine-related: /wine-regions/, /grape-varieties/, /wine-tasting/, /food-pairing/, /wine-storage/, /sommelier-tips/"
            )},
            {"role": "user", "content": (
                f"Write a comprehensive, premium wine blog post about: {topic}\n"
                f"Title: {title}\n"
                f"Use this expert outline: {outline}\n\n"
                f"Create an exceptional piece that showcases deep wine expertise and provides immense value to readers. "
                f"Include specific wine recommendations, technical insights, personal sommelier experiences, "
                f"detailed tasting notes, food pairing suggestions, and practical advice.\n\n"
                f"Structure with:\n"
                f"- Engaging introduction that hooks wine lovers\n"
                f"- 15-20 comprehensive sections following your outline\n"
                f"- Specific wine recommendations throughout\n"
                f"- Technical details and winemaking insights\n"
                f"- Food pairing sections with detailed explanations\n"
                f"- Expert tips and personal anecdotes\n"
                f"- Comprehensive FAQ section (10+ questions)\n"
                f"- Conclusion with key takeaways and call-to-action\n"
                f"- Structured data schema in <script> tags\n\n"
                f"Make this the definitive guide on the topic that wine enthusiasts will bookmark and reference."
            )}
        ]
        
        print(f"✍️ Generating premium wine content (this may take a moment for quality)...")
        content_response = completion_with_backoff(
            model="gpt-4",
            messages=content_conversation,
            max_tokens=6000,  # Increased significantly for comprehensive content
            temperature=0.3
        )
        
        blog_content = content_response.choices[0].message.content
        
        print(f"✅ Premium wine blog generated successfully!")
        print(f"📊 Content length: ~{len(blog_content)} characters")
        
        # Create result dictionary
        result = {
            'Topic': topic,
            'URL Slug': slug,
            'Meta Title': title,
            'Meta Description': meta,
            'Keywords Used': keywords_str,
            'Tone': tone,
            'Formality': formality,
            'Blog Content': blog_content,
        }
        
        # Save to CSV
        with output_lock:
            global output_df
            output_df = pd.concat([output_df, pd.DataFrame([result])], ignore_index=True)
            output_df.to_csv('wine_output.csv', index=False)
            print(f"📂 Saved premium wine blog: {slug}")
        
        # Post to Shopify
        payload = {
            "article": {
                "title": title,
                "author": author,
                "tags": f"Wine Expert, Premium Content, {keywords_str.replace(', ', ', Wine ')}, {tone.title()}",
                "body_html": blog_content,
                "summary": meta
            }
        }
        
        print(f"📤 Posting premium wine blog to Shopify: {slug}")
        create_shopify_post(payload)
        
        return result
        
    except Exception as e:
        print(f"❌ Error generating premium wine blog: {e}")
        return None

# ==============================
# ✨ Main Entry - Background Worker
# ==============================

def main():
    """
    Background worker that generates one premium wine blog post every 2 days
    """
    print("🍷 Starting Wine Blog Background Worker...")
    print("📅 Generating one premium article every 2 days...")
    
    while True:
        try:
            print(f"\n🚀 Starting new premium wine blog generation cycle...")
            print(f"⏰ Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Generate one premium wine blog post
            result = generate_wine_blog_post()
            
            if result:
                print(f"✅ Successfully generated and published: {result['Meta Title']}")
                print(f"🏷️ Keywords used: {result['Keywords Used']}")
                print(f"🎭 Style: {result['Tone']} tone, {result['Formality']} formality")
            else:
                print("❌ Failed to generate wine blog post")
            
            # Wait 2 days (48 hours = 172800 seconds)
            wait_time = 2 * 24 * 60 * 60  # 2 days in seconds
            print(f"💤 Waiting 2 days until next article generation...")
            print(f"⏰ Next article scheduled for: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + wait_time))}")
            
            time.sleep(wait_time)
            
        except KeyboardInterrupt:
            print("\n🛑 Background worker stopped by user")
            break
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
            print("⏰ Waiting 30 minutes before retry...")
            time.sleep(1800)  # Wait 30 minutes on error

if __name__ == "__main__":
    main()
