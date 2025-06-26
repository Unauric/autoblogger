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
# 🍷 EXPANDED Wine Keywords & Topics Pool (Wine Map Focused)
# ==============================

WINE_KEYWORD_SETS = [
    # Classic European Wine Regions (Map-focused)
    ["burgundy terroir", "côte d'or", "pinot noir", "vineyard mapping"],
    ["bordeaux classifications", "left bank", "cabernet sauvignon", "wine geography"],
    ["champagne region", "reims", "epernay", "sparkling wine origins"],
    ["tuscany landscapes", "chianti classico", "sangiovese", "italian wine regions"],
    ["rioja valleys", "tempranillo", "spanish wine heritage", "denominación de origen"],
    ["rhône valley", "northern rhône", "southern rhône", "syrah terroir"],
    ["loire valley", "sauvignon blanc", "chenin blanc", "french wine diversity"],
    ["piedmont wines", "barolo", "barbaresco", "nebbiolo territory"],
    ["douro valley", "port wine", "portuguese vineyards", "river valley viticulture"],
    ["mosel region", "riesling slopes", "german wine culture", "steep vineyard geography"],
    
    # Wine Geography & Terroir Exploration
    ["terroir mapping", "soil composition", "climate zones", "microclimate influence"],
    ["altitude viticulture", "mountain vineyards", "elevation effects", "hillside terroir"],
    ["coastal wine regions", "maritime influence", "ocean proximity", "sea breeze effects"],
    ["volcanic soils", "mineral wines", "geological impact", "terroir expression"],
    ["limestone terroir", "chalky soils", "mineral complexity", "geological influence"],
    ["river valley wines", "alluvial soils", "water influence", "riparian viticulture"],
    ["continental climate", "temperature variation", "seasonal changes", "weather patterns"],
    ["mediterranean viticulture", "dry climate", "sun exposure", "heat management"],
    ["alpine wine regions", "high altitude", "cool climate", "mountain terroir"],
    ["desert viticulture", "extreme conditions", "irrigation techniques", "arid climate"],
    
    # Wine Education & Map Learning
    ["wine region education", "sommelier training", "WSET studies", "wine knowledge"],
    ["appellation systems", "wine laws", "quality classifications", "regional regulations"],
    ["grape variety mapping", "varietal distribution", "indigenous grapes", "local varieties"],
    ["wine route planning", "vineyard visits", "wine tourism", "cellar door experiences"],
    ["harvest timing", "seasonal variations", "vintage conditions", "weather impact"],
    ["wine atlas", "cartography", "geographic wine study", "regional boundaries"],
    ["denomination origins", "AOC system", "DOC classifications", "quality control"],
    ["wine heritage", "historical regions", "ancient vineyards", "traditional methods"],
    ["cultural wine traditions", "local customs", "regional practices", "wine rituals"],
    ["wine language", "tasting terminology", "descriptive vocabulary", "sensory analysis"],
    
    # Italian Wine Regions (Italy Map Focus)
    ["piedmont excellence", "alba wines", "italian wine nobility", "northwest italy"],
    ["veneto wines", "prosecco region", "amarone production", "venetian viticulture"],
    ["tuscany icons", "brunello montalcino", "vino nobile", "central italy wines"],
    ["sicilian renaissance", "etna wines", "island viticulture", "volcanic vineyards"],
    ["umbria treasures", "sagrantino", "central italian gems", "hill country wines"],
    ["marche discoveries", "verdicchio", "adriatic influence", "coastal italian wines"],
    ["abruzzo potential", "montepulciano", "mountain wines", "eastern italian regions"],
    ["friuli venezia giulia", "white wine excellence", "northeastern italy", "border regions"],
    ["campania revival", "aglianico", "southern italian heritage", "ancient grape varieties"],
    ["puglia emergence", "primitivo", "heel of italy", "mediterranean influence"],
    
    # French Wine Regions (France Map Focus)
    ["alsace uniqueness", "german influence", "aromatic whites", "rhine valley"],
    ["beaujolais crus", "gamay expression", "carbonic maceration", "granite soils"],
    ["jura secrets", "oxidative wines", "yellow wine", "mountain viticulture"],
    ["languedoc revolution", "southern france", "new world techniques", "bulk to boutique"],
    ["provence rosé", "mediterranean lifestyle", "pink wine culture", "southern french charm"],
    ["corsica wines", "island terroir", "indigenous varieties", "mountain meets sea"],
    ["savoy wines", "alpine viticulture", "mountain wines", "high altitude vineyards"],
    ["southwest france", "cahors malbec", "regional diversity", "ancient wine regions"],
    ["northern rhône precision", "syrah mastery", "steep slopes", "granite terroir"],
    ["southern rhône blends", "grenache dominance", "galets stones", "warm climate"],
    
    # Spanish Wine Regions (Spain Map Focus)
    ["rías baixas", "albariño", "galician wines", "atlantic influence"],
    ["priorat power", "llicorella slate", "concentrated wines", "catalan excellence"],
    ["ribera del duero", "tempranillo altitude", "high plains", "continental climate"],
    ["jerez tradition", "sherry production", "solera system", "andalusian heritage"],
    ["valencia diversity", "monastrell", "bobal", "eastern spanish coast"],
    ["navarra innovation", "modern winemaking", "diverse varieties", "northern spain"],
    ["castilla la mancha", "central plateau", "bulk wine region", "spanish heartland"],
    ["bierzo discovery", "mencía", "northwest spain", "slate soils"],
    ["jumilla strength", "monastrell mastery", "warm climate", "southeastern spain"],
    ["canary islands", "volcanic wines", "atlantic islands", "unique terroir"],
    
    # Wine Collecting & Display
    ["wine map collecting", "wall art", "home decor", "wine enthusiast gifts"],
    ["cellar design", "wine storage", "collection display", "proper preservation"],
    ["wine investment", "vintage appreciation", "collectible wines", "market value"],
    ["wine library", "reference materials", "educational resources", "wine books"],
    ["tasting room setup", "home wine bar", "entertaining space", "wine presentation"],
    ["wine gifts", "sommelier presents", "wine lover accessories", "thoughtful giving"],
    ["wine memorabilia", "collectible items", "wine artifacts", "vintage pieces"],
    ["wine photography", "label collection", "wine imagery", "visual wine culture"],
    ["wine travel", "vineyard visits", "wine destinations", "terroir exploration"],
    ["wine education tools", "learning resources", "study materials", "wine reference"],
    
    # Seasonal & Occasion-Based
    ["spring wine selections", "fresh beginnings", "light wines", "seasonal transitions"],
    ["summer wine pairings", "warm weather", "chilled wines", "outdoor dining"],
    ["autumn harvest", "vintage celebrations", "seasonal flavors", "harvest festivals"],
    ["winter wine comfort", "rich reds", "holiday traditions", "cozy evenings"],
    ["holiday wine gifts", "festive selections", "celebration wines", "special occasions"],
    ["valentine's day wines", "romantic selections", "love and wine", "intimate dining"],
    ["easter wine traditions", "spring celebrations", "renewal themes", "family gatherings"],
    ["thanksgiving pairings", "gratitude wines", "harvest celebration", "family feasts"],
    ["christmas wine selections", "holiday spirits", "gift giving", "winter warmth"],
    ["new year champagne", "celebration bubbles", "fresh starts", "midnight toasts"],
    
    # Wine & Food Culture
    ["regional cuisine pairing", "local food traditions", "terroir harmony", "cultural combinations"],
    ["cheese and wine maps", "artisanal pairings", "dairy traditions", "flavor matching"],
    ["chocolate wine pairings", "dessert wines", "sweet combinations", "indulgent treats"],
    ["seafood wine matches", "coastal pairings", "fresh flavors", "maritime cuisine"],
    ["meat wine pairings", "protein matching", "hearty combinations", "carnivore choices"],
    ["vegetarian wine pairings", "plant-based dining", "garden fresh", "green cuisine"],
    ["spicy food wines", "heat management", "cooling wines", "international spices"],
    ["barbecue wine selections", "grilled flavors", "smoky notes", "outdoor cooking"],
    ["picnic wine choices", "portable wines", "casual dining", "outdoor enjoyment"],
    ["fine dining wines", "restaurant selections", "sommelier choices", "elegant pairings"],
    
    # Modern Wine Trends
    ["natural wine movement", "minimal intervention", "authentic expression", "pure terroir"],
    ["biodynamic viticulture", "holistic farming", "cosmic influence", "sustainable practices"],
    ["orange wines", "skin contact", "ancient techniques", "modern revival"],
    ["low alcohol wines", "health conscious", "lighter styles", "modern preferences"],
    ["sustainable winemaking", "environmental responsibility", "green practices", "eco-friendly"],
    ["urban wineries", "city production", "local sourcing", "metropolitan wine"],
    ["women in wine", "female winemakers", "gender equality", "industry change"],
    ["climate change impact", "adaptation strategies", "shifting regions", "future challenges"],
    ["technology in wine", "precision viticulture", "modern tools", "scientific approach"],
    ["wine marketing", "brand storytelling", "consumer engagement", "digital presence"],
    
    # Historical Wine Context
    ["ancient wine regions", "historical viticulture", "wine origins", "archaeological evidence"],
    ["monastic winemaking", "religious traditions", "abbey vineyards", "spiritual connection"],
    ["royal wine preferences", "noble selections", "court wines", "aristocratic taste"],
    ["wine trade history", "merchant routes", "commercial development", "economic impact"],
    ["phylloxera recovery", "replanting efforts", "rootstock solutions", "industry resilience"],
    ["prohibition impact", "wine industry disruption", "legal challenges", "recovery stories"],
    ["war and wine", "conflict effects", "vineyard destruction", "rebuilding efforts"],
    ["wine pioneers", "visionary winemakers", "industry leaders", "innovation stories"],
    ["vintage legends", "famous years", "exceptional harvests", "memorable wines"],
    ["wine scandals", "industry controversies", "quality issues", "trust rebuilding"],
    
    # Technical Wine Knowledge
    ["vineyard management", "canopy control", "yield optimization", "quality focus"],
    ["harvest techniques", "picking methods", "timing decisions", "quality preservation"],
    ["fermentation science", "yeast selection", "temperature control", "process management"],
    ["barrel aging", "oak influence", "maturation process", "flavor development"],
    ["blending artistry", "component selection", "harmony creation", "final assembly"],
    ["quality control", "testing procedures", "standards maintenance", "consistency assurance"],
    ["wine faults", "problem identification", "prevention methods", "quality issues"],
    ["storage conditions", "cellar management", "preservation techniques", "aging requirements"],
    ["serving protocols", "temperature control", "glassware selection", "presentation standards"],
    ["wine analysis", "chemical composition", "laboratory testing", "quality assessment"]
]

WINE_TONES = [
    "sophisticated", "educational", "passionate", "accessible", "expert", 
    "inspiring", "authoritative", "friendly", "professional", "enthusiastic",
    "scholarly", "conversational", "refined", "approachable", "knowledgeable"
]

WINE_FORMALITIES = [
    "professional", "conversational", "refined", "academic", "casual",
    "formal", "approachable", "technical", "friendly", "authoritative"
]

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
# 🍷 High-Quality Wine Blog Generator (Wine Map Focused)
# ==============================

def generate_wine_blog_post():
    """
    Generate a premium, high-quality wine blog post that naturally connects to wine maps
    and geographic wine education - perfect for Made for Wine's audience.
    """
    try:
        # Randomly select keywords, tone, and formality for variety
        keywords = random.choice(WINE_KEYWORD_SETS)
        tone = random.choice(WINE_TONES)
        formality = random.choice(WINE_FORMALITIES)
        keywords_str = ", ".join(keywords)
        
        print(f"🍷 Generating premium wine content with keywords: {keywords_str}")
        print(f"🎭 Using tone: {tone}, formality: {formality}")
        
        # Stage 1: Expert wine topic creation with wine map connection
        topic_prompt = [
            {"role": "system", "content": (
                f"You are a world-renowned wine expert, WSET-certified sommelier, and wine educator who specializes in "
                f"wine geography and terroir education. You create content for wine enthusiasts who are passionate about "
                f"understanding wine regions, appellations, and the geographic factors that influence wine character. "
                f"Your audience includes wine collectors, WSET students, sommeliers, and wine lovers who appreciate "
                f"educational wine maps and geographic wine knowledge.\n\n"
                f"You must create blog topics that naturally connect to wine geography, regional wine education, and "
                f"the value of understanding wine through maps and regional context. Think about how wine maps help people "
                f"understand terroir, plan wine tours, learn appellations, and deepen their wine knowledge.\n\n"
                f"Tone: {tone}\n"
                f"Formality: {formality}\n"
                f"Target audience: Wine map enthusiasts, geography-minded wine lovers, wine educators, and collectors"
            )},
            {"role": "user", "content": (
                f"Using these wine-related keywords as inspiration: {keywords_str}\n\n"
                f"Please create a premium wine blog post concept that subtly connects to wine geography and regional education. "
                f"The connection should feel natural - perhaps discussing how understanding wine regions enhances appreciation, "
                f"how geographic knowledge helps in wine selection, or how visual learning aids wine education.\n\n"
                f"Create:\n"
                f"1. A sophisticated topic that demonstrates deep wine expertise AND geographic awareness\n"
                f"2. An SEO-friendly title (55-60 characters) that hints at regional/geographic elements\n"
                f"3. A clean URL slug (lowercase, hyphens)\n"
                f"4. An enticing meta description (150-160 characters)\n"
                f"5. A comprehensive outline with 15-20 detailed sections including:\n"
                f"   - Regional wine knowledge and terroir insights\n"
                f"   - Geographic factors affecting wine character\n"
                f"   - Practical guidance for wine lovers\n"
                f"   - Educational elements about wine regions\n"
                f"   - Visual learning and map-related references where natural\n"
                f"   - Specific wine recommendations by region\n\n"
                f"Format your response as:\n"
                f"TOPIC: [your expert wine topic]\n"
                f"TITLE: [your SEO title]\n"
                f"SLUG: [your-url-slug]\n"
                f"META: [your meta description]\n"
                f"OUTLINE:\n[your detailed 15-20 point outline with brief descriptions]"
            )}
        ]
        
        print(f"🧠 Creating expert wine topic with geographic focus...")
        topic_response = completion_with_backoff(
            model="gpt-4",
            messages=topic_prompt,
            max_tokens=2000,
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
        
        # Stage 2: Generate premium, comprehensive wine content with geographic focus
        content_conversation = [
            {"role": "system", "content": (
                f"You are a master sommelier and acclaimed wine writer creating premium content for sophisticated wine enthusiasts "
                f"who value geographic wine knowledge. Your writing should demonstrate exceptional wine expertise while emphasizing "
                f"the importance of understanding wine regions, terroir, and geographic factors. You write for people who appreciate "
                f"wine maps, regional wine education, and the visual learning aspect of wine knowledge.\n\n"
                f"CONTENT REQUIREMENTS:\n"
                f"- Write in {tone} tone with {formality} formality\n"
                f"- Each major section should have 4-5 well-developed, substantial paragraphs\n"
                f"- Include specific wine recommendations with regions, appellations, and geographic context\n"
                f"- Add geographic and terroir details throughout\n"
                f"- Naturally mention the value of visual learning, regional understanding, and wine education\n"
                f"- Include references to wine regions, appellations, and geographic wine knowledge\n"
                f"- Add tables, lists, and visual elements for engagement\n"
                f"- Include a comprehensive FAQ section with expert answers\n"
                f"- Write in HTML format with proper headers (h2, h3), paragraphs, lists, tables\n"
                f"- Minimum 3500-4500 words for comprehensive coverage\n"
                f"- Internal links should be wine-related: /wine-regions/, /terroir-guide/, /wine-education/, /appellation-systems/, /wine-geography/, /regional-wine-styles/\n"
                f"- Subtly highlight the educational value of understanding wine geography and regions\n"
                f"- Where natural, mention how visual aids and maps enhance wine learning"
            )},
            {"role": "user", "content": (
                f"Write a comprehensive, premium wine blog post about: {topic}\n"
                f"Title: {title}\n"
                f"Use this expert outline: {outline}\n\n"
                f"Create an exceptional piece that showcases deep wine expertise with a focus on geographic wine knowledge. "
                f"Throughout the article, naturally emphasize the importance of understanding wine regions, terroir, and "
                f"geographic factors. Where appropriate, mention how visual learning aids (like wine maps) enhance wine education "
                f"and appreciation.\n\n"
                f"Structure with:\n"
                f"- Engaging introduction that hooks geography-minded wine lovers\n"
                f"- 15-20 comprehensive sections following your outline\n"
                f"- Regional wine recommendations with geographic context\n"
                f"- Terroir and geographic insights throughout\n"
                f"- Educational elements about wine regions and appellations\n"
                f"- Expert tips on using geographic knowledge for wine selection\n"
                f"- Natural mentions of wine education and visual learning benefits\n"
                f"- Comprehensive FAQ section (10+ questions)\n"
                f"- Conclusion emphasizing the value of geographic wine knowledge\n"
                f"- Structured data schema in <script> tags\n\n"
                f"Make this the definitive guide that wine enthusiasts will reference for both wine knowledge and geographic understanding."
            )}
        ]
        
        print(f"✍️ Generating premium wine content with geographic focus...")
        content_response = completion_with_backoff(
            model="gpt-4",
            messages=content_conversation,
            max_tokens=6000,
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
        
        # Post to Shopify with wine map relevant tags
        payload = {
            "article": {
                "title": title,
                "author": author,
                "tags": f"Wine Expert, Wine Geography, Regional Wine Education, {keywords_str.replace(', ', ', Wine ')}, {tone.title()}, Wine Maps",
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
    print("🍷 Starting Wine Blog Background Worker (Made for Wine Edition)...")
    print("📅 Generating one premium article every 2 days...")
    print("🗺️ Focus: Wine geography, regional education, and terroir knowledge")
    
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
