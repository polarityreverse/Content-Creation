import os
import json
import time
import requests
import logging
import random
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from googleapiclient.http import MediaFileUpload
from google import genai
from google.genai import types

from utils.schema import flowstate
from utils.youtube_auth import get_youtube_client
from utils.sheets import get_worksheet
from config import (
    OUTPUT_DIR, INSTA_ACCESS_TOKEN, INSTA_ACCOUNT_ID, 
    GEMINI_API_KEY_1, VIDEO_METADATA_GENERATION_MODEL
)

# Set up production logging
logger = logging.getLogger(__name__)

# --- HELPER 1: Metadata Generator ---
def get_llm_metadata(topic):
    """Generates viral metadata with a strict Format Guard to prevent strategy drift."""
    client = genai.Client(api_key=GEMINI_API_KEY_1)
    
    # We move the task to the very end of the prompt (the 'recency effect' for LLMs)
    prompt = f"""
Act as a Senior YouTube Strategist for 'Zeteon'.
Topic: '{topic}'

YOUR TASK:
Rewrite the topic into a high-retention, curiosity-driven YouTube title that includes #Shorts.
The title must create an immediate “I need to know this” reaction.

CONTENT CREATION RULES:
- The YouTube title must feel like a scientific mystery, paradox, or hidden mechanism being revealed.
- The description must be cinematic, concise, and focused on the core scientific hook.
- Tags must be platform-relevant, SEO-friendly, and based on real viewer search intent.
- The pinned comment must encourage engagement through a question or surprising fact.
- The Instagram caption must be short, punchy, and optimized for retention.
- Instagram hashtags must be niche-relevant, not generic spam tags.

STRICT OUTPUT FORMAT:
You must return ONLY a JSON object with two top-level keys: 'youtube' and 'insta'.
Do not add explanations, markdown, or text outside the JSON.

REQUIRED JSON STRUCTURE:
{{
    "youtube": {{
        "title": "string",
        "description": "string",
        "tags": ["#tag1", "#tag2"],
        "pinned_comment": "string"
    }},
    "insta": {{
        "caption": "string",
        "hashtags": ["#tag1", "#tag2"]
    }}
}}

ADDITIONAL SCHEMA RULES:
- All fields must be non-empty strings or arrays of strings.
- Tags and hashtags must be lowercase and formatted with leading '#'.
- Do not introduce new keys or rename existing ones.
- Do not include emojis unless they enhance clarity.
"""

    try:
        response = client.models.generate_content(
            model=VIDEO_METADATA_GENERATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                # This is the most important setting for AWS production
                response_mime_type="application/json",
                temperature=0.2, # Lower temperature reduces 'creative' drift
            )
        )
        
        data = json.loads(response.text.strip())

        # Flatten list if Gemini wraps it
        if isinstance(data, list):
            data = data[0]

        # VALIDATION & LOGGING
        if 'youtube' in data and 'insta' in data:
            logger.info(f"✅ Metadata successfully generated for: {topic}")
            return data
        else:
            # If it fails, we log the bad keys to CloudWatch for debugging
            logger.error(f"❌ LLM Drift Detected. Received keys: {list(data.keys())}")
            return None

    except Exception as e:
        logger.error(f"❌ Metadata Gen Error: {str(e)}")
        return None

# --- HELPER 2: YouTube Uploader (Modified to return Video Link) ---
def upload_to_youtube(video_path, metadata, row_idx):
    """Uploads to YouTube and returns (status, video_url)."""
    try:
        youtube = get_youtube_client()
        
        body = {
            'snippet': {
                'title': metadata['title'][:100],
                'description': metadata['description'],
                'tags': metadata.get('tags', []),
                'categoryId': '28' 
            },
            'status': {'privacyStatus': 'public', 'selfDeclaredMadeForKids': False}
        }
        
        media = MediaFileUpload(video_path, chunksize=1024*1024, resumable=True)
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
        response = request.execute()
        
        video_id = response['id']
        video_url = f"https://www.youtube.com/shorts/{video_id}"
        logger.info(f"🎬 YouTube Uploaded: {video_url}")

        # Wait for Processing (Rendering Wait)
        for _ in range(20): 
            time.sleep(45)
            status_res = youtube.videos().list(part="processingDetails", id=video_id).execute()
            items = status_res.get("items", [])
            if not items: break
            
            p_status = items[0].get("processingDetails", {}).get("processingStatus")
            if p_status == "succeeded":
                logger.info("✅ YouTube: Processing complete.")
                break
        
        # Add Pinned Comment
        youtube.commentThreads().insert(
            part="snippet",
            body={
                "snippet": {
                    "videoId": video_id,
                    "topLevelComment": {"snippet": {"textOriginal": metadata['pinned_comment']}}
                }
            }
        ).execute()
        
        return "SUCCESS", video_url
    except Exception as e:
        logger.error(f"❌ YouTube Error: {e}")
        return "FAILED", None

# --- HELPER 3: Instagram Uploader ---
def upload_to_insta(video_url, metadata):
    """Uploads to Instagram via Meta Graph API."""
    base_url = f"https://graph.facebook.com/v19.0/{INSTA_ACCOUNT_ID}"
    caption = f"{metadata['caption']}\n\n{' '.join(metadata['hashtags'])}"
    
    try:
        res = requests.post(f"{base_url}/media", data={
            'video_url': video_url, 
            'caption': caption,
            'media_type': 'REELS', 
            'access_token': INSTA_ACCESS_TOKEN
        }, timeout=60)
        
        container_id = res.json().get('id')
        if not container_id: return "FAILED"

        # Polling for finish
        for i in range(30):
            time.sleep(20)
            status_res = requests.get(f"https://graph.facebook.com/v19.0/{container_id}", 
                                      params={'fields': 'status_code', 'access_token': INSTA_ACCESS_TOKEN}).json()
            if status_res.get('status_code') == 'FINISHED':
                publish_res = requests.post(f"{base_url}/media_publish", 
                                            data={'creation_id': container_id, 'access_token': INSTA_ACCESS_TOKEN})
                if "id" in publish_res.json():
                    return "SUCCESS"
        return "FAILED"
    except Exception as e:
        logger.error(f"⚠️ Insta Error: {e}")
        return "ERROR"

# --- MAIN EXECUTION NODE ---
def video_upload_node(state: flowstate) -> flowstate:
    """Node 5: Final Upload and Persistence (including Column I for YT Link)."""
    row_idx = state['row_index']
    topic = state['idea']
    
    log_extra = {"row_index": row_idx, "topic": topic}
    logger.info(f"🚀 Starting Zeteon Final Sync", extra=log_extra)

    try:
        worksheet = get_worksheet("ideas")
        final_video_path = os.path.join(OUTPUT_DIR, f"Video_Row_{row_idx}.mp4")
        github_video_uri = worksheet.cell(row_idx, 4).value 

        # Current Status Check
        youtube_status = worksheet.cell(row_idx, 5).value
        insta_status = worksheet.cell(row_idx, 7).value

        if youtube_status != "UPLOADED" or insta_status != "UPLOADED":
            meta = get_llm_metadata(topic)
            if not meta: return state

            # 1. YouTube Upload + Link Save (Column I is Index 9)
            if youtube_status != "UPLOADED":
                status, yt_link = upload_to_youtube(final_video_path, meta['youtube'], row_idx)
                if status == "SUCCESS":
                    worksheet.update_cell(row_idx, 5, "UPLOADED")
                    worksheet.update_cell(row_idx, 6, json.dumps(meta['youtube']))
                    worksheet.update_cell(row_idx, 9, yt_link) # <--- SAVING TO COLUMN I
                    logger.info(f"✔️ YouTube Link saved to Col I: {yt_link}")

            # 2. Instagram Upload
            if insta_status != "UPLOADED":
                if upload_to_insta(github_video_uri, meta['insta']) == "SUCCESS":
                    worksheet.update_cell(row_idx, 7, "UPLOADED")
                    worksheet.update_cell(row_idx, 8, json.dumps(meta['insta']))
                    logger.info(f"✔️ Instagram status updated to UPLOADED")

        state["isvideouploaded"] = True
        logger.info(f"✅ Row {row_idx} fully synchronized.")

    except Exception as e:
        logger.error(f"Critical Node 5 Failure: {e}", exc_info=True)
        state["isvideouploaded"] = False

    return state