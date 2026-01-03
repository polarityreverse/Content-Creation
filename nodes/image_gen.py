import asyncio
import aiohttp
import random
import base64
import os
import shutil
import logging
from typing import Optional
from config import IMAGEN_IMAGE_GENERATION_API_URL_1, IMAGEN_IMAGE_GENERATION_API_URL_2, OUTPUT_DIR

# Set up production logging
logger = logging.getLogger(__name__)

async def generate_single_image_async(
    session: aiohttp.ClientSession, 
    prompt: str, 
    img_filename: str, 
    retries_per_url: int = 3
) -> Optional[str]:
    """
    Advanced Worker for AWS Production.
    Exhausts all retries on URL_1 before switching to URL_2.
    """
    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": "9:16",
            "outputMimeType": "image/png"
        }
    }
    
    urls_to_try = [IMAGEN_IMAGE_GENERATION_API_URL_1, IMAGEN_IMAGE_GENERATION_API_URL_2]

    for url_idx, target_url in enumerate(urls_to_try):
        key_label = f"Key {url_idx + 1}"
        
        for attempt in range(retries_per_url):
            try:
                async with session.post(target_url, json=payload, timeout=90) as response:
                    if response.status == 200:
                        resp_data = await response.json()
                        image_b64 = resp_data["predictions"][0]["bytesBase64Encoded"]
                        
                        with open(img_filename, "wb") as f:
                            f.write(base64.b64decode(image_b64))
                        return img_filename
                    
                    elif response.status == 429:
                        wait_time = (2 ** attempt) * 8 + (random.uniform(0, 2))
                        logger.warning(
                            f"🕒 {key_label} Rate Limited (429). "
                            f"Attempt {attempt+1}/{retries_per_url}. Retrying in {wait_time:.1f}s..."
                        )
                        await asyncio.sleep(wait_time)
                    
                    elif response.status == 400:
                        logger.error(f"⚠️ Safety/Prompt Filter Triggered on {key_label}. Skipping scene.")
                        return None
                    
                    else:
                        error_text = await response.text()
                        logger.error(f"⚠️ {key_label} API Error {response.status}: {error_text}")
                        
            except Exception as e:
                logger.error(f"❌ {key_label} Async Request Failed: {str(e)}")
                await asyncio.sleep(5)
        
        # If we finished the inner loop and didn't return, URL_1 is likely exhausted
        if url_idx == 0:
            logger.error(f"🚨 {key_label} fully exhausted or persistently limited. Switching to URL 2...")

    return None

async def image_generation(state: dict) -> dict:
    """Node 3: Generates images using text-based consistency and concurrent workers."""
    
    if not state.get("isvoicegenerated"):
        logger.warning("⚠️ Skipping Image Gen: Voiceover was not generated.")
        return state

    row_id = state.get("row_index")
    scenes = state["script"]["scenes"]
    metadata = state["script"].get("Metadata", {})
    
    log_extra = {"row_index": row_id}
    logger.info(f"Starting Image Generation for Row {row_id}", extra=log_extra)
    
    state["image_paths"] = [None] * len(scenes)

    # Visual Continuity logic remains intact as per Zeteon guidelines
    anchor = metadata.get("Global_Environmental_Anchor", "Cinematic background")
    subject = metadata.get("Visual_Continuity_Subject", "")
    style_suffix = f", featuring {subject}, set in {anchor}, photorealistic, 8k, extreme detail, cinematic lighting"
    
    semaphore = asyncio.Semaphore(2) 

    async def throttled_gen(session, prompt, img_filename, idx):
        async with semaphore:
            full_prompt = f"{prompt}{style_suffix}"
            result = await generate_single_image_async(session, full_prompt, img_filename)
            await asyncio.sleep(1.5) # Grace period for API stability
            return result

    async with aiohttp.ClientSession() as session:
        tasks = []
        task_indices = []
        
        for i, scene in enumerate(scenes):
            img_filename = os.path.join(OUTPUT_DIR, f"row_{row_id}_scene_{i+1}.png")
            
            # Logic for Cache Handling
            if os.path.exists(img_filename):
                state["image_paths"][i] = img_filename
                logger.info(f"📦 Cache Hit: Scene {i+1} found.")
                continue
            
            prompt = scene.get("Image_Action_Prompt") or scene.get("Video_Action_Prompt")
            task = asyncio.create_task(throttled_gen(session, prompt, img_filename, i))
            tasks.append(task)
            task_indices.append(i)

        if tasks:
            logger.info(f"🖼️ Dispatching {len(tasks)} concurrent image requests...")
            results = await asyncio.gather(*tasks)
            
            for idx_in_tasks, res in enumerate(results):
                original_scene_idx = task_indices[idx_in_tasks]
                if res: 
                    state["image_paths"][original_scene_idx] = res

    # Fallback Logic (shutil.copy)
    missing = [i for i, path in enumerate(state["image_paths"]) if path is None]
    if missing:
        logger.warning(f"⚠️ Missing {len(missing)} images. Applying fallback...")
        for i in missing:
            if i > 0 and state["image_paths"][i-1]:
                src = state["image_paths"][i-1]
                dst = os.path.join(OUTPUT_DIR, f"row_{row_id}_scene_{i+1}.png")
                shutil.copy(src, dst)
                state["image_paths"][i] = dst
            elif i < len(state["image_paths"]) - 1 and state["image_paths"][i+1]:
                src = state["image_paths"][i+1]
                dst = os.path.join(OUTPUT_DIR, f"row_{row_id}_scene_{i+1}.png")
                shutil.copy(src, dst)
                state["image_paths"][i] = dst

    # Final Verification for LangGraph State
    if all(state["image_paths"]):
        state["isimagesgenerated"] = True
        logger.info(f"✅ Image generation complete for Row {row_id} (All assets verified).")
    else:
        logger.error(f"❌ Image generation critical failure for Row {row_id}.")
        state["isimagesgenerated"] = False

    return state