import os
import sys
import logging
import random
import json
import asyncio
import datetime
import time
import traceback
from typing import List

# LangGraph Imports
from langgraph.graph import StateGraph, END

# Project Imports
from utils.schema import flowstate
from utils.sheets import get_worksheet
from utils.youtube_view_count import get_performance_context
from config import IDEA_GENERATION_API_URL, IDEA_SYSTEM_INSTRUCTIONS

# Node Imports
from nodes.script_gen import script_generation
from nodes.audio_gen import audio_generation
from nodes.image_gen import image_generation
from nodes.video_assembly import video_stitching_slideshow
from nodes.final_upload import video_upload_node

# --- LOGGING CONFIGURATION (AWS READY) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ZeteonPipeline")

# --- IDEA MANAGEMENT ---

def get_ready_idea(sheet_name="ideas"):
    """Fetches a pending idea or triggers generation of new ones."""
    try:
        worksheet = get_worksheet(sheet_name)
        all_records = worksheet.get_all_values()
        if not all_records: return None

        headers = all_records[0]
        data = all_records[1:]

        # Mapping headers to indices
        idx_map = {header: i for i, header in enumerate(headers)}
        idx_trigger = idx_map.get('Trigger Status')
        idx_idea = idx_map.get('Idea')

        pending_ideas = []
        uploaded_ideas = []

        for i, row in enumerate(data):
            sheet_row = i + 2 
            
            # Check if row needs processing
            trigger_status = row[idx_trigger].strip().upper() if len(row) > idx_trigger else ""
            
            needs_upload = trigger_status!='TRIGGERED'

            if needs_upload:
                pending_ideas.append((sheet_row, row))
            else:
                uploaded_ideas.append(row[idx_idea])

        if not pending_ideas:
            logger.info("Empty queue. Generating 3 new 'What Happens When' ideas...")
            new_ideas = generate_3_ideas(uploaded_ideas)
            
            today = datetime.date.today().strftime("%Y-%m-%d")
            ideas_to_add = [[today, idea, '', '', 'NOT-UPLOADED', '', 'NOT-UPLOADED', '', ''] for idea in new_ideas]
            
            if ideas_to_add:
                worksheet.append_rows(ideas_to_add)
                logger.info(f"✅ Added {len(ideas_to_add)} ideas to Sheet.")
                return get_ready_idea(sheet_name)

        row_num, chosen_row = random.choice(pending_ideas)
        logger.info(f"🎯 Target Idea: {chosen_row[idx_idea]} | Row: {row_num}")
        worksheet.update_cell(row_num, 10, 'TRIGGERED')
        return {"row_index": row_num, "idea": chosen_row[idx_idea]}

    except Exception as e:
        logger.error(f"Error in get_ready_idea: {str(e)}")
        return None

def generate_3_ideas(uploaded_ideas: List) -> List:
    """LLM call to generate next viral science topics."""
    performance_data = get_performance_context()
    
    prompt = f"""
You are the Zeteon Science Lead.
Your job is to propose three **viral, curiosity‑driven science explainer topics** optimized for short‑form video platforms.

Use the following performance context to understand what resonates:
{performance_data}

Avoid repeating or overlapping with these previously uploaded ideas:
{uploaded_ideas}

Return only valid JSON that matches this structure:

{{
  "ideas": ["string", "string", ...]
}}
"""
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": IDEA_SYSTEM_INSTRUCTIONS}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "object",
                "properties": {"ideas": {"type": "array", "items": {"type": "string"}}},
                "required": ["ideas"]
            }
        }
    }
    
    for attempt in range(3):
        try:
            import requests
            resp = requests.post(IDEA_GENERATION_API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            data = json.loads(resp.json()['candidates'][0]['content']['parts'][0]['text'])
            return data.get('ideas', [])
        except Exception as e:
            logger.warning(f"Idea Gen Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return []

# --- LANGGRAPH ORCHESTRATION ---

def build_workflow():
    """Constructs the LangGraph state machine."""
    workflow = StateGraph(flowstate)

    # Define Nodes
    workflow.add_node("script_gen", script_generation)
    workflow.add_node("audio_gen", audio_generation)
    workflow.add_node("image_gen", image_generation)
    workflow.add_node("video_assembly", video_stitching_slideshow)
    workflow.add_node("final_upload", video_upload_node)

    # Define Conditional Edge Logic
    def should_continue(state):
        if not state.get("isscriptgenerated"): return END
        if not state.get("isvoicegenerated"): return "assembly" # Try assembly with fallback
        if not state.get("isimagesgenerated"): return END
        if not state.get("isvideogenerated"): return END
        return "upload"

    # Set Entry Point
    workflow.set_entry_point("script_gen")

    # Define Connections
    workflow.add_edge("script_gen", "audio_gen")
    workflow.add_edge("audio_gen", "image_gen")
    workflow.add_edge("image_gen", "video_assembly")
    workflow.add_edge("video_assembly", "final_upload")
    workflow.add_edge("final_upload", END)

    return workflow.compile()

async def main():
    logger.info("🚀 Starting Zeteon Production Pipeline")
    
    # 1. Fetch Job
    initial_data = get_ready_idea()
    if not initial_data:
        logger.error("No pending tasks found in Google Sheets.")
        return

    # 2. Initialize State
    state: flowstate = {
        "idea": initial_data["idea"],
        "row_index": initial_data["row_index"],
        "script": {},
        "vo_path": "",
        "video_paths": [], 
        "alignment_data": {},
        "image_paths": [], 
        "isscriptgenerated": False,
        "isvoicegenerated": False,
        "isimagesgenerated": False,
        "isvideogenerated": False,
        "isvideouploaded" : False,
        "topic_comment": "" # Will be populated by script_gen
    }

    # 3. Run Graph
    app = build_workflow()
    try:
        # Running the graph as an async stream or direct invoke
        final_state = await app.ainvoke(state)
        
        if final_state.get("isvideouploaded"):
            logger.info(f"✅ Pipeline Successfully Completed for Row {initial_data['row_index']}")
        else:
            logger.error("❌ Pipeline failed at a critical node. Check logs above.")
            
    except Exception as e:
        logger.critical(f"💥 Unhandled exception in Main Graph: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())