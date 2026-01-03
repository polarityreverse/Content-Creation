import subprocess
import os
import json
import re
import random
import logging
from utils.sheets import get_worksheet
from config import OUTPUT_DIR

# Set up production logging
logger = logging.getLogger(__name__)

def ass_ts(sec):
    """Timestamp helper for ASS Subtitles."""
    sec = max(0, sec)
    h, m, s = int(sec // 3600), int((sec % 3600) // 60), sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def generate_ass_karaoke(state, alignment_data, topic_comment, pause_at_end, max_words=4):
    """Node 4 Helper: Generates high-retention karaoke subtitles with CTA."""
    row_id = state["row_index"]
    ass_path = os.path.join(OUTPUT_DIR, f"subs_row_{row_id}.ass")
    
    chars = alignment_data["characters"]
    starts = alignment_data["character_start_times_seconds"]
    ends = alignment_data["character_end_times_seconds"]

    words, cur_word, word_start = [], "", None
    for i, ch in enumerate(chars):
        if not word_start: word_start = starts[i]
        if ch.isspace() or i == len(chars) - 1:
            if cur_word.strip():
                clean = re.sub(r'[^a-zA-Z0-9,\.\!\?\']', '', cur_word.strip())
                words.append({"text": clean.upper(), "start": word_start, "end": ends[i]})
            cur_word, word_start = "", None
        else: 
            cur_word += ch

    ass_header = [
        "[Script Info]", "ScriptType: v4.00+", "PlayResX: 1080", "PlayResY: 1920", "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,BorderStyle,Outline,Shadow,Alignment,MarginV",
        "Style: Default,Calibri,85,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,1,1,4,0,2,280",
        "Style: CTA,Calibri,105,&H0000FFFF,&H000000FF,&H00000000,&H00000000,1,0,1,4,0,2,960",
        "", "[Events]", "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text"
    ]

    events, current_chunk = [], []
    def create_line(chunk):
        s_t, e_t = chunk[0]["start"], chunk[-1]["end"]
        line_text = "".join([f"{{\\k{int((w['end']-w['start'])*100)}}}{w['text']} " for w in chunk])
        return f"Dialogue: 0,{ass_ts(s_t)},{ass_ts(e_t)},Default,,0,0,0,,{line_text.strip()}"

    for word_obj in words:
        current_chunk.append(word_obj)
        if len(current_chunk) >= max_words or any(p in word_obj["text"] for p in ["!", ".", "?"]):
            events.append(create_line(current_chunk))
            current_chunk = []

    if current_chunk:
        events.append(create_line(current_chunk))

    # --- 1.5s Pause CTA Logic ---
    final_vo_time = ends[-1]
    events.append(f"Dialogue: 0,{ass_ts(final_vo_time)},{ass_ts(final_vo_time + pause_at_end)},CTA,,0,0,0,,{topic_comment.upper()}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ass_header + events))
    return ass_path

def sync_to_cloud(file_path, row_id):
    """Syncs final assets to GitHub and updates Google Sheets status."""
    GITHUB_USER, GITHUB_REPO, GITHUB_BRANCH = "polarityreverse", "Content-Creation", "master"
    try:
        # In AWS, ensure the Git environment is initialized or use a dedicated API upload
        subprocess.run(["git", "add", file_path], check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", f"Upload Video_Row_{row_id}"], check=True, capture_output=True)
        subprocess.run(["git", "push", "origin", GITHUB_BRANCH], check=True, capture_output=True)
        logger.info(f"Git: Video {row_id} pushed to branch {GITHUB_BRANCH}")
    except Exception as e:
        logger.error(f"Git Push failed for row {row_id}: {e}")
    
    raw_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/assets/{os.path.basename(file_path)}"
    
    try:
        sheet = get_worksheet("ideas")
        sheet.update_cell(row_id, 4, raw_url)
        # Update columns for YT and Insta status
        for col in [5, 7]:
            sheet.update_cell(row_id, col, "GIT_READY")
        logger.info(f"Sheets: Row {row_id} updated with GIT_READY and raw_url")
    except Exception as e:
        logger.error(f"Google Sheets update failed for row {row_id}: {e}")

    return raw_url

def video_stitching_slideshow(state):
    """Node 4: Main FFmpeg engine with synced timing and CTA pause."""
    row_id = state.get("row_index")
    output_filename = f"Video_Row_{row_id}.mp4"
    final_video_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # 1. CACHE & LOGGING
    log_extra = {"row_index": row_id}
    if os.path.exists(final_video_path):
        logger.info(f"📦 Cache Hit: Assembled video found for Row {row_id}", extra=log_extra)
        state["isvideogenerated"] = True
        state["final_video_path"] = final_video_path
        return state

    image_files = state.get("image_paths", [])
    scenes = state["script"].get("scenes", [])
    topic_comment = state.get("topic_comment") or "LIKE & FOLLOW FOR MORE!"
    audio_vo = os.path.join(OUTPUT_DIR, f"vo_row_{row_id}.mp3")

    try:
        # 2. TIMING & DURATION CALCULATIONS
        with open(os.path.join(OUTPUT_DIR, f"alignment_row_{row_id}.json"), "r") as f:
            alignment_data = json.load(f)

        cmd_dur = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{audio_vo}"'
        vo_duration = float(subprocess.check_output(cmd_dur, shell=True))
        pause_at_end = 1.5 
        total_target_dur = vo_duration + pause_at_end

        total_scene_script_dur = sum(float(s["Scene_Duration"]) for s in scenes)
        stretch_factor = vo_duration / total_scene_script_dur
        
        calc_durs = []
        for i in range(len(image_files)):
            base_dur = float(scenes[i]["Scene_Duration"]) * stretch_factor
            calc_durs.append(base_dur + (0.5 if i < len(image_files) - 1 else pause_at_end))

        # 3. CONSTRUCT FILTERS
        v_filters = []
        for i in range(len(image_files)):
            v_filters.append(
                f"[{i}:v]scale=2160:-1,format=yuv420p,fps=30,"
                f"zoompan=z='min(zoom+0.001,1.5)':d={int(calc_durs[i]*30)}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920[v{i}];"
            )

        concat_filter, last_v, cur_offset = "", "v0", 0
        for i in range(1, len(image_files)):
            cur_offset += (float(scenes[i-1]["Scene_Duration"]) * stretch_factor)
            concat_filter += f"[{last_v}][v{i}]xfade=transition=fade:duration=0.5:offset={cur_offset:.3f}[xf{i}];"
            last_v = f"xf{i}"

        # 4. AUDIO MASTERING
        selected_music = None
        if os.path.exists(OUTPUT_DIR):
            mp3s = [f for f in os.listdir(OUTPUT_DIR) if f.lower().startswith('bkg_music') and f.lower().endswith('.mp3')]
            if mp3s: selected_music = os.path.join(OUTPUT_DIR, random.choice(mp3s))

        vo_idx, bg_idx = len(image_files), len(image_files) + 1
        fade_start = total_target_dur - 1.0

        if selected_music:
            audio_filter = (
                f"[{vo_idx}:a]apad=pad_dur={pause_at_end},asplit=2[vo_p1][vo_p2];"
                f"[{bg_idx}:a]volume=0.12,aloop=loop=-1:size=2e+09,afade=t=out:st={fade_start}:d=1[bg_loop];"
                f"[bg_loop][vo_p1]sidechaincompress=threshold=0.05:ratio=12:attack=20:release=200[bg_duck];"
                f"[vo_p2][bg_duck]amix=inputs=2:duration=longest:weights=1 1[a_final]"
            )
        else:
            audio_filter = f"[{vo_idx}:a]apad=pad_dur={pause_at_end}[a_final]"
            
        ass_path = generate_ass_karaoke(state, alignment_data, topic_comment, pause_at_end)
        escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:").replace(" ", "\\ ")

        # 5. EXECUTE FFMPEG
        cmd = ["ffmpeg", "-y"]
        for i, img in enumerate(image_files):
            cmd += ["-loop", "1", "-t", f"{calc_durs[i]:.3f}", "-i", img]
        cmd += ["-i", audio_vo]
        if selected_music: cmd += ["-i", selected_music]

        full_filter = "".join(v_filters) + concat_filter + f"[{last_v}]ass=filename='{escaped_ass}'[v_final];" + audio_filter

        cmd += [
            "-filter_complex", full_filter,
            "-map", "[v_final]", "-map", "[a_final]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "veryfast",
            "-t", f"{total_target_dur:.3f}", final_video_path
        ]

        logger.info(f"🎬 Starting FFmpeg assembly for Row {row_id}...")
        subprocess.run(cmd, check=True, capture_output=True)
        
        # 6. SYNC & CLEANUP
        sync_to_cloud(final_video_path, row_id)
        state["final_video_path"] = final_video_path
        state["isvideogenerated"] = True
        
        # Cleanup temporary .ass file to save space in AWS /tmp
        if os.path.exists(ass_path):
            os.remove(ass_path)

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed for Row {row_id}: {e.stderr.decode()}", exc_info=True)
        state["isvideogenerated"] = False
    except Exception as e:
        logger.error(f"Node 4 Critical Failure: {str(e)}", exc_info=True)
        state["isvideogenerated"] = False

    return state