Zeteon is a fully automated content creation system designed to generate cinematic science explainer videos for platforms like YouTube and Instagram. 
It orchestrates script generation, audio synthesis, image creation, video assembly, and multi-platform publishing — all from a single idea.

🚀 Pipeline Overview
main.py
├── Phase 1: Fetch idea from Google Sheet
├── Phase 2: Generate script (script_gen.py)
├── Phase 3: Generate audio (audio_gen.py)
├── Phase 4: Generate images (image_gen.py)
├── Phase 5: Assemble video (video_assembly.py)
├── Phase 6: Upload to YouTube & Instagram (final_upload.py)



📁 Project Structure
Zeteon/
├── main.py                  # Entry point for full pipeline
├── config.py                # Loads secrets and constants from .env
├── .env                     # Local secrets (ignored in Git)
├── .gitignore               # Prevents secrets and cache from being committed
├── token.pickle             # Google OAuth token (ignored)
├── client_secret.json       # Google API credentials (ignored)
├── credentials.json         # Additional OAuth credentials (ignored)
│
├── nodes/                   # Modular pipeline stages
│   ├── script_gen.py
│   ├── audio_gen.py
│   ├── image_gen.py
│   ├── video_assembly.py
│   ├── final_upload.py
│   └── test.py              # Standalone test runner
│
├── utils/                   # Core helpers
│   ├── sheets.py            # Google Sheets integration
│   ├── schema.py            # Flowstate and metadata structure
│   ├── youtube_auth.py      # YouTube OAuth setup
│   ├── youtube_view_count.py# Analytics fetcher
│
├── prompts/                 # Prompt templates for LLMs
├── assets/                  # Generated media (images, audio, video)



🔐 Secrets & Environment Setup
Create a .env file in the root directory:
GEMINI_API_KEY_1=your_key_here
CLAUDE_API_KEY=your_key_here
INSTA_ACCESS_TOKEN=your_token_here
INSTA_ACCOUNT_ID=your_account_id
YOUTUBE_CLIENT_ID=...
YOUTUBE_CLIENT_SECRET=...


Then load it in config.py using python-dotenv.

📦 Dependencies
Install required packages:
pip install -r requirements.txt


Recommended packages include:
- google-auth, google-api-python-client
- python-dotenv
- requests, ffmpeg-python
- openai, anthropic, Pillow


