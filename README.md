---
title: AI Doodle Character Sheet Generator
emoji: ✨
colorFrom: pink
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# ✨ AI Doodle Character Sheet Generator

Upload any character or person photo and get a kawaii doodle-collage character sheet —
analyzed with GPT-4o Vision and generated with DALL-E 3.

---

## How It Works

| Step | Model | What it does |
|------|-------|--------------|
| 1 · Analyze | GPT-4o Vision | Reads your photo, extracts hair, eyes, outfit, palette |
| 2 · Prompt  | GPT-4o        | Turns the analysis into a doodle-collage image prompt  |
| 3 · Generate| DALL-E 3      | Creates a kawaii sticker-sheet with 6–8 poses          |

---

## Deploy to Hugging Face Spaces (Docker)

### Step 1 — Create a new Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Give it a name (e.g. `ai-doodle-generator`)
3. Select **Docker** as the SDK
4. Set visibility to Public or Private
5. Click **Create Space**

### Step 2 — Add your OpenAI API key as a Secret

1. Open your Space → **Settings** tab
2. Scroll to **Variables and secrets** → click **New secret**
3. Name: `OPENAI_API_KEY`  /  Value: your key (`sk-...`)
4. Click **Save**

> Secrets are encrypted and injected as environment variables at runtime.
> Never put your API key in the code or commit it to git.

### Step 3 — Push your files

**Option A — Git (recommended):**

```bash
# Clone the empty Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/ai-doodle-generator
cd ai-doodle-generator

# Copy your files in, then push
git add app.py requirements.txt Dockerfile README.md
git commit -m "Initial deploy"
git push
```

**Option B — HF web UI:**

Go to your Space → **Files** → **Add file** → upload all four files.

HF Spaces reads `README.md` for the YAML frontmatter (`sdk: docker`, `app_port: 7860`),
builds the Docker image from `Dockerfile`, and starts the container automatically.

### Step 4 — Watch the build

Click the **Logs** tab to follow the Docker build in real time.
A green **Running** badge means the app is live.

---

## GitHub → HF Spaces Auto-Deploy

Add this file to your GitHub repo:

```yaml
# .github/workflows/deploy.yml
name: Deploy to HF Spaces

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Push to HF Spaces
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git remote add space https://YOUR_USERNAME:$HF_TOKEN@huggingface.co/spaces/YOUR_USERNAME/ai-doodle-generator
          git push space main --force
```

Add `HF_TOKEN` (a Hugging Face write token) to your GitHub repo secrets.
Every push to `main` will automatically redeploy the Space.

---

## Local Development

```bash
# Build and run with Docker
docker build -t ai-doodle-generator .
docker run -p 7860:7860 -e OPENAI_API_KEY="sk-your-key-here" ai-doodle-generator
# Open http://localhost:7860

# Or run directly with Python
pip install -r requirements.txt
export OPENAI_API_KEY="sk-your-key-here"   # macOS/Linux
# $env:OPENAI_API_KEY="sk-your-key-here"  # Windows PowerShell
python app.py
```

---

## Project Structure

```
ai-doodle-generator/
├── app.py            # Gradio app — full 3-step pipeline
├── requirements.txt  # Python dependencies
├── Dockerfile        # Docker config for HF Spaces
└── README.md         # This file (also the HF Spaces config via frontmatter)
```

---

## Tech Stack

- [Gradio](https://gradio.app) — UI framework
- [OpenAI GPT-4o](https://platform.openai.com) — Vision analysis + prompt crafting
- [DALL-E 3](https://platform.openai.com/docs/guides/images) — Image generation
- [Hugging Face Spaces](https://huggingface.co/spaces) — Hosting (Docker SDK)

---

## Cost per Generation (approximate)

| Step | Model | Cost |
|------|-------|------|
| Image analysis   | GPT-4o Vision       | ~$0.003 |
| Prompt crafting  | GPT-4o              | ~$0.001 |
| Image generation | DALL-E 3 1024×1024  | ~$0.040 |
| **Total**        |                     | **~$0.044** |

---

## License

MIT
