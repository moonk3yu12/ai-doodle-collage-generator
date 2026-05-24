# ✨ AI Doodle Character Sheet Generator

Upload any character or person photo and get a kawaii doodle-collage character sheet — automatically analyzed and generated with GPT-4o Vision + DALL-E 3.

![demo preview](https://img.shields.io/badge/demo-Hugging%20Face%20Spaces-blue?logo=huggingface)
![python](https://img.shields.io/badge/python-3.11-blue)
![gradio](https://img.shields.io/badge/gradio-4.x-orange)
![openai](https://img.shields.io/badge/openai-GPT--4o%20%2B%20DALL--E%203-green)

---

## How It Works

| Step | Model | What it does |
|------|-------|-------------|
| 1 · Analyze | GPT-4o Vision | Reads your photo and extracts every visual detail |
| 2 · Prompt | GPT-4o | Converts the analysis into a rich doodle-collage prompt |
| 3 · Generate | DALL-E 3 | Creates a kawaii sticker-sheet with 6–8 poses |

---

## Local Development

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/ai-doodle-generator
cd ai-doodle-generator
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API key

```bash
# Windows PowerShell
cp .env.example .env
# Edit .env and add your key, then:
$env:OPENAI_API_KEY="sk-your-key-here"

# macOS / Linux
cp .env.example .env
export OPENAI_API_KEY="sk-your-key-here"
```

### 5. Run the app

```bash
python app.py
# Opens at http://localhost:7860
```

---

## Deploy to Hugging Face Spaces (Docker)

### Step 1 — Create a new Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose a Space name (e.g. `ai-doodle-generator`)
3. Select **Docker** as the SDK
4. Set visibility to **Public** or **Private**
5. Click **Create Space**

### Step 2 — Add your OpenAI API key as a Secret

1. Open your Space → **Settings** tab
2. Scroll to **Repository secrets**
3. Click **New secret**
4. Name: `OPENAI_API_KEY`
5. Value: your OpenAI API key (`sk-...`)
6. Click **Save**

> Secrets are encrypted and injected as environment variables at runtime.  
> Never put your API key in the code or commit it to git.

### Step 3 — Push your code

**Option A — Via the HF web UI:**

Drag and drop these files into the Space file browser:
- `app.py`
- `requirements.txt`
- `Dockerfile`
- `README.md`

**Option B — Via Git (recommended):**

```bash
# Add the HF Space as a remote
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/ai-doodle-generator

# Push
git push space main
```

The Space will automatically build the Docker image and start the app on port 7860.

### Step 4 — Watch the build

Click the **Logs** tab in your Space to follow the Docker build in real time.  
A green ✅ badge means the app is live. A red ❌ means check the logs for errors.

---

## GitHub Integration

### Link GitHub → Hugging Face

1. In your Space → **Settings** → **GitHub Actions** section
2. Follow the instructions to add the HF token as a GitHub secret: `HF_TOKEN`
3. Add this workflow file to your repo:

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
          lfs: true

      - name: Push to HF Spaces
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git remote add space https://USER:$HF_TOKEN@huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME
          git push space main --force
```

Every push to `main` will automatically redeploy your Space.

---

## Docker — Local Build & Run

```bash
# Build the image
docker build -t ai-doodle-generator .

# Run with your API key
docker run -p 7860:7860 -e OPENAI_API_KEY="sk-your-key-here" ai-doodle-generator

# Open http://localhost:7860
```

---

## Project Structure

```
ai-doodle-generator/
├── app.py              # Main Gradio application
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker config for HF Spaces
├── .env.example        # Environment variable template
└── README.md           # This file
```

---

## Tech Stack

- **[Gradio](https://gradio.app)** — UI framework
- **[OpenAI GPT-4o](https://platform.openai.com)** — Vision analysis + prompt crafting
- **[DALL-E 3](https://platform.openai.com/docs/guides/images)** — Image generation
- **[Hugging Face Spaces](https://huggingface.co/spaces)** — Hosting
- **Docker** — Containerization

---

## OpenAI Costs (approximate)

| Step | Model | Cost per run |
|------|-------|-------------|
| Image analysis | GPT-4o Vision | ~$0.003 |
| Prompt generation | GPT-4o | ~$0.001 |
| Image generation | DALL-E 3 1024×1024 | ~$0.040 |
| **Total per generation** | | **~$0.044** |

---

## License

MIT
