import gradio as gr
import openai
import base64
import os
import io
import requests
from PIL import Image


# ── OpenAI client ──────────────────────────────────────────────────────────────

def get_client() -> openai.OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it in HF Spaces → Settings → Secrets.")
    return openai.OpenAI(api_key=api_key)


# ── Helpers ────────────────────────────────────────────────────────────────────

def pil_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def url_to_pil(url: str) -> Image.Image:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


# ── Pipeline steps ─────────────────────────────────────────────────────────────

def analyze_character(client: openai.OpenAI, image: Image.Image) -> str:
    """GPT-4o Vision: extract visual traits from the uploaded image."""
    b64 = pil_to_base64(image)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Describe this character or person in detail for use in image generation.\n"
                        "Cover every visual trait: hair (color, length, style), eye color and shape, "
                        "skin tone, clothing style and colors, accessories, overall aesthetic, "
                        "personality vibe, and dominant color palette.\n"
                        "Be specific and descriptive."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
            ],
        }],
        max_tokens=450,
    )
    return resp.choices[0].message.content.strip()


def build_doodle_prompt(client: openai.OpenAI, analysis: str) -> str:
    """GPT-4o: craft a DALL-E 3 prompt for a doodle-collage character sheet."""
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": (
                f"Character description:\n{analysis}\n\n"
                "Write a single image-generation prompt (no headers, no bullets, one paragraph) "
                "for a cute kawaii doodle-collage character sheet. Requirements:\n"
                "- Hand-drawn doodle / sketch illustration style\n"
                "- 6 to 8 mini poses and expressions of the SAME character arranged like a sticker sheet\n"
                "- Soft pastel colors matching the character's palette\n"
                "- Small decorative doodles between poses: stars, hearts, tiny flowers, sparkles\n"
                "- Clean white or very light background\n"
                "- Anime-inspired but with a hand-drawn illustration twist\n"
                "- Chibi proportions encouraged\n"
                "Keep the prompt under 180 words."
            ),
        }],
        max_tokens=250,
    )
    return resp.choices[0].message.content.strip()


def generate_sheet(client: openai.OpenAI, prompt: str) -> Image.Image:
    """DALL-E 3: generate the character sheet image."""
    resp = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    return url_to_pil(resp.data[0].url)


# ── Gradio handler ─────────────────────────────────────────────────────────────

def run_pipeline(image: Image.Image, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Please upload a character image first.")

    try:
        client = get_client()
    except ValueError as e:
        raise gr.Error(str(e))

    try:
        progress(0.10, desc="Analyzing character features...")
        analysis = analyze_character(client, image)

        progress(0.45, desc="Crafting doodle prompt...")
        prompt = build_doodle_prompt(client, analysis)

        progress(0.70, desc="Generating character sheet...")
        sheet = generate_sheet(client, prompt)

        progress(1.00, desc="Done!")
        return sheet, analysis, prompt

    except openai.AuthenticationError:
        raise gr.Error("Invalid OpenAI API key. Double-check your OPENAI_API_KEY secret.")
    except openai.RateLimitError:
        raise gr.Error("OpenAI rate limit hit. Please wait a moment and try again.")
    except openai.BadRequestError as e:
        raise gr.Error(f"OpenAI rejected the request: {e}. Try a different image.")
    except Exception as e:
        raise gr.Error(f"Unexpected error: {e}")


# ── Custom CSS ─────────────────────────────────────────────────────────────────

CSS = """
#app-header { text-align: center; padding: 1.5rem 0 0.5rem; }
#app-header h1 {
    font-size: 2.2rem;
    background: linear-gradient(90deg, #f472b6, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
#app-header p { color: #64748b; font-size: 1rem; }

#generate-btn {
    background: linear-gradient(90deg, #f472b6, #a78bfa) !important;
    color: white !important;
    border: none !important;
    font-size: 1rem !important;
    border-radius: 8px !important;
    padding: 0.7rem 1.5rem !important;
    width: 100% !important;
}
#generate-btn:hover { opacity: 0.88; transform: translateY(-1px); }

.tip-box {
    background: #fdf4ff;
    border: 1px solid #e9d5ff;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    color: #7c3aed;
}

.step-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
"""


# ── UI Layout ──────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="AI Doodle Character Sheet Generator",
    theme=gr.themes.Soft(primary_hue="pink", secondary_hue="purple"),
    css=CSS,
) as demo:

    with gr.Column(elem_id="app-header"):
        gr.Markdown(
            "# ✨ AI Doodle Character Sheet Generator\n"
            "Upload any character or person photo → get a kawaii doodle collage character sheet"
        )

    gr.Markdown("---")

    with gr.Row(equal_height=True):

        with gr.Column(scale=1):
            gr.Markdown("### 📸 Upload Image")
            image_input = gr.Image(
                type="pil",
                label="Character or Person Photo",
                height=340,
            )
            generate_btn = gr.Button(
                "🎨  Generate Character Sheet",
                variant="primary",
                elem_id="generate-btn",
            )
            gr.Markdown(
                "**Tips:** Clear front-facing photos work best.  \n"
                "Anime / game characters, OCs, and real people all work!",
                elem_classes="tip-box",
            )

        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ Character Sheet")
            image_output = gr.Image(
                type="pil",
                label="Generated Character Sheet",
                height=340,
                show_download_button=True,
            )

    with gr.Accordion("📋 Analysis & Prompt Details", open=False):
        with gr.Row():
            analysis_out = gr.Textbox(
                label="🔍 Character Analysis  (GPT-4o Vision)",
                lines=7,
                interactive=False,
                placeholder="Analysis will appear here after generation…",
            )
            prompt_out = gr.Textbox(
                label="✍️ DALL-E 3 Prompt",
                lines=7,
                interactive=False,
                placeholder="Generated prompt will appear here…",
            )

    gr.Markdown("---\n### 💡 How It Works")
    with gr.Row():
        gr.Markdown(
            "**1 · Analyze**  \nGPT-4o Vision reads your photo and extracts every visual detail: "
            "hair, eyes, outfit, color palette, and vibe.",
            elem_classes="step-box",
        )
        gr.Markdown(
            "**2 · Prompt**  \nGPT-4o converts the analysis into a rich kawaii doodle-collage "
            "image generation prompt.",
            elem_classes="step-box",
        )
        gr.Markdown(
            "**3 · Generate**  \nDALL-E 3 creates a sticker-sheet style character sheet with "
            "6–8 poses, expressions, and cute doodle decorations.",
            elem_classes="step-box",
        )

    gr.Markdown(
        "---\n"
        "*Built with [Gradio](https://gradio.app) · "
        "[OpenAI GPT-4o](https://platform.openai.com) · "
        "[DALL-E 3](https://platform.openai.com/docs/guides/images) · "
        "Hosted on [Hugging Face Spaces](https://huggingface.co/spaces)*"
    )

    generate_btn.click(
        fn=run_pipeline,
        inputs=[image_input],
        outputs=[image_output, analysis_out, prompt_out],
    )


# Docker / HF Spaces: bind to 0.0.0.0 on port 7860
demo.launch(server_name="0.0.0.0", server_port=7860)
