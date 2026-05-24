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
    """GPT-4o Vision: extract rich, structured visual traits for doodle collage generation."""
    b64 = pil_to_base64(image)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a professional character designer preparing a reference brief.\n"
                        "Analyze this image and extract every visual detail. Be extremely specific — "
                        "vague descriptions produce bad illustrations.\n\n"
                        "Structure your answer in these sections:\n"
                        "HAIR: exact color with adjective (e.g. 'bubblegum pink with bleached tips'), "
                        "length, and style (twin tails, messy bob, etc.)\n"
                        "EYES: exact color, shape, notable features (e.g. star pupils, thick lashes)\n"
                        "SKIN: tone in plain words\n"
                        "OUTFIT: every piece — top, bottom, shoes, layers — with colors and any patterns\n"
                        "ACCESSORIES: every item present (hair clips, ribbons, bags, jewelry, etc.)\n"
                        "SIGNATURE DETAILS: 2-3 things that make this character instantly recognizable\n"
                        "COLOR PALETTE: 5 dominant color names (e.g. 'dusty rose', 'warm cream', 'cobalt blue')\n"
                        "PERSONALITY VIBE: 3 adjectives describing their energy\n\n"
                        "Do not add commentary. Just the structured facts."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
            ],
        }],
        max_tokens=600,
    )
    return resp.choices[0].message.content.strip()


def build_doodle_prompt(client: openai.OpenAI, analysis: str) -> str:
    """GPT-4o: craft a gpt-image-1 prompt for a chaotic doodle collage character sheet."""
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a world-class anime illustrator and prompt engineer. "
                    "You specialize in chaotic kawaii doodle collage art — messy, dense, adorable, "
                    "and visually overwhelming. Your image prompts always produce outputs that look like "
                    "a famous anime artist's personal sketchbook page: crammed with poses, handwritten "
                    "notes, tiny mascot doodles, arrows, symbols, and layered decorations. "
                    "You never write clean, minimal, or sterile prompts. "
                    "Every prompt you write produces something social-media-worthy and portfolio-quality."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CHARACTER REFERENCE DATA:\n{analysis}\n\n"
                    "Write ONE image generation prompt for a CHAOTIC DOODLE COLLAGE CHARACTER SHEET.\n\n"
                    "The prompt MUST describe all of the following elements:\n\n"
                    "COMPOSITION:\n"
                    "- 8 to 10 versions of the character crammed across the page — overlapping, rotated, "
                    "no grid, deliberately asymmetric and messy\n"
                    "- Mix of render styles: some fully colored, some rough pencil sketches, some "
                    "half-finished with visible construction lines and stray marks\n"
                    "- At least 2 super-deformed chibi versions\n"
                    "- One large expressive close-up face with exaggerated emotion\n"
                    "- One tiny full-body silhouette doodle in a corner\n\n"
                    "ANNOTATIONS AND TEXT:\n"
                    "- Messy handwritten labels pointing at outfit details with small arrows: "
                    "'her fav!!', 'so soft~', 'notice me senpai!!', 'iconic look'\n"
                    "- Character name or nickname scrawled in bubbly handwritten font somewhere visible\n"
                    "- At least one dialogue bubble or thought cloud with a short phrase\n\n"
                    "DECORATIONS (scattered densely across every empty space):\n"
                    "- Bold outlined stars ★, doodled hearts ♡, sparkle bursts ✦, "
                    "double exclamation marks !!, tiny clouds, small flowers, asterisks *\n"
                    "- One small mascot animal or creature doodle (cat, bunny, or ghost)\n"
                    "- Sticker-like elements overlapping the art: small stamps, tiny icons, "
                    "washi tape strips\n"
                    "- Repeating tiny pattern filling gaps (dots, x marks, tiny stars)\n\n"
                    "STYLE AND TEXTURE:\n"
                    "- Slightly off-white aged sketchbook paper background with faint ruled lines or "
                    "watercolor paper texture\n"
                    "- Mixed media feel: flat pastel fills, loose watercolor washes, rough ink outlines, "
                    "colored pencil hatching\n"
                    "- MS Paint-adjacent roughness in some elements — wobbly lines, uneven fills\n"
                    "- Overall palette matches the character's colors; soft pastels dominate\n\n"
                    "CHARACTER FIDELITY (non-negotiable):\n"
                    "- Every single version of the character must have the EXACT same hairstyle, "
                    "hair color, eye color, outfit, and accessories as described in the reference\n"
                    "- The character must be instantly recognizable across all poses and styles\n\n"
                    "FORMAT: One dense paragraph, no bullet points, no headers. "
                    "200 to 250 words. Be aggressive and specific. "
                    "Write as if briefing a professional anime illustrator with no room for interpretation."
                ),
            },
        ],
        max_tokens=450,
    )
    return resp.choices[0].message.content.strip()


def generate_sheet(client: openai.OpenAI, prompt: str) -> Image.Image:
    """gpt-image-1: generate the character sheet (returns base64, no URL download needed)."""
    resp = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024",
        quality="high",
    )
    image_bytes = base64.b64decode(resp.data[0].b64_json)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


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

with gr.Blocks(title="AI Doodle Character Sheet Generator") as demo:

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
                label="✍️ gpt-image-1 Prompt",
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
            "**3 · Generate**  \ngpt-image-1 renders a chaotic doodle collage with 8–10 poses, "
            "handwritten annotations, mascot doodles, and layered decorations.",
            elem_classes="step-box",
        )

    gr.Markdown(
        "---\n"
        "*Built with [Gradio](https://gradio.app) · "
        "[OpenAI GPT-4o](https://platform.openai.com) · "
        "[gpt-image-1](https://platform.openai.com/docs/guides/images) · "
        "Hosted on [Hugging Face Spaces](https://huggingface.co/spaces)*"
    )

    generate_btn.click(
        fn=run_pipeline,
        inputs=[image_input],
        outputs=[image_output, analysis_out, prompt_out],
    )


# Docker / HF Spaces: bind to 0.0.0.0 on port 7860
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    theme=gr.themes.Soft(primary_hue="pink", secondary_hue="purple"),
    css=CSS,
)
