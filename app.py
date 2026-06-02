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


# ── Generation mode configs ────────────────────────────────────────────────────
# Each entry defines the visual brief injected into build_doodle_prompt().
# Keys are English (used internally); Korean labels are shown in the Radio UI.

MODE_CONFIGS = {
    # Layout-only briefs. Style/texture comes entirely from build_doodle_prompt().
    # Do NOT add style words here — they contaminate the texture anchor.
    "Full Character Sheet": {
        "brief": (
            "multiple loose sketches of the same character scattered across the page, "
            "different poses and expressions, some unfinished and half-drawn, "
            "chibi versions included, messy overlapping layout, "
            "handwritten notes and arrows around the sketches, "
            "hearts ♡ stars ★ !! scribbled in gaps"
        ),
    },
    "Portrait Doodle": {
        "brief": (
            "single upper-body portrait centered on page, "
            "expressive face close-up, "
            "small star and heart doodles loosely framing the drawing, "
            "handwritten nickname or label nearby"
        ),
    },
    "Upper Body Character": {
        "brief": (
            "waist-up character sketch centered on page, "
            "outfit clearly visible, "
            "a few tiny doodle accents at the edges"
        ),
    },
    "Chibi Sticker": {
        "brief": (
            "3 to 5 chibi doodles of the character, "
            "scattered loosely like stickers, not in a grid, "
            "small hearts and stars scribbled between them"
        ),
    },
    "Simple Clean Portrait": {
        "brief": (
            "single character sketch centered on page, "
            "minimal layout, "
            "one or two tiny star or heart doodles nearby, nothing more"
        ),
    },
}

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


def build_doodle_prompt(client: openai.OpenAI, analysis: str, mode: str) -> str:
    """GPT-4o: write a short texture-first prompt that produces raw sketchbook doodles."""
    brief = MODE_CONFIGS.get(mode, MODE_CONFIGS["Full Character Sheet"])["brief"]

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You write short image prompts that produce raw anime sketchbook doodles. "
                    "NOT polished illustrations. NOT professional art. NOT rendered artwork.\n"
                    "Rules you must follow:\n"
                    "- Always open with physical media words: "
                    "'rough ballpoint pen sketch', 'cheap copic marker scribbles', "
                    "'scratchy ink doodles', 'white notebook paper'\n"
                    "- Keep the total prompt under 90 words\n"
                    "- Never use these words: polished, rendered, detailed, cinematic, "
                    "professional, refined, masterpiece, high quality, artstation, "
                    "glossy, elegant, portfolio, illustration, digital art"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CHARACTER: {analysis}\n\n"
                    f"LAYOUT: {brief}\n\n"
                    "Write ONE image prompt. Follow this exact structure:\n"
                    "1. Media anchor (first phrase): rough ballpoint pen sketch / "
                    "cheap marker coloring / white notebook paper / scratchy unfinished lines\n"
                    "2. Character (one sentence): exact hair color and style, "
                    "eye color, outfit — taken from CHARACTER above\n"
                    "3. Layout (1-2 short phrases): taken from LAYOUT above\n"
                    "4. Texture closer (last phrase): messy fanart energy, "
                    "amateur sketchbook feel, scanned notebook page\n"
                    "Total: under 90 words. One paragraph. No headers."
                ),
            },
        ],
        max_tokens=180,
    )
    return resp.choices[0].message.content.strip()


def generate_sheet(client: openai.OpenAI, prompt: str) -> Image.Image:
    """gpt-image-1: generate the character sheet (returns base64, no URL download needed)."""
    resp = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024",
        quality="medium",
    )
    image_bytes = base64.b64decode(resp.data[0].b64_json)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


# ── Gradio handler ─────────────────────────────────────────────────────────────

def run_pipeline(image: Image.Image, mode: str, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Please upload an image first.")

    try:
        client = get_client()
    except ValueError as e:
        raise gr.Error(str(e))

    try:
        progress(0.10, desc="Analyzing character features...")
        analysis = analyze_character(client, image)

        progress(0.45, desc="Writing image prompt...")
        prompt = build_doodle_prompt(client, analysis, mode)

        progress(0.70, desc="Generating character sheet...")
        sheet = generate_sheet(client, prompt)

        progress(1.00, desc="Done!")
        return sheet, analysis, prompt

    except openai.AuthenticationError:
        raise gr.Error("Invalid OpenAI API key. Check your OPENAI_API_KEY secret.")
    except openai.RateLimitError:
        raise gr.Error("OpenAI rate limit hit. Please wait and try again.")
    except openai.BadRequestError as e:
        raise gr.Error(f"OpenAI rejected the request: {e}. Try a different image.")
    except Exception as e:
        raise gr.Error(f"Unexpected error: {e}")


# ── Custom CSS ─────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

body, .gradio-container {
    font-family: 'Nunito', sans-serif !important;
    background: linear-gradient(135deg, #fff0f6 0%, #f5f0ff 50%, #f0f4ff 100%) !important;
}

#app-header {
    text-align: center;
    padding: 2rem 0 0.5rem;
}
#app-header h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff6eb4, #c084fc, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
    letter-spacing: -0.5px;
    text-shadow: none;
}
#app-header p {
    color: #a78bca;
    font-size: 1rem;
    font-weight: 600;
}

/* 메인 카드 패널 */
.panel-card {
    background: white;
    border-radius: 24px !important;
    box-shadow: 0 4px 24px rgba(192, 132, 252, 0.12) !important;
    border: 2px solid #f3e8ff !important;
    padding: 1.2rem !important;
}

/* 섹션 제목 */
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #9333ea;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.3rem;
}

/* 생성 버튼 */
#generate-btn {
    background: linear-gradient(135deg, #ff6eb4, #c084fc, #818cf8) !important;
    color: white !important;
    border: none !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    border-radius: 50px !important;
    padding: 0.75rem 1.5rem !important;
    width: 100% !important;
    margin-top: 0.5rem !important;
    box-shadow: 0 4px 16px rgba(192, 132, 252, 0.4) !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.3px !important;
}
#generate-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(192, 132, 252, 0.5) !important;
}
#generate-btn:active {
    transform: translateY(0px) !important;
}

/* 팁 박스 */
.tip-box {
    background: linear-gradient(135deg, #fdf4ff, #f5f0ff);
    border: 1.5px solid #e9d5ff;
    border-radius: 16px;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    font-weight: 600;
    color: #9333ea;
    margin-top: 0.5rem;
}

/* 스텝 박스 */
.step-box {
    background: white;
    border: 2px solid #f3e8ff;
    border-radius: 20px;
    padding: 1.2rem;
    text-align: center;
    box-shadow: 0 2px 12px rgba(192, 132, 252, 0.1);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.step-box:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(192, 132, 252, 0.2);
}

/* 라디오 버튼 */
.gr-radio label {
    border-radius: 12px !important;
    border: 1.5px solid #e9d5ff !important;
    font-weight: 600 !important;
}
.gr-radio label:hover {
    background: #fdf4ff !important;
    border-color: #c084fc !important;
}

/* 구분선 */
hr {
    border: none !important;
    border-top: 2px dashed #f0e4ff !important;
    margin: 1rem 0 !important;
}

/* 이미지 업로드 영역 */
.gr-image {
    border-radius: 18px !important;
    border: 2px dashed #d8b4fe !important;
}

/* 텍스트박스 */
.gr-textbox textarea {
    border-radius: 14px !important;
    border: 1.5px solid #e9d5ff !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.9rem !important;
}

/* 아코디언 */
.gr-accordion {
    border-radius: 16px !important;
    border: 2px solid #f3e8ff !important;
    overflow: hidden !important;
}
"""


# ── UI Layout ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="AI Doodle Character Sheet Generator") as demo:

    with gr.Column(elem_id="app-header"):
        gr.Markdown(
            "# ✨ AI Doodle Character Sheet ✨\n"
            "ʕ •ᴥ•ʔ 캐릭터 사진을 올리면 → 귀여운 낙서 콜라주로 만들어드려요 ♡"
        )

    gr.Markdown("---")

    with gr.Row(equal_height=True):

        with gr.Column(scale=1):
            gr.Markdown("### 📸 사진 올리기")
            image_input = gr.Image(
                type="pil",
                label="Character or Person Photo",
                height=300,
            )
            mode_selector = gr.Radio(
                choices=[
                    ("🎨 Full Character Sheet", "Full Character Sheet"),
                    ("🖼️ Portrait Doodle", "Portrait Doodle"),
                    ("👗 Upper Body Character", "Upper Body Character"),
                    ("🌟 Chibi Sticker", "Chibi Sticker"),
                    ("✨ Simple Clean Portrait", "Simple Clean Portrait"),
                ],
                value="Full Character Sheet",
                label="🎭 Generation Mode",
            )
            generate_btn = gr.Button(
                "✨  낙서 시트 만들기  ♡",
                variant="primary",
                elem_id="generate-btn",
            )
            gr.Markdown(
                "💡 **Tip:** 정면 사진일수록 잘 나와요!  \n"
                "애니 캐릭터, 게임 캐릭터, 실제 사람 모두 OK ʕ•ᴥ•ʔ",
                elem_classes="tip-box",
            )

        with gr.Column(scale=1):
            gr.Markdown("### 🎨 완성된 낙서 시트 ♡")
            image_output = gr.Image(
                type="pil",
                label="Generated Character Sheet",
                height=460,
            )

    with gr.Accordion("🔍 분석 & 프롬프트 보기 ▾", open=False):
        with gr.Row():
            analysis_out = gr.Textbox(
                label="👁️ 캐릭터 분석 결과 (GPT-4o Vision)",
                lines=7,
                interactive=False,
                placeholder="생성 후 캐릭터 분석 결과가 여기에 나타나요 ✨",
            )
            prompt_out = gr.Textbox(
                label="✍️ 이미지 생성 프롬프트 (gpt-image-1)",
                lines=7,
                interactive=False,
                placeholder="생성된 프롬프트가 여기에 나타나요 ♡",
            )

    gr.Markdown("---\n### ✨ 어떻게 만들어지나요?")
    with gr.Row():
        gr.Markdown(
            "**1단계 · 분석 🔍**  \nGPT-4o가 사진에서 머리카락, 눈, 의상, "
            "색상 팔레트 등 모든 특징을 추출해요.",
            elem_classes="step-box",
        )
        gr.Markdown(
            "**2단계 · 프롬프트 ✍️**  \nGPT-4o가 선택한 모드에 맞는 "
            "낙서 스타일 이미지 프롬프트를 작성해요.",
            elem_classes="step-box",
        )
        gr.Markdown(
            "**3단계 · 생성 🎨**  \ngpt-image-1이 스케치북 낙서 느낌의 "
            "캐릭터 시트를 그려줘요 ♡",
            elem_classes="step-box",
        )

    gr.Markdown(
        "---\n"
        "<p style='text-align:center; color:#c084fc; font-size:0.85rem;'>"
        "Made with ♡ using Gradio · OpenAI GPT-4o · gpt-image-1 · Hugging Face Spaces"
        "</p>"
    )

    generate_btn.click(
        fn=run_pipeline,
        inputs=[image_input, mode_selector],
        outputs=[image_output, analysis_out, prompt_out],
    )


# Docker / HF Spaces: bind to 0.0.0.0 on port 7860
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    theme=gr.themes.Soft(
        primary_hue="pink",
        secondary_hue="purple",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Nunito"), "sans-serif"],
        radius_size=gr.themes.sizes.radius_lg,
    ),
    css=CSS,
)
