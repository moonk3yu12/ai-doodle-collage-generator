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
        quality="high",
    )
    image_bytes = base64.b64decode(resp.data[0].b64_json)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


# ── Gradio handler ─────────────────────────────────────────────────────────────

def run_pipeline(image: Image.Image, mode: str, progress=gr.Progress()):
    if image is None:
        raise gr.Error("이미지를 먼저 업로드해주세요.")

    try:
        client = get_client()
    except ValueError as e:
        raise gr.Error(str(e))

    try:
        progress(0.10, desc="캐릭터 특징 분석 중...")
        analysis = analyze_character(client, image)

        progress(0.45, desc="이미지 생성 프롬프트 작성 중...")
        prompt = build_doodle_prompt(client, analysis, mode)

        progress(0.70, desc="캐릭터 시트 생성 중...")
        sheet = generate_sheet(client, prompt)

        progress(1.00, desc="완료!")
        return sheet, analysis, prompt

    except openai.AuthenticationError:
        raise gr.Error("OpenAI API 키가 올바르지 않습니다. OPENAI_API_KEY 시크릿을 확인해주세요.")
    except openai.RateLimitError:
        raise gr.Error("OpenAI 요청 한도에 도달했습니다. 잠시 후 다시 시도해주세요.")
    except openai.BadRequestError as e:
        raise gr.Error(f"OpenAI가 요청을 거부했습니다: {e}. 다른 이미지로 시도해보세요.")
    except Exception as e:
        raise gr.Error(f"예기치 않은 오류가 발생했습니다: {e}")


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

with gr.Blocks(title="AI 두들 캐릭터 시트 생성기") as demo:

    with gr.Column(elem_id="app-header"):
        gr.Markdown(
            "# ✨ AI 두들 캐릭터 시트 생성기\n"
            "캐릭터나 인물 사진을 업로드하면 → 카와이 두들 콜라주 캐릭터 시트를 자동으로 만들어드려요"
        )

    gr.Markdown("---")

    with gr.Row(equal_height=True):

        with gr.Column(scale=1):
            gr.Markdown("### 📸 이미지 업로드")
            image_input = gr.Image(
                type="pil",
                label="캐릭터 또는 인물 사진",
                height=300,
            )
            mode_selector = gr.Radio(
                choices=[
                    ("🎨 전체 캐릭터 시트", "Full Character Sheet"),
                    ("🖼️ 포트레이트 낙서", "Portrait Doodle"),
                    ("👗 상반신 캐릭터", "Upper Body Character"),
                    ("🌟 치비 스티커", "Chibi Sticker"),
                    ("✨ 심플 클린 초상화", "Simple Clean Portrait"),
                ],
                value="Full Character Sheet",
                label="🎭 생성 모드 선택",
            )
            generate_btn = gr.Button(
                "🎨  캐릭터 시트 생성",
                variant="primary",
                elem_id="generate-btn",
            )
            gr.Markdown(
                "**Tip:** 정면 사진일수록 결과가 좋아요.  \n"
                "애니 캐릭터, 게임 캐릭터, 오리지널 캐릭터, 실사 인물 모두 가능해요!",
                elem_classes="tip-box",
            )

        with gr.Column(scale=1):
            gr.Markdown("### 🖼️ 생성된 캐릭터 시트")
            image_output = gr.Image(
                type="pil",
                label="생성된 캐릭터 시트",
                height=460,
            )

    with gr.Accordion("📋 분석 결과 및 프롬프트", open=False):
        with gr.Row():
            analysis_out = gr.Textbox(
                label="🔍 캐릭터 분석 (GPT-4o Vision)",
                lines=7,
                interactive=False,
                placeholder="생성 후 분석 결과가 여기에 표시됩니다…",
            )
            prompt_out = gr.Textbox(
                label="✍️ 이미지 생성 프롬프트 (gpt-image-1)",
                lines=7,
                interactive=False,
                placeholder="생성된 프롬프트가 여기에 표시됩니다…",
            )

    gr.Markdown("---\n### 💡 작동 방식")
    with gr.Row():
        gr.Markdown(
            "**1 · 분석**  \nGPT-4o Vision이 사진에서 헤어, 눈, 의상, "
            "색상 팔레트 등 모든 시각적 특징을 구조화하여 추출합니다.",
            elem_classes="step-box",
        )
        gr.Markdown(
            "**2 · 프롬프트**  \nGPT-4o가 분석 결과와 선택한 생성 모드를 바탕으로 "
            "최적화된 이미지 생성 프롬프트를 작성합니다.",
            elem_classes="step-box",
        )
        gr.Markdown(
            "**3 · 생성**  \ngpt-image-1이 선택한 모드에 맞춰 "
            "포즈, 손글씨 주석, 낙서 장식이 가득한 캐릭터 시트를 렌더링합니다.",
            elem_classes="step-box",
        )

    gr.Markdown(
        "---\n"
        "*[Gradio](https://gradio.app) · "
        "[OpenAI GPT-4o](https://platform.openai.com) · "
        "[gpt-image-1](https://platform.openai.com/docs/guides/images) · "
        "[Hugging Face Spaces](https://huggingface.co/spaces) 에서 호스팅*"
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
    theme=gr.themes.Soft(primary_hue="pink", secondary_hue="purple"),
    css=CSS,
)
