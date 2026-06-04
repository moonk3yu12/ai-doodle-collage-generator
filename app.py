import gradio as gr
import openai
import base64
import os
import io
import pathlib
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


# ── Style reference system ─────────────────────────────────────────────────────

STYLES_DIR = pathlib.Path("styles")
STYLE_REF_FILE = pathlib.Path("style_reference.txt")

_STYLE_ANALYSIS_PROMPT = (
    "Analyze the VISUAL STYLE of this artwork only.\n"
    "Do NOT describe any character's identity, name, specific outfit, or specific weapons.\n"
    "Focus ONLY on artistic style elements:\n\n"
    "- LINEART: pen type (ballpoint, marker, etc.), line quality, weight, consistency\n"
    "- SKETCH QUALITY: rough/clean, sketch marks visible, underdrawing, ink blobs\n"
    "- DOODLE DENSITY: how packed the page is, overlap, spacing\n"
    "- COLORING STYLE: medium used, coverage, hatching, messiness\n"
    "- PAGE LAYOUT: collage, sticker sheet, reference sheet, scatter arrangement\n"
    "- ANNOTATION STYLE: handwritten notes, arrows, labels, text style\n"
    "- SYMBOLS & DECORATIONS: hearts, stars, speech bubbles, stickers, barcodes\n"
    "- COLOR PALETTE USAGE: limited/wide, dominant tones, warm/cool/pastel\n"
    "- OVERALL AESTHETIC: fan art, notebook doodle, amateur, polished, chaotic\n\n"
    "Output a concise bulleted list of style observations only. No character details."
)

_STYLE_SYNTHESIS_PROMPT = (
    "Below are visual style analyses of 10 artworks that share a common aesthetic.\n"
    "Your task: distill a single compact master style description.\n\n"
    "Process:\n"
    "1. Identify traits that appear in MOST samples (not just one or two).\n"
    "2. Deduplicate — if multiple samples mention the same trait, write it ONCE.\n"
    "3. Remove ALL character-specific details:\n"
    "   - No character names, game names, franchise names\n"
    "   - No specific outfits, hairstyles, or weapons\n"
    "   - No colors tied to a specific character\n"
    "4. Keep ONLY universal, reusable style descriptors.\n"
    "5. Be concise — target 15 to 25 lines total.\n\n"
    "Output format — start with 'STYLE REFERENCE', then one descriptor per line:\n\n"
    "STYLE REFERENCE\n"
    "[descriptor]\n"
    "[descriptor]\n"
    "...\n\n"
    "The result must be a distilled master style description, "
    "not a collection of all observations. Maximum 600 tokens."
)

_style_cache: str = ""


def load_style_reference() -> str:
    global _style_cache
    if STYLE_REF_FILE.exists():
        _style_cache = STYLE_REF_FILE.read_text(encoding="utf-8").strip()
    else:
        _style_cache = ""
    return _style_cache


def run_generate_style_reference(progress=gr.Progress()):
    try:
        client = get_client()
    except ValueError as e:
        raise gr.Error(str(e))

    if not STYLES_DIR.exists():
        raise gr.Error(
            "styles/ 폴더가 없어요. 프로젝트 루트에 styles/ 폴더를 만들고 "
            "PNG/JPG 샘플 이미지를 10장 넣어주세요."
        )

    image_paths = sorted([
        *STYLES_DIR.glob("*.png"),
        *STYLES_DIR.glob("*.jpg"),
        *STYLES_DIR.glob("*.jpeg"),
    ])[:10]

    if not image_paths:
        raise gr.Error("styles/ 폴더에 이미지가 없어요. PNG/JPG 파일을 넣어주세요.")

    try:
        analyses = []
        for i, img_path in enumerate(image_paths):
            progress(
                (i + 1) / (len(image_paths) + 2),
                desc=f"샘플 {i+1}/{len(image_paths)} 스타일 분석 중...",
            )
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((512, 512))
            b64 = pil_to_base64(img)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a visual style analyst. "
                            "Describe only the artistic style of images, "
                            "never the character identity or content."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _STYLE_ANALYSIS_PROMPT},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    },
                ],
                max_tokens=400,
            )
            analyses.append(resp.choices[0].message.content.strip())

        progress(0.92, desc="공통 스타일 추출 중...")
        combined = "\n\n---\n\n".join(
            f"[Sample {i+1}]\n{a}" for i, a in enumerate(analyses)
        )
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"{_STYLE_SYNTHESIS_PROMPT}\n\n{combined}",
            }],
            max_tokens=600,
        )
        style_ref = resp.choices[0].message.content.strip()

        STYLE_REF_FILE.write_text(style_ref, encoding="utf-8")
        global _style_cache
        _style_cache = style_ref

        progress(1.0, desc="완료!")
        return style_ref, f"✅ {len(image_paths)}장 분석 완료 · style_reference.txt 저장됨"

    except gr.Error:
        raise
    except openai.AuthenticationError:
        raise gr.Error("Invalid OpenAI API key. Check your OPENAI_API_KEY secret.")
    except openai.RateLimitError:
        raise gr.Error("OpenAI rate limit hit. Please wait and try again.")
    except Exception as e:
        raise gr.Error(f"Style reference 생성 실패: {e}")


# Load at startup (if style_reference.txt exists in the repo)
load_style_reference()


# ── Generation mode configs ────────────────────────────────────────────────────

MODE_CONFIGS = {
    "Full Character Sheet": {
        "brief": (
            "multiple poses and expressions of the same character across a plain white page, "
            "large main portrait center, full body standing pose, several chibi versions, "
            "close-up face expressions (happy, sleepy, embarrassed, angry, smug), "
            "color palette swatches in a corner, barcode sticker, "
            "handwritten character name and annotations with small arrows, "
            "hearts ♡ stars ★ sparkles ✦ speech bubbles scattered around"
        ),
    },
    "Portrait Doodle": {
        "brief": (
            "single upper-body portrait centered on plain white page, "
            "large expressive face, outfit top visible, "
            "small hearts and stars framing the drawing, "
            "handwritten name or nickname label nearby"
        ),
    },
    "Upper Body Character": {
        "brief": (
            "waist-up character illustration centered on plain white page, "
            "outfit details clearly visible, "
            "subtle decorative accents at the edges: stars, hearts, sparkles"
        ),
    },
    "Chibi Sticker": {
        "brief": (
            "4 to 6 chibi versions of the character on plain white background, "
            "different expressions and mini poses, scattered like a sticker sheet, "
            "hearts and stars between them, bold outlines, flat marker fills"
        ),
    },
    "Simple Clean Portrait": {
        "brief": (
            "single character portrait centered on plain white page, "
            "clean and minimal, "
            "only a few small star or heart accents nearby"
        ),
    },
}


# ── Pipeline steps ─────────────────────────────────────────────────────────────

_REFUSAL_PHRASES = (
    "i'm sorry", "i can't assist", "i cannot assist",
    "i'm unable", "i cannot help", "i can't help",
    "i'm not able", "unable to", "not able to",
)


def _is_refusal(text: str) -> bool:
    low = text.lower()
    return any(phrase in low for phrase in _REFUSAL_PHRASES)


_SYSTEM_MSG = (
    "You are a professional visual design analyst specializing in character art, "
    "illustration, and concept art. "
    "Your task is to describe the visual design elements of artwork — colors, shapes, "
    "clothing, accessories, weapons, and stylistic features. "
    "You never identify real people. You only describe what is visually present in the artwork."
)


def analyze_character(client: openai.OpenAI, image: Image.Image) -> tuple[str, object]:
    """GPT-4o Vision: extract visual design traits. Retries with simpler prompt on refusal."""
    b64 = pil_to_base64(image)
    image_block = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}

    primary_prompt = (
        "This is a digital artwork / character illustration.\n"
        "Do NOT identify the character. Do NOT name any franchise or IP.\n"
        "Extract every visible design element as bullet points under each section.\n"
        "Do NOT write paragraphs. Do NOT summarize. Do NOT omit small accessories.\n"
        "Prefer exhaustive extraction over concise descriptions.\n\n"

        "HAIR\n"
        "- exact color(s) with adjectives (e.g. ice blue, silver-white with pale tips)\n"
        "- length and silhouette\n"
        "- style (loose, twin tails, braid, etc.) and how it flows\n"
        "- any visible layers, gradients, or streaks\n\n"

        "HEAD ACCESSORIES\n"
        "- every crown, tiara, horn, ornament, clip, pin, ribbon on or around the head\n"
        "- for each: shape, color, material, exact position\n\n"

        "EYES\n"
        "- exact iris color\n"
        "- pupil shape (round, slit, star, cross, etc.)\n"
        "- eye shape (almond, round, upturned, etc.)\n"
        "- lash style (thick, long, sparse)\n"
        "- any glow, gradient, or special iris pattern\n\n"

        "FACE\n"
        "- skin tone\n"
        "- face shape\n"
        "- every marking: freckles, scars, blush marks, tattoos, runes, beauty marks\n"
        "- earrings: style, color, length, material\n\n"

        "OUTFIT\n"
        "- neckline and collar style\n"
        "- top / bodice: garment type, color, fabric texture, patterns, embroidery, cutouts\n"
        "- necklaces, chokers, pendants: shape, color, material\n"
        "- chest emblem or insignia: shape, color, position\n"
        "- shoulder armor or pauldrons: shape, color, engravings\n"
        "- sleeves: length, style, color, any cuffs or arm guards\n"
        "- gloves: length, color, material, finger coverage\n"
        "- belt or waist piece: width, color, buckle or clasp design\n"
        "- cape or coat: shape, length, color, lining color, clasp or brooch\n"
        "- any symbols, runes, or crests: shape and location on outfit\n\n"

        "SKIRT\n"
        "- type (full skirt, layered, shorts, pants, wrap, etc.)\n"
        "- length\n"
        "- every layer with its color and material\n"
        "- frills, slits, overlays, or decorative edges\n\n"

        "LEGS\n"
        "- stockings or leggings: color, pattern (striped, lace, plain, sheer), height\n"
        "- any armor or decorative pieces on the legs\n\n"

        "FOOTWEAR\n"
        "- boot or shoe style (knee-high, ankle, platform, heeled, etc.)\n"
        "- color of each section\n"
        "- heel type and height\n"
        "- buckles, laces, straps, or armor plates\n"
        "- toe shape\n\n"

        "HELD OBJECTS\n"
        "- for every staff, wand, sword, shield, orb, or carried object:\n"
        "  - overall shape and proportions\n"
        "  - color of each part\n"
        "  - material appearance (metal, crystal, wood, cloth, etc.)\n"
        "  - any glowing elements, gems, engravings, or unique decorations\n\n"

        "SPECIAL EFFECTS\n"
        "- any aura, glow, energy trails, floating elements, particles\n"
        "- for each: color, shape, position relative to character\n\n"

        "COLOR PALETTE\n"
        "- list 6 to 8 specific color names used in this design\n\n"

        "Output only the sections above with bullet points. No commentary. No prose."
    )

    fallback_prompt = (
        "This is a fictional character illustration.\n"
        "List every visible design element as bullet points under these sections:\n"
        "HAIR / HEAD ACCESSORIES / EYES / FACE / OUTFIT / SKIRT / LEGS / FOOTWEAR "
        "/ HELD OBJECTS / SPECIAL EFFECTS / COLOR PALETTE\n"
        "Do not summarize. Do not omit accessories. Do not identify anyone."
    )

    simple_prompt = (
        "List the visual design of this illustrated character using bullet points. "
        "Sections: hair, head accessories, eyes, face, outfit, skirt, legs, footwear, "
        "held objects, special effects, color palette. "
        "Be specific. Do not identify anyone."
    )

    for prompt_text in (primary_prompt, fallback_prompt, simple_prompt):
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _SYSTEM_MSG},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt_text}, image_block],
                },
            ],
            max_tokens=1000,
        )
        result = resp.choices[0].message.content.strip()
        if not _is_refusal(result):
            return result, resp.usage

    raise gr.Error("Character analysis failed. Please try another image.")


def build_doodle_prompt(analysis: str, mode: str) -> str:
    """Build prompt optimised for images.edit: character preservation first, style second."""
    layout = MODE_CONFIGS.get(mode, MODE_CONFIGS["Full Character Sheet"])["brief"]

    # ── 1. Character preservation block (FIRST — highest weight in edit mode) ──
    CHARACTER_BLOCK = (
        "Redraw the character from the reference image as a messy hand-drawn doodle collage.\n\n"
        "PRESERVE EXACTLY from the reference image — do not change any of these:\n"
        f"{analysis}\n\n"
        "CRITICAL PRESERVATION RULES:\n"
        "- Keep the exact hair color, style, and all hair ornaments/crowns/accessories\n"
        "- Keep the exact eye color and shape\n"
        "- Keep every piece of the outfit with its original colors and details\n"
        "- Keep all accessories, jewelry, armor, and decorative elements\n"
        "- Keep all held objects and equipment with their original shape and colors\n"
        "- Keep the character's full color palette — do not substitute any colors\n"
        "- This must be recognizably the SAME character as in the reference image\n\n"
    )

    # ── 2. Layout block ────────────────────────────────────────────────────────
    LAYOUT_BLOCK = (
        f"PAGE LAYOUT: {layout}.\n\n"
        "DENSITY REQUIREMENTS — the entire canvas must be overcrowded with content:\n"
        "- 1 large full-body drawing of the character (dominant, center or near-center)\n"
        "- at least 2 medium upper-body or bust portraits\n"
        "- at least 8 facial expression drawings (happy, sad, angry, surprised, embarrassed, sleepy, smug, crying)\n"
        "- at least 3 chibi versions in different poses\n"
        "- 1 color palette swatch box with the character's colors labeled\n"
        "- 20 or more small doodles, sketches, and decorative elements filling every gap\n"
        "- handwritten notes, labels, and annotations scattered across the entire page\n\n"
        "EMPTY SPACE RULES:\n"
        "- NO empty space anywhere on the canvas\n"
        "- Every corner must contain at least one of: sketch, arrow, star, heart, note, reaction face, tiny doodle, or decorative mark\n"
        "- All gaps between drawings must be filled with hearts ♡, stars ★, sparkles ✦, arrows →, or scribbled words\n"
        "- The page must feel completely packed, like a fan ran out of space while drawing\n\n"
    )

    # ── 3. Style block (AFTER character — concise, positive framing) ──────────
    STYLE_BLOCK = (
        "ART STYLE — redraw everything in this style:\n"
        "Rough messy doodle collage on pure white #FFFFFF background. "
        "Amateur ballpoint pen sketch texture. "
        "Wobbly uneven lineart, coloring outside the lines, scratchy hatching. "
        "Imperfect chibi-like proportions, big expressive heads. "
        "Page completely packed — multiple rough sketches of the same character, "
        "some tilted, some overlapping, some half-finished. "
        "Hand-drawn decorations everywhere: "
        "♡♡♡ hearts, ★★ stars, ✦ sparkles, messy arrows → with handwritten labels, "
        "speech bubbles (owo, uwu, !!, ??), barcode sticker, color swatches, "
        "chibi reaction faces, character name in bubble letters. "
        "Feels like a devoted fan's chaotic sketchbook page. "
        "No paper texture. White background only."
    )

    return f"{CHARACTER_BLOCK}{LAYOUT_BLOCK}{STYLE_BLOCK}"


def generate_sheet(
    client: openai.OpenAI,
    prompt: str,
    quality: str,
    reference_image: Image.Image | None = None,
) -> tuple[Image.Image, object]:
    """gpt-image-1: edit mode when reference image provided, generate mode otherwise."""
    if reference_image is not None:
        # images.edit — pass original image so character design is preserved
        buf = io.BytesIO()
        reference_image.save(buf, format="PNG")
        buf.seek(0)
        buf.name = "reference.png"
        resp = client.images.edit(
            model="gpt-image-1",
            image=buf,
            prompt=prompt,
            size="1024x1024",
            quality=quality,
            input_fidelity="high",
        )
    else:
        resp = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
            quality=quality,
        )
    image_bytes = base64.b64decode(resp.data[0].b64_json)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB"), resp.usage


# ── Gradio handler ─────────────────────────────────────────────────────────────

def run_pipeline(image: Image.Image, mode: str, quality: str, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Please upload an image first.")

    try:
        client = get_client()
    except ValueError as e:
        raise gr.Error(str(e))

    try:
        progress(0.10, desc="Analyzing character features...")
        analysis, u1 = analyze_character(client, image)

        progress(0.45, desc="Building image prompt...")
        prompt = build_doodle_prompt(analysis, mode)

        progress(0.70, desc="Generating character sheet...")
        sheet, u3 = generate_sheet(client, prompt, quality, reference_image=image)

        progress(1.00, desc="Done!")

        style_ref_status = (
            f"✅ Style reference active ({len(_style_cache)} chars)"
            if _style_cache
            else "⚠️ No style reference — use the Style Reference panel to generate one"
        )

        token_summary = (
            f"Step 1 · Analyze   (GPT-4o Vision)\n"
            f"  in: {u1.prompt_tokens:,}   out: {u1.completion_tokens:,}   total: {u1.total_tokens:,}\n\n"
            f"Step 2 · Prompt    (template — no API call)\n"
            f"  Full analysis injected directly. 0 tokens used.\n\n"
            f"Step 3 · Generate  (gpt-image-1 — image edit mode)\n"
            f"  in: {u3.input_tokens:,}   out: {u3.output_tokens:,}   total: {u3.total_tokens:,}\n\n"
            f"{'─' * 40}\n"
            f"Grand total: {u1.total_tokens + u3.total_tokens:,} tokens\n\n"
            f"{style_ref_status}"
        )

        style_ref_display = _style_cache if _style_cache else "(style reference not loaded)"

        return sheet, analysis, prompt, token_summary, style_ref_display

    except gr.Error:
        raise
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

body, .gradio-container, .gradio-container * {
    font-family: 'Nunito', sans-serif !important;
    box-sizing: border-box;
}
body, .gradio-container {
    background: linear-gradient(135deg, #fff0f6 0%, #f5f0ff 50%, #f0f4ff 100%) !important;
    min-height: 100vh;
}
.block, .form, .wrap, .panel,
.gradio-container .block,
section.block, div.block,
.gradio-container .wrap {
    background: white !important;
    border-color: #f3e8ff !important;
}
.image-container, .upload-container,
div[data-testid="image"],
div[data-testid="image"] > div,
.svelte-p3y7hu, .empty {
    background: #fdf4ff !important;
    border-color: #e9d5ff !important;
}
.upload-container, .upload-button,
.wrap.svelte-i3tvor {
    border: 2px dashed #d8b4fe !important;
    border-radius: 16px !important;
    background: #fdf4ff !important;
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
}
#app-header p { color: #a78bca; font-size: 1rem; font-weight: 600; }
#left-panel, #right-panel {
    background: white !important;
    border: 2px solid #f3e8ff !important;
    border-radius: 24px !important;
    box-shadow: 0 4px 24px rgba(192, 132, 252, 0.12) !important;
    padding: 1.4rem !important;
}
#generate-btn {
    background: linear-gradient(135deg, #ff6eb4, #c084fc, #818cf8) !important;
    color: white !important;
    border: none !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    border-radius: 50px !important;
    padding: 0.75rem 1.5rem !important;
    width: 100% !important;
    box-shadow: 0 4px 16px rgba(192, 132, 252, 0.4) !important;
    transition: all 0.2s ease !important;
}
#generate-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(192, 132, 252, 0.55) !important;
    opacity: 1 !important;
}
#style-ref-btn {
    border: 2px solid #c084fc !important;
    color: #c084fc !important;
    border-radius: 50px !important;
    font-weight: 700 !important;
}
.tip-box {
    background: linear-gradient(135deg, #fdf4ff, #f5f0ff) !important;
    border: 1.5px solid #e9d5ff !important;
    border-radius: 16px !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    color: #c084fc !important;
}
.tip-box p, .tip-box strong, .tip-box * { color: #c084fc !important; }
.step-box {
    background: white !important;
    border: 2px solid #e9d5ff !important;
    border-radius: 20px !important;
    padding: 1.2rem !important;
    text-align: center !important;
    box-shadow: 0 2px 12px rgba(192, 132, 252, 0.1) !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    color: #c084fc !important;
}
.step-box p, .step-box strong, .step-box * { color: #c084fc !important; }
.step-box:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 20px rgba(192, 132, 252, 0.2) !important;
}
.gradio-container h3, .gradio-container h2 {
    color: #c084fc !important;
    font-weight: 800 !important;
}
details summary, details summary span,
.accordion-header, .label-wrap span {
    color: #c084fc !important;
    font-weight: 700 !important;
}
.gradio-container p, .gradio-container .prose p { color: #c084fc !important; }
hr {
    border: none !important;
    border-top: 2px dashed #f0e4ff !important;
    margin: 1rem 0 !important;
}
textarea, input[type="text"] {
    border-radius: 14px !important;
    border: 1.5px solid #e9d5ff !important;
    background: #fdf4ff !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.9rem !important;
    color: #4c1d95 !important;
}
details, .accordion {
    border-radius: 16px !important;
    border: 2px solid #f3e8ff !important;
    background: white !important;
    overflow: hidden !important;
}
input[type="radio"] + span, .wrap label span {
    font-weight: 600 !important;
    color: #7c3aed !important;
}
label span, .block > label > span {
    color: #9333ea !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
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

    with gr.Row(equal_height=False):

        with gr.Column(scale=1, elem_id="left-panel"):
            gr.Markdown("### 📸 사진 올리기")
            image_input = gr.Image(
                type="pil",
                label="캐릭터 또는 인물 사진",
                height=280,
            )
            mode_selector = gr.Radio(
                choices=[
                    ("🎨 풀 캐릭터 시트", "Full Character Sheet"),
                    ("🖼️ 포트레이트 낙서", "Portrait Doodle"),
                    ("👗 상반신 캐릭터", "Upper Body Character"),
                    ("🌟 치비 스티커", "Chibi Sticker"),
                    ("✨ 심플 클린 포트레이트", "Simple Clean Portrait"),
                ],
                value="Full Character Sheet",
                label="🎭 생성 모드",
            )
            quality_selector = gr.Radio(
                choices=[
                    ("💸 Low — 약 15원", "low"),
                    ("⚡ Medium — 약 58원", "medium"),
                    ("✨ High — 약 230원", "high"),
                ],
                value="medium",
                label="🖼️ 이미지 퀄리티",
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

        with gr.Column(scale=1, elem_id="right-panel"):
            gr.Markdown("### 🎨 완성된 낙서 시트 ♡")
            image_output = gr.Image(
                type="pil",
                label="생성된 캐릭터 시트",
                height=500,
            )

    # ── Style Reference panel ──────────────────────────────────────────────────
    with gr.Accordion("🎨 Style Reference (스타일 레퍼런스)", open=False):
        gr.Markdown(
            "styles/ 폴더에 샘플 이미지 10장을 넣고 아래 버튼을 누르면 "
            "공통 스타일을 추출해서 모든 이미지 생성 프롬프트 맨 앞에 자동 삽입해요.\n\n"
            "**사용법:** `styles/sample01.png` ~ `styles/sample10.png` 형식으로 넣기"
        )
        with gr.Row():
            style_ref_btn = gr.Button(
                "🖼️ styles/ 폴더에서 Style Reference 생성",
                variant="secondary",
                elem_id="style-ref-btn",
            )
            style_ref_status = gr.Textbox(
                label="상태",
                interactive=False,
                lines=1,
                placeholder="버튼을 눌러 style_reference.txt를 생성하세요",
                value=f"✅ style_reference.txt 로드됨 ({len(_style_cache)} chars)" if _style_cache else "⚠️ style_reference.txt 없음",
            )
        style_ref_content = gr.Textbox(
            label="현재 Style Reference 내용",
            value=_style_cache if _style_cache else "(비어있음 — 생성 버튼을 눌러주세요)",
            lines=12,
            interactive=False,
        )

    # ── Analysis, Prompt & Token Usage panel ──────────────────────────────────
    with gr.Accordion("📋 Analysis, Prompt & Token Usage", open=False):
        style_ref_out = gr.Textbox(
            label="🎨 삽입된 Style Reference",
            lines=5,
            interactive=False,
            placeholder="생성 후 여기에 실제 삽입된 style reference가 표시됩니다…",
        )
        with gr.Row():
            analysis_out = gr.Textbox(
                label="🔍 Character Analysis (GPT-4o Vision)",
                lines=7,
                interactive=False,
                placeholder="Analysis will appear here after generation…",
            )
            prompt_out = gr.Textbox(
                label="✍️ Image Generation Prompt (gpt-image-1)",
                lines=7,
                interactive=False,
                placeholder="Generated prompt will appear here…",
            )
        token_out = gr.Textbox(
            label="📊 Token Usage",
            lines=7,
            interactive=False,
            placeholder="Token usage will appear here after generation…",
        )

    gr.Markdown("---\n### ✨ 어떻게 만들어지나요?")
    with gr.Row():
        gr.Markdown(
            "**1단계 · 분석 🔍**  \nGPT-4o가 사진에서 머리카락, 눈, 의상, "
            "색상 팔레트 등 모든 특징을 추출해요.",
            elem_classes="step-box",
        )
        gr.Markdown(
            "**2단계 · 프롬프트 ✍️**  \nStyle Reference + 캐릭터 분석 결과를 "
            "합쳐서 이미지 생성 프롬프트를 만들어요.",
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

    # ── Button wiring ──────────────────────────────────────────────────────────
    generate_btn.click(
        fn=run_pipeline,
        inputs=[image_input, mode_selector, quality_selector],
        outputs=[image_output, analysis_out, prompt_out, token_out, style_ref_out],
    )

    style_ref_btn.click(
        fn=run_generate_style_reference,
        inputs=[],
        outputs=[style_ref_content, style_ref_status],
    )


# Docker / HF Spaces: bind to 0.0.0.0 on port 7860
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    theme=gr.themes.Soft(
        primary_hue="pink",
        secondary_hue="purple",
        neutral_hue="pink",
        font=[gr.themes.GoogleFont("Nunito"), "sans-serif"],
        radius_size=gr.themes.sizes.radius_lg,
    ),
    css=CSS,
)
