import gradio as gr
import openai
import base64
import os
import io
import pathlib
import requests
from PIL import Image, ImageDraw


# ── OpenAI client ──────────────────────────────────────────────────────────────

_api_key_at_startup = os.environ.get("OPENAI_API_KEY", "")
print(
    f"[STARTUP] OPENAI_API_KEY present={bool(_api_key_at_startup)} "
    f"length={len(_api_key_at_startup)} "
    f"prefix={_api_key_at_startup[:7] if len(_api_key_at_startup) >= 7 else '(too short)'}",
    flush=True,
)


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

# DEBUG: set False to disable style reference images (character-only mode)
_STYLE_IMAGES_ENABLED = True

# Load 2 random style reference images from styles/ at startup
import random as _random

_STYLE_IMAGES: list = []
if not _STYLE_IMAGES_ENABLED:
    print("[STARTUP] Style images DISABLED (debug flag) — single-image mode", flush=True)
else:
    _style_paths = sorted([
        *STYLES_DIR.glob("*.png"),
        *STYLES_DIR.glob("*.jpg"),
        *STYLES_DIR.glob("*.jpeg"),
    ]) if STYLES_DIR.exists() else []
    if _style_paths:
        _STYLE_IMAGES = [
            Image.open(p).convert("RGB")
            for p in _random.sample(_style_paths, min(2, len(_style_paths)))
        ]
        print(f"[STARTUP] Style images loaded: {len(_STYLE_IMAGES)} image(s)", flush=True)
    else:
        print("[STARTUP] No style images found — single-image edit mode", flush=True)

# Load gallery images from test/ folder at startup
GALLERY_DIR = pathlib.Path("test")
_gallery_images: list[str] = []
if GALLERY_DIR.exists():
    _raw = sorted(
        [*GALLERY_DIR.glob("*.png"), *GALLERY_DIR.glob("*.jpg"), *GALLERY_DIR.glob("*.jpeg")],
        key=lambda p: (int(p.stem) if p.stem.isdigit() else float("inf"), p.stem),
    )
    _gallery_images = [str(p) for p in _raw]
    print(f"[STARTUP] Gallery images loaded: {len(_gallery_images)} image(s)", flush=True)


# ── Goods simulation config ───────────────────────────────────────────────────

GOODS_DIR = pathlib.Path("goods")

# area: (x, y, width, height) — where the character image is pasted on the template
GOODS_CONFIG: dict[str, dict] = {
    "📸 포토카드": {"file": "photocard.png", "canvas": (300, 420), "area": (30,  50, 240, 320)},
    "🌟 스티커":   {"file": "sticker.png",   "canvas": (400, 400), "area": (60,  60, 280, 280)},
    "🔑 키링":     {"file": "keyring.png",   "canvas": (280, 330), "area": (55,  65, 170, 200)},
    "👕 옷":       {"file": "clothes.png",   "canvas": (480, 560), "area": (140, 120, 200, 240)},
}

# ── Generation mode configs ────────────────────────────────────────────────────

MODE_CONFIGS = {
    "Full Character Sheet": {
        "brief": (
            "multiple poses and expressions of the same character across a plain white page, "
            "large main portrait center, full body standing pose, several chibi versions, "
            "close-up face expressions (happy, sleepy, embarrassed, angry, smug), "
            "color palette swatches in a corner, barcode sticker, "
            "hearts ♡ stars ★ sparkles ✦ scattered around"
        ),
    },
    "Portrait Doodle": {
        "brief": (
            "single upper-body portrait centered on plain white page, "
            "large expressive face, outfit top visible, "
            "small hearts and stars framing the drawing"
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
        "- exact iris color (specific name required: cerulean, amber, violet — never just 'blue' or 'light')\n"
        "- pupil shape (round, slit, star, cross, etc.)\n"
        "- eye shape (almond, round, upturned, downturned, etc.)\n"
        "- lash style (thick, long, sparse)\n"
        "- any glow, gradient, or special iris pattern\n\n"

        "FACE\n"
        "- skin tone (e.g. fair, warm ivory, light tan)\n"
        "- face shape (oval, angular, round, heart-shaped, square — required)\n"
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

        "CHARACTER NAME\n"
        "- If a character name or title is written as visible text in this image, transcribe it exactly\n"
        "- If no name is visible as text in the image, write: ???\n"
        "- Do NOT use character recognition — only transcribe visible text\n\n"

        "CRITICAL OUTPUT FORMAT — follow exactly:\n"
        "- Section names must be plain UPPERCASE on their own line (e.g. HAIR)\n"
        "- Each detail on its own line starting with '- '\n"
        "- No markdown, no bold, no asterisks, no colons after section names\n"
        "- Example:\nHAIR\n- ice blue with silver tips\n- shoulder length, twin tails\n\nEYES\n- cerulean blue iris\n- almond-shaped\n- round pupil"
    )

    fallback_prompt = (
        "This is a fictional character illustration.\n"
        "Describe the visual design using the exact format below.\n"
        "Plain text only — no markdown, no bold, no asterisks.\n"
        "Section names in UPPERCASE on their own line, then bullet points.\n\n"

        "HAIR\n"
        "- exact color with adjectives (e.g. ice blue, sandy blonde)\n"
        "- length and style\n\n"

        "HEAD ACCESSORIES\n"
        "- list every item on or around the head with color and position\n\n"

        "EYES\n"
        "- exact iris color — use a specific name (cerulean, amber, violet, crimson)\n"
        "- eye shape (almond, round, upturned, downturned)\n"
        "- pupil shape (round, slit, star)\n\n"

        "FACE\n"
        "- skin tone\n"
        "- face shape (oval, angular, round, heart-shaped, square)\n"
        "- any markings, blush, freckles, or scars\n\n"

        "OUTFIT\n"
        "- garment type, colors, and key details\n\n"

        "SKIRT\n"
        "- type, length, colors\n\n"

        "LEGS\n"
        "- stockings or leggings with color and pattern\n\n"

        "FOOTWEAR\n"
        "- style, color, details\n\n"

        "HELD OBJECTS\n"
        "- each object: shape, color, material\n\n"

        "SPECIAL EFFECTS\n"
        "- aura or energy: color and position\n\n"

        "COLOR PALETTE\n"
        "- list 5 to 7 specific color names (e.g. cerulean blue, ivory white, gold)\n\n"

        "CHARACTER NAME\n"
        "- Visible name text in image (transcribe only), or ???\n\n"

        "Do not identify the character. Do not name any franchise."
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


def _parse_analysis_sections(analysis: str) -> dict[str, str]:
    """Split GPT-4o analysis into named sections.

    Handles two formats:
      Format A (primary prompt)  — plain uppercase headers:  HAIR\\n- ...
      Format B (fallback prompt) — markdown bold headers:    - **Hair**:\\n  - ...
    """
    import re
    result = {}

    # Format A: plain uppercase line (e.g. "HAIR", "HEAD ACCESSORIES")
    plain_matches = list(re.finditer(r'^([A-Z][A-Z &/]+)$', analysis, re.MULTILINE))
    if plain_matches:
        for i, m in enumerate(plain_matches):
            key = m.group(1).strip()
            start = m.end()
            end = plain_matches[i + 1].start() if i + 1 < len(plain_matches) else len(analysis)
            result[key] = analysis[start:end].strip()
        return result

    # Format B: markdown bold header (e.g. "- **Hair**:" or "**Hair**:")
    md_matches = list(re.finditer(r'^\s*-?\s*\*\*([^*:]+?)\*\*\s*:?', analysis, re.MULTILINE))
    if md_matches:
        for i, m in enumerate(md_matches):
            key = m.group(1).strip().upper()
            start = m.end()
            end = md_matches[i + 1].start() if i + 1 < len(md_matches) else len(analysis)
            result[key] = analysis[start:end].strip()
        return result

    return result


def build_doodle_prompt(analysis: str, mode: str) -> str:
    """Build prompt optimised for images.edit: character preservation first, style second."""
    layout = MODE_CONFIGS.get(mode, MODE_CONFIGS["Full Character Sheet"])["brief"]

    # ── Parse sections early so char_name is available for zone maps ─────────
    secs = _parse_analysis_sections(analysis)
    print(f"[PARSE] sections found={len(secs)} keys={list(secs.keys())}", flush=True)
    print(f"[PARSE] analysis raw (first 300 chars): {repr(analysis[:300])}", flush=True)

    name_raw = secs.get("CHARACTER NAME", "").strip()
    name_lines = [l.strip().lstrip("- ").strip() for l in name_raw.split("\n") if l.strip()]
    char_name = name_lines[0] if name_lines else "???"
    if not char_name or char_name.lower() in ("none", "unknown", "n/a", ""):
        char_name = "???"

    _eye_lines = [l.strip().lstrip("- ").strip() for l in secs.get("EYES", "").split("\n") if l.strip()]
    eyes_info = _eye_lines[0] if _eye_lines else "?"

    _acc_lines = [l.strip().lstrip("- ").strip() for l in secs.get("HEAD ACCESSORIES", "").split("\n") if l.strip()]
    accessory_info = _acc_lines[0] if _acc_lines else "?"

    _pal_lines = [l.strip().lstrip("- ").strip() for l in secs.get("COLOR PALETTE", "").split("\n") if l.strip()]
    palette_info = " / ".join(_pal_lines[:4]) if _pal_lines else "?"

    # ── 1. Layout block (FIRST — spatial zones anchor the composition) ──────────
    _ZONE_MAPS = {
        "Full Character Sheet": (
            "READABLE TEXT RULE: only the Character Info Box (defined below) may contain printed text.\n"
            "Everywhere else: NO words, NO names, NO speech bubbles — decorative symbols only: ♡ ★ ✦ → !! ??\n\n"
            "SPATIAL ZONE MAP — fill every zone completely:\n\n"
            "TOP ZONE (upper 30%):\n"
            "  - large bust portrait on the left\n"
            "  - second bust portrait in the center-right\n"
            "  - CHARACTER INFO BOX in the upper-right corner (small thin-bordered rectangle, handwritten font):\n"
            f"      Character Info\n"
            f"      Name: {char_name}\n"
            f"      Eyes: {eyes_info}\n"
            f"      Accessory: {accessory_info}\n"
            f"      Palette: {palette_info}\n\n"
            "CENTER ZONE (middle 35%):\n"
            "  - one large full-body standing character (dominant element, center)\n\n"
            "BOTTOM ZONE (lower 35%):\n"
            "  - row of 8 expression face close-ups (happy, sad, angry, surprised, embarrassed, sleepy, smug, crying)\n"
            "  - 3 chibi full-body poses scattered across the zone\n"
            "  - color palette swatch box (no text labels)\n\n"
            "ALL FOUR CORNERS: stars ★, hearts ♡, sparkles ✦, arrows →\n"
            "ALL GAPS: fill with ♡ ★ ✦ →\n\n"
        ),
        "Upper Body Character": (
            "SPATIAL ZONE MAP:\n\n"
            "ONE SINGLE DRAWING — the entire page is filled by one bust portrait:\n"
            "  - one large BUST PORTRAIT centered and dominant\n"
            "  - shows face, hair, shoulders, and upper chest only\n"
            "  - cropped at waist level — no lower body visible\n"
            "  - no text or labels\n\n"
            "STRICT RULES:\n"
            "  Draw ONLY ONE character portrait — no additional drawings.\n"
            "  DO NOT draw legs, feet, or shoes.\n"
            "  DO NOT draw chibi versions.\n"
            "  DO NOT draw multiple portraits or expression studies.\n"
            "  DO NOT add hearts, stars, or sparkle decorations.\n\n"
        ),
        "Portrait Doodle": (
            "SPATIAL ZONE MAP:\n\n"
            "CENTER (dominant, fills upper 60%):\n"
            "  - one large expressive face and head portrait\n"
            "  - outfit collar or top just barely visible at the bottom edge\n"
            "  - no text or name labels\n\n"
            "SURROUNDING AREA:\n"
            "  - 4 to 6 smaller expression face sketches around the main portrait\n\n"
            "DO NOT draw any full-body poses or chibi versions.\n"
            "DO NOT add hearts, stars, or sparkle decorations.\n\n"
        ),
        "Chibi Sticker": (
            "SPATIAL ZONE MAP — sticker sheet, no fixed zones:\n\n"
            "FULL PAGE:\n"
            "  - 5 to 6 chibi full-body poses scattered at random angles across the entire page\n"
            "  - each chibi in a different pose or expression\n"
            "  - some slightly tilted, overlapping edges\n"
            "  - small hearts ♡ and stars ★ between each chibi\n\n"
            "ALL GAPS: fill with ♡ ★ tiny doodles\n\n"
            "DO NOT draw any full-size portraits or full-body realistic poses.\n\n"
        ),
        "Simple Clean Portrait": (
            "SPATIAL ZONE MAP:\n\n"
            "CENTER (dominant):\n"
            "  - single character portrait centered on the page\n"
            "  - clean and uncluttered — do not pack the page\n\n"
            "KEEP SPARSE — do not add hearts, stars, sparkles, or dense doodles.\n\n"
        ),
    }

    zone_map = _ZONE_MAPS.get(mode, _ZONE_MAPS["Full Character Sheet"])

    LAYOUT_BLOCK = (
        "IMAGE ROLES:\n"
        "- Image 1: CHARACTER REFERENCE — preserve hair, eyes, outfit, colors, and identity exactly\n"
        "- Images 2-3: STYLE REFERENCE — copy only the drawing style, page layout, and doodle density\n\n"
        f"PAGE LAYOUT: {layout}.\n\n"
        f"{zone_map}"
    )

    # ── 2. Character preservation block (silhouette-priority ordering) ───────
    # Priority: hair silhouette → head accessories → signature objects → colors → face
    _IDENTITY_KEYS = ["HAIR", "HEAD ACCESSORIES", "HELD OBJECTS", "COLOR PALETTE", "FACE", "EYES"]
    _COMPRESS_MAP = {"OUTFIT": 3, "SKIRT": 1, "LEGS": 1,
                     "FOOTWEAR": 1, "SPECIAL EFFECTS": 1}

    if secs:
        identity_parts = [f"{k}\n{secs[k]}" for k in _IDENTITY_KEYS if k in secs]
        identity_detail = "\n\n".join(identity_parts)

        outfit_lines = []
        for k, n in _COMPRESS_MAP.items():
            if k in secs:
                bullets = [l for l in secs[k].split("\n") if l.strip()][:n]
                outfit_lines.append(f"{k}: {' | '.join(bullets)}")
        outfit_summary = "\n".join(outfit_lines)

        analysis_block = (
            "VISUAL IDENTITY — preserve these exactly (priority order):\n"
            "1. HAIR silhouette, color, and style\n"
            "2. HEAD ACCESSORIES (every crown, hat, ribbon, ornament)\n"
            "3. SIGNATURE OBJECTS AND WEAPONS\n"
            "4. DOMINANT COLOR COMBINATION\n"
            "5. FACE and EYES\n\n"
            f"{identity_detail}\n\n"
            "OUTFIT REFERENCE (secondary):\n"
            f"{outfit_summary}\n\n"
        )
    else:
        analysis_block = f"PRESERVE EXACTLY:\n{analysis}\n\n"

    CHARACTER_BLOCK = (
        "Redraw the character from Image 1 as a messy hand-drawn doodle collage.\n\n"
        "CRITICAL PRESERVATION RULES — apply to every single drawing on the page:\n"
        "- HAIR: preserve exact silhouette, color, length, and style in every drawing\n"
        "- HEAD ACCESSORIES: every crown, hat, horn, ribbon must appear in every drawing\n"
        "- WEAPONS AND HELD OBJECTS: preserve shape and color in full-body drawings\n"
        "- COLOR PALETTE: do not substitute any colors\n"
        "- PAGE CONSISTENCY: every sketch, chibi, and expression on this page must depict\n"
        "  the SAME character — same hair, same accessories, same color scheme\n"
        "- Even chibi versions must show the exact hair silhouette and head accessories\n"
        "- This must be recognizably the SAME character as in Image 1\n\n"
        f"{analysis_block}"
    )

    # ── 3. Style block ─────────────────────────────────────────────────────────
    _decorations = (
        "Hand-drawn decorations everywhere: "
        "♡♡♡ hearts, ★★ stars, ✦ sparkles, messy arrows →, barcode sticker. "
        "NO readable text, NO speech bubbles, NO handwritten words. "
    ) if mode in ("Full Character Sheet", "Chibi Sticker") else ""

    STYLE_BLOCK = (
        "ART STYLE — redraw everything in this style:\n"
        "Rough messy doodle collage on pure white #FFFFFF background. "
        "Amateur ballpoint pen sketch texture. "
        "Wobbly uneven lineart, coloring outside the lines, scratchy hatching. "
        "Imperfect chibi-like proportions, big expressive heads. "
        "Multiple rough sketches of the same character, "
        "some tilted, some overlapping, some half-finished. "
        f"{_decorations}"
        "Feels like a devoted fan's chaotic sketchbook page. "
        "No paper texture. White background only."
    )

    return f"{LAYOUT_BLOCK}{CHARACTER_BLOCK}{STYLE_BLOCK}"


def generate_sheet(
    client: openai.OpenAI,
    prompt: str,
    quality: str,
    reference_image: Image.Image | None = None,
) -> tuple[Image.Image, object]:
    """gpt-image-1: edit mode when reference image provided, generate mode otherwise."""
    if reference_image is not None:
        char_buf = io.BytesIO()
        reference_image.save(char_buf, format="PNG")
        char_buf.seek(0)
        char_buf.name = "reference.png"

        image_input: list | io.BytesIO = char_buf
        if _STYLE_IMAGES:
            style_bufs = []
            for idx, simg in enumerate(_STYLE_IMAGES):
                sbuf = io.BytesIO()
                simg.save(sbuf, format="PNG")
                sbuf.seek(0)
                sbuf.name = f"style{idx + 1}.png"
                style_bufs.append(sbuf)
            image_input = [char_buf] + style_bufs

        image_count = len(image_input) if isinstance(image_input, list) else 1
        image_names = [f.name for f in image_input] if isinstance(image_input, list) else [image_input.name]
        print(f"[GENERATE] images sent to edit(): count={image_count} names={image_names}", flush=True)

        resp = client.images.edit(
            model="gpt-image-1",
            image=image_input,
            prompt=prompt,
            size="1024x1536",
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
    result_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    print(f"[GENERATE] returned image size: {result_img.size[0]}x{result_img.size[1]}", flush=True)
    return result_img, resp.usage


# ── Analysis quality check ────────────────────────────────────────────────────

_WEAK_PHRASES = [
    "not visible", "not distinctly", "not clearly",
    "obscured", "partially obscured",
    "cannot", "can't see", "hard to", "difficult to",
    "not shown", "not present", "unclear",
]


def _is_weak_analysis(analysis: str) -> tuple[bool, str]:
    """Return (is_weak, reason) based on FACE/EYES section quality."""
    secs = _parse_analysis_sections(analysis)
    reasons = []
    for section in ("EYES", "FACE"):
        content = secs.get(section, "").lower()
        if not content:
            reasons.append(f"{section}: missing")
        elif len(content) < 20:
            reasons.append(f"{section}: too short")
        else:
            for phrase in _WEAK_PHRASES:
                if phrase in content:
                    reasons.append(f"{section}: '{phrase}'")
                    break
    return bool(reasons), " | ".join(reasons)


# ── Goods simulation ───────────────────────────────────────────────────────────

def _placeholder_template(goods_type: str, canvas: tuple) -> Image.Image:
    w, h = canvas
    img = Image.new("RGBA", (w, h), (253, 244, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([8, 8, w - 8, h - 8], radius=18, outline=(192, 132, 252), width=3)
    px, py, pw, ph = GOODS_CONFIG[goods_type]["area"]
    draw.rectangle([px, py, px + pw, py + ph], outline=(209, 180, 252), width=2)
    return img


def simulate_goods(image: Image.Image | None, goods_type: str) -> Image.Image | None:
    if image is None:
        return None
    cfg = GOODS_CONFIG.get(goods_type)
    if cfg is None:
        return None
    cw, ch = cfg["canvas"]
    px, py, pw, ph = cfg["area"]
    template_path = GOODS_DIR / cfg["file"]
    if template_path.exists():
        template = Image.open(template_path).convert("RGBA").resize((cw, ch), Image.LANCZOS)
    else:
        template = _placeholder_template(goods_type, (cw, ch))
    char = image.resize((pw, ph), Image.LANCZOS).convert("RGBA")
    result = template.copy()
    result.paste(char, (px, py), char)
    return result.convert("RGB")


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

        is_weak, weak_reason = _is_weak_analysis(analysis)
        print(f"[QUALITY] weak={is_weak} reason='{weak_reason}'", flush=True)

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

        quality_warning = (
            f"⚠️  Face data insufficient: {weak_reason}\n"
            f"   → Use a clearer frontal image for better character identity.\n\n"
            if is_weak else ""
        )

        token_summary = (
            f"{quality_warning}"
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

        import time as _time
        sheet.save(pathlib.Path("/tmp/.current_goods.png"))
        img_html = f'<img src="/current-image?t={int(_time.time())}" alt="generated doodle" />'
        return (
            img_html, analysis, prompt, token_summary, style_ref_display,
            gr.update(visible=True),   # goods_link_row
            gr.update(visible=False),  # empty_state_row
            gr.update(visible=False),  # loading_row
        )

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
@import url('https://fonts.googleapis.com/css2?family=Gaegu:wght@400;700&family=Noto+Sans+KR:wght@300;400;500;700;900&family=Fredoka:wght@400;600;700&family=Nunito:wght@400;600;700;800&display=swap');

body, .gradio-container, .gradio-container * {
    font-family: 'Noto Sans KR', 'Nunito', sans-serif !important;
    box-sizing: border-box;
    color: #4A3E3D;
}
body, .gradio-container {
    background-color: #FAF7F2 !important;
    background-image: none !important;
    min-height: 100vh;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F8F5F0; }
::-webkit-scrollbar-thumb { background: #C084FC; border-radius: 4px; }

/* Image component areas */
div[data-testid="image"],
div[data-testid="image"] *,
div[data-testid="image"] > div,
div[data-testid="image"] .upload-container,
div[data-testid="image"] .image-container,
div[data-testid="image"] .wrap,
div[data-testid="image"] .empty,
div[data-testid="image"] .placeholder,
.upload-container, .upload-button {
    background: white !important;
    background-color: white !important;
}
div[data-testid="image"] {
    border: 2px dashed #8B7E7D !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
div[data-testid="image"] > div { border: none !important; }
div[data-testid="image"] .empty svg,
div[data-testid="image"] .icon-wrap svg {
    color: #8B7E7D !important;
    fill: #8B7E7D !important;
}
#dg-output-empty, #dg-loading {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 380px;
    width: 100%;
    background: #FDFBF7;
    border: 2px dashed #8B7E7D;
    border-radius: 12px;
    padding: 40px 20px;
}
@keyframes dg-bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}
#dg-loading .dg-spin {
    font-size: 40px;
    animation: dg-bounce 1.2s ease-in-out infinite;
    display: inline-block;
    margin-bottom: 14px;
}

/* Header */
#app-header { text-align: center; padding: 1.5rem 0 0.25rem; }
#app-header h1 {
    font-family: 'Gaegu', 'Fredoka', cursive !important;
    font-size: 2.4rem; font-weight: 800;
    color: #4A3E3D !important;
    -webkit-text-fill-color: #4A3E3D !important;
    margin-bottom: 0.3rem;
}
#app-header p { color: #7C6E6D !important; font-size: 0.9rem; font-weight: 600; }

/* Panels — sketch card style */
#left-panel, #right-panel {
    background: white !important;
    border: 3px solid #4A3E3D !important;
    border-radius: 20px !important;
    box-shadow: 5px 5px 0px 0px #C084FC !important;
    padding: 1.4rem !important;
    position: relative !important;
    overflow: visible !important;
    transition: all 0.2s ease !important;
}
#left-panel:hover, #right-panel:hover {
    box-shadow: 7px 7px 0px 0px #C084FC !important;
}

/* Washi tapes */
#left-panel::before {
    content: '';
    position: absolute; top: -10px; left: 50%;
    transform: translateX(-50%) rotate(-1.5deg);
    width: 100px; height: 20px;
    background-color: rgba(244, 143, 177, 0.8);
    border-left: 2px dashed rgba(255,255,255,0.4);
    border-right: 2px dashed rgba(255,255,255,0.4);
    z-index: 10; pointer-events: none;
}
#right-panel::before {
    content: '';
    position: absolute; top: -10px; left: 50%;
    transform: translateX(-50%) rotate(2deg);
    width: 100px; height: 20px;
    background-color: rgba(255, 213, 79, 0.8);
    border-left: 2px dashed rgba(255,255,255,0.4);
    border-right: 2px dashed rgba(255,255,255,0.4);
    z-index: 10; pointer-events: none;
}

/* Generate button */
#generate-btn {
    background: #E9D5FF !important;
    color: #4A3E3D !important;
    border: 2px solid #4A3E3D !important;
    font-size: 1rem !important; font-weight: 900 !important;
    border-radius: 12px !important; padding: 0.75rem !important;
    width: 100% !important;
    box-shadow: 3px 3px 0px 0px #4A3E3D !important;
    transition: all 0.2s ease !important;
}
#generate-btn:hover {
    background: #D8B4FE !important;
    transform: translate(-1px, -1px) !important;
    box-shadow: 4px 4px 0px 0px #4A3E3D !important;
    opacity: 1 !important;
}
#generate-btn:active {
    transform: translate(2px, 2px) !important;
    box-shadow: 1px 1px 0px 0px #4A3E3D !important;
}

/* Style ref button */
#style-ref-btn {
    border: 2px solid #4A3E3D !important;
    color: #4A3E3D !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    box-shadow: 2px 2px 0px #4A3E3D !important;
    background: white !important;
}

/* Tip box */
.tip-box {
    background: #F7F4EF !important;
    border: 2px solid #4A3E3D !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.88rem !important; font-weight: 700 !important;
    color: #4A3E3D !important;
    box-shadow: 2px 2px 0px #4A3E3D !important;
}
.tip-box p, .tip-box strong, .tip-box * { color: #4A3E3D !important; }

/* Headings */
.gradio-container h3, .gradio-container h2 {
    font-family: 'Gaegu', 'Fredoka', cursive !important;
    color: #4A3E3D !important; font-weight: 800 !important;
}
details summary, details summary span, .accordion-header, .label-wrap span {
    color: #4A3E3D !important; font-weight: 700 !important;
}
.gradio-container p, .gradio-container .prose p { color: #4A3E3D !important; }

/* HR */
hr { border: none !important; border-top: 2px dashed #8B7E7D !important; margin: 1rem 0 !important; }

/* Text inputs */
textarea, input[type="text"] {
    border-radius: 10px !important;
    border: 2px solid #8B7E7D !important;
    background: #FDFBF7 !important;
    font-family: 'Noto Sans KR', sans-serif !important;
    font-size: 0.9rem !important; color: #4A3E3D !important;
}

/* Accordion */
details, .accordion {
    border-radius: 16px !important;
    border: 2px solid #4A3E3D !important;
    background: white !important;
    overflow: hidden !important;
    box-shadow: 3px 3px 0px #C084FC !important;
    margin-bottom: 0.5rem !important;
}

/* Tabs */
.tabs .tab-nav { border-bottom: 2px solid #4A3E3D !important; background: #F7F4EF !important; }
.tabs .tab-nav button {
    font-weight: 700 !important;
    font-family: 'Noto Sans KR', sans-serif !important;
    color: #7C6E6D !important;
    border-right: 1px solid #D6CFC8 !important;
    font-size: 0.88rem !important;
}
.tabs .tab-nav button.selected {
    color: #C084FC !important;
    background: white !important;
    font-weight: 900 !important;
}

/* Help tooltip — CSS hover only */
.dg-help-wrap { position: relative; display: flex; justify-content: flex-end; margin-bottom: 6px; }
.dg-help-icon {
    background: #FEF08A; border: 2px solid #4A3E3D; border-radius: 50%;
    width: 26px; height: 26px; font-size: 13px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 1px 1px 0px #4A3E3D; transition: transform 0.15s;
    padding: 0; line-height: 1;
}
.dg-help-icon:hover { transform: scale(1.05); }
.dg-help-pop {
    display: none; position: absolute; bottom: calc(100% + 8px); right: 0;
    width: 240px; background: white; border: 2px solid #4A3E3D;
    border-radius: 12px; padding: 12px 14px;
    box-shadow: 3px 3px 0px 0px #C084FC; z-index: 9999;
}
.dg-help-wrap:hover .dg-help-pop { display: block; }
.dg-help-title { font-weight: 800; color: #4A3E3D !important; font-size: 0.85rem; margin-bottom: 6px; }
.dg-help-list { margin: 0; padding-left: 14px; color: #4A3E3D !important; font-size: 0.8rem; font-weight: 600; line-height: 1.6; }
.dg-help-list li { color: #4A3E3D !important; }

/* Radio groups hidden */
#mode-radio-group, #quality-radio-group {
    position: fixed !important; top: -9999px !important; left: 0 !important;
}

/* Custom image display (replaces gr.Image) */
#dg-image-display img {
    width: 100%;
    height: auto;
    max-height: 420px;
    object-fit: contain;
    border-radius: 12px;
    border: 2px dashed #8B7E7D;
    display: block;
}
/* Hide input image label */
#left-panel div[data-testid="image"] .label-wrap,
#left-panel div[data-testid="image"] .block-label {
    display: none !important;
}


/* Action buttons row inside right panel */
#dg-action-row { display: block !important; }
#dg-action-row > div { width: 100% !important; }

/* Override Gradio CSS variables that cause crimson/dark background */
:root, html {
    --block-background-fill: white !important;
    --background-fill-primary: white !important;
    --background-fill-secondary: #F8F5F0 !important;
    --color-accent: #C084FC !important;
    --primary-500: #A855F7 !important;
    --primary-600: #9333EA !important;
    --secondary-500: #A855F7 !important;
    --secondary-600: #9333EA !important;
}
/* Fix crimson/dark block on image components */
.gradio-container label.block,
.gradio-container .block,
.gradio-container [data-testid="image"],
.gradio-container [data-testid="image"] *,
.gradio-container [data-testid="image"] > div,
.gradio-container [data-testid="image"] .wrap,
.gradio-container [data-testid="image"] .empty {
    background: white !important;
    background-color: white !important;
}
"""


_HELP_HTML = """
<div class="dg-help-wrap" id="dg-help-wrap">
  <button class="dg-help-icon" aria-label="생성 팁">💡</button>
  <div class="dg-help-pop">
    <div class="dg-help-title">💡 낙서 작화 마스터 꿀팁!</div>
    <ul class="dg-help-list">
      <li>얼굴이 정면에 가깝고 선명할수록 사랑스러운 낙서가 나옵니다.</li>
      <li>어둡거나 흔들린 사진보다 밝고 선명한 실내/자연광 컷이 좋습니다.</li>
      <li>Full Character Sheet는 생성마다 결과가 달라질 수 있어요.</li>
      <li>파스텔 수채화 스타일은 부드러운 다꾸 표현에 가장 예쁘게 어울립니다.</li>
    </ul>
  </div>
</div>
"""

_MODE_BUTTONS_HTML = """
<div style="margin-top:8px;">
  <div style="margin-bottom:6px;">
    <span style="background:#FEE2E2;border:2px solid #4A3E3D;color:#4A3E3D;font-size:10px;
      font-weight:700;padding:2px 8px;border-radius:6px;display:inline-block;box-shadow:2px 2px 0px #4A3E3D;">
      2단계: 드로잉 레이아웃 콘셉트</span>
  </div>
  <div id="gen-mode-grid">
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;">
      <button type="button" onclick="setGenMode(this,0)"
        style="padding:8px 4px;border-radius:10px;border:2px solid #4A3E3D;background:#E9D5FF;
        color:#4A3E3D;font-weight:800;font-size:11px;cursor:pointer;text-align:center;
        box-shadow:3px 3px 0px #4A3E3D;transform:translate(-1px,-1px);transition:all 0.15s;">
        🌸 풀 캐릭터 시트</button>
      <button type="button" onclick="setGenMode(this,1)"
        style="padding:8px 4px;border-radius:10px;border:2px solid #4A3E3D;background:white;
        color:#4A3E3D;font-weight:800;font-size:11px;cursor:pointer;text-align:center;
        box-shadow:1px 1px 0px #4A3E3D;transition:all 0.15s;">
        🖼️ 포트레이트 낙서</button>
      <button type="button" onclick="setGenMode(this,2)"
        style="padding:8px 4px;border-radius:10px;border:2px solid #4A3E3D;background:white;
        color:#4A3E3D;font-weight:800;font-size:11px;cursor:pointer;text-align:center;
        box-shadow:1px 1px 0px #4A3E3D;transition:all 0.15s;">
        🧜‍♀️ 상반신 캐릭터</button>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:6px;">
      <button type="button" onclick="setGenMode(this,3)"
        style="padding:8px 4px;border-radius:10px;border:2px solid #4A3E3D;background:white;
        color:#4A3E3D;font-weight:800;font-size:11px;cursor:pointer;text-align:center;
        box-shadow:1px 1px 0px #4A3E3D;transition:all 0.15s;">
        🍀 치비 스티커</button>
      <button type="button" onclick="setGenMode(this,4)"
        style="padding:8px 4px;border-radius:10px;border:2px solid #4A3E3D;background:white;
        color:#4A3E3D;font-weight:800;font-size:11px;cursor:pointer;text-align:center;
        box-shadow:1px 1px 0px #4A3E3D;transition:all 0.15s;">
        ✨ 심플 클린 포트레이트</button>
    </div>
  </div>
</div>
"""

_QUALITY_BUTTONS_HTML = """
<div style="display:flex;align-items:center;justify-content:space-between;margin-top:10px;">
  <span style="background:#FEE2E2;border:2px solid #4A3E3D;color:#4A3E3D;font-size:10px;
    font-weight:700;padding:2px 8px;border-radius:6px;display:inline-block;box-shadow:2px 2px 0px #4A3E3D;">
    3단계: 드로잉 퀄리티</span>
  <div style="display:flex;align-items:center;gap:6px;">
    <div style="display:flex;gap:3px;background:#FAF9F5;padding:4px;border-radius:8px;
      border:2px solid #4A3E3D;" id="quality-tabs">
      <button type="button" onclick="setQuality(this,0)"
        style="padding:4px 14px;border-radius:5px;border:none;background:transparent;
        font-size:10px;font-weight:900;color:#7C6E6D;cursor:pointer;transition:all 0.15s;">
        Low</button>
      <button type="button" onclick="setQuality(this,1)"
        style="padding:4px 14px;border-radius:5px;border:none;background:transparent;
        font-size:10px;font-weight:900;color:#7C6E6D;cursor:pointer;transition:all 0.15s;">
        Medium</button>
      <button type="button" onclick="setQuality(this,2)"
        style="padding:4px 14px;border-radius:5px;border:1px solid #4A3E3D;background:white;
        font-size:10px;font-weight:900;color:#4A3E3D;cursor:pointer;
        box-shadow:1px 1px 0px #4A3E3D;transition:all 0.15s;">
        High</button>
    </div>
    <div class="dg-help-wrap" style="position:relative;margin:0;">
      <button class="dg-help-icon" aria-label="생성 팁">💡</button>
      <div class="dg-help-pop">
        <div class="dg-help-title">💡 낙서 작화 마스터 꿀팁!</div>
        <ul class="dg-help-list">
          <li>얼굴이 정면에 가깝고 선명할수록 사랑스러운 낙서가 나옵니다.</li>
          <li>어둡거나 흔들린 사진보다 밝고 선명한 실내/자연광 컷이 좋습니다.</li>
          <li>Full Character Sheet는 생성마다 결과가 달라질 수 있어요.</li>
          <li>파스텔 수채화 스타일은 부드러운 다꾸 표현에 가장 예쁘게 어울립니다.</li>
        </ul>
      </div>
    </div>
  </div>
</div>
"""

_ACTION_BUTTONS_HTML = """
<div style="display:flex;flex-direction:column;gap:8px;padding-top:8px;">
  <button onclick="dgOpenModal()"
    style="width:100%;background:#FEF08A;color:#4A3E3D;border:2px solid #4A3E3D;
    border-radius:12px;font-weight:900;font-size:0.9rem;padding:10px 16px;
    box-shadow:3px 3px 0px #4A3E3D;cursor:pointer;font-family:inherit;transition:all 0.15s;"
    onmouseover="this.style.transform='translate(-1px,-1px)';this.style.boxShadow='4px 4px 0px #4A3E3D'"
    onmouseout="this.style.transform='none';this.style.boxShadow='3px 3px 0px #4A3E3D'">
    🎁 나만의 다꾸 굿즈 제작하기 ✨
  </button>
  <div style="display:flex;gap:8px;">
    <button onclick="dgSaveImage()"
      style="flex:1;background:#2D2727;color:white;border:2px solid #4A3E3D;
      border-radius:10px;font-weight:700;font-size:0.85rem;padding:9px 16px;
      box-shadow:2px 2px 0px #4A3E3D;cursor:pointer;font-family:inherit;transition:all 0.15s;"
      onmouseover="this.style.transform='translate(-1px,-1px)'"
      onmouseout="this.style.transform='none'">
      ⬇ 스케치북에 저장
    </button>
    <button onclick="dgResetImage()"
      style="background:white;color:#4A3E3D;border:2px solid #4A3E3D;border-radius:50%;
      font-weight:900;font-size:1rem;width:42px;height:42px;min-width:42px;
      box-shadow:2px 2px 0px #4A3E3D;cursor:pointer;font-family:inherit;transition:all 0.15s;
      display:flex;align-items:center;justify-content:center;"
      onmouseover="this.style.transform='translate(-1px,-1px)'"
      onmouseout="this.style.transform='none'">
      ↺
    </button>
  </div>
</div>
"""

_GOODS_MODAL_HTML = """
<div id="dg-modal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,0.62);
    backdrop-filter:blur(4px);z-index:99999;align-items:center;justify-content:center;padding:8px;">
  <div style="background:#FAF7F2;border:3px solid #4A3E3D;border-radius:20px;
    box-shadow:5px 5px 0px #C084FC;width:100%;max-width:1240px;max-height:94vh;
    overflow-y:auto;position:relative;padding:1rem 1.2rem 1.2rem;">

    <button onclick="dgCloseModal()" style="position:absolute;top:10px;right:12px;
      background:none;border:none;font-size:1.3rem;cursor:pointer;color:#4A3E3D;font-weight:900;">✕</button>

    <div style="text-align:center;margin-bottom:10px;">
      <h2 style="font-family:'Gaegu','Fredoka',cursive;font-size:1.3rem;font-weight:900;color:#4A3E3D;margin:0 0 2px;">
        🪄 나만의 드로잉 굿즈 디자인스튜디오</h2>
      <p style="font-size:0.72rem;color:#7C6E6D;font-weight:600;margin:0;">
        생성된 낙서로 나만의 다꾸 굿즈를 직접 꾸며보세요 ♡</p>
    </div>

    <!-- Top bar: goods type + bg removal -->
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px;
      background:white;border:2px solid #4A3E3D;border-radius:12px;padding:8px 12px;margin-bottom:10px;">
      <div style="display:flex;gap:5px;flex-wrap:wrap;">
        <button id="dg-bt-phone" onclick="dgSetType('phone')"
          style="padding:5px 9px;border-radius:7px;border:2px solid #4A3E3D;background:#E9D5FF;
          font-size:12px;font-weight:900;color:#4A3E3D;cursor:pointer;box-shadow:2px 2px 0px #4A3E3D;">📱 폰케이스</button>
        <button id="dg-bt-griptok" onclick="dgSetType('griptok')"
          style="padding:5px 9px;border-radius:7px;border:2px solid #8B7E7D;background:white;
          font-size:12px;font-weight:900;color:#4A3E3D;cursor:pointer;">🔘 그립톡</button>
        <button id="dg-bt-mug" onclick="dgSetType('mug')"
          style="padding:5px 9px;border-radius:7px;border:2px solid #8B7E7D;background:white;
          font-size:12px;font-weight:900;color:#4A3E3D;cursor:pointer;">🥛 머그컵</button>
        <button id="dg-bt-keyring" onclick="dgSetType('keyring')"
          style="padding:5px 9px;border-radius:7px;border:2px solid #8B7E7D;background:white;
          font-size:12px;font-weight:900;color:#4A3E3D;cursor:pointer;">🔑 키링</button>
        <button id="dg-bt-tshirt" onclick="dgSetType('tshirt')"
          style="padding:5px 9px;border-radius:7px;border:2px solid #8B7E7D;background:white;
          font-size:12px;font-weight:900;color:#4A3E3D;cursor:pointer;">👕 티셔츠</button>
      </div>
      <label style="display:flex;align-items:center;gap:7px;cursor:pointer;user-select:none;">
        <span style="font-size:12px;font-weight:700;color:#4A3E3D;">✨ 누끼따기</span>
        <div style="position:relative;width:40px;height:20px;flex-shrink:0;">
          <input type="checkbox" id="dg-bg-toggle" onchange="dgToggleBg(this.checked)"
            style="opacity:0;width:0;height:0;position:absolute;">
          <div id="dg-bg-track"
            style="position:absolute;inset:0;background:#D1D5DB;border-radius:20px;border:2px solid #4A3E3D;cursor:pointer;transition:background 0.2s;">
            <div id="dg-bg-knob" style="position:absolute;top:1px;left:1px;width:14px;height:14px;
              background:white;border-radius:50%;transition:transform 0.2s;pointer-events:none;"></div>
          </div>
        </div>
      </label>
    </div>

    <!-- 3-panel body -->
    <div style="display:grid;grid-template-columns:1fr 1.15fr 200px;gap:10px;align-items:start;">

      <!-- Panel 1: Sticker Cropper -->
      <div style="background:white;border:2px solid #4A3E3D;border-radius:12px;padding:10px;">
        <div style="font-size:13px;font-weight:900;color:#4A3E3D;margin-bottom:8px;">✂️ 스티커 크롭</div>
        <div style="position:relative;width:100%;aspect-ratio:1/1;background:#F8F5F0;
          border-radius:8px;border:1px dashed #8B7E7D;overflow:hidden;">
          <canvas id="dg-crop-canvas" style="position:absolute;inset:0;width:100%;height:100%;display:block;"></canvas>
          <canvas id="dg-crop-sel" style="position:absolute;inset:0;width:100%;height:100%;cursor:crosshair;"></canvas>
        </div>
        <div style="display:flex;gap:4px;margin-top:6px;margin-bottom:6px;">
          <button id="dg-crop-rect" onclick="dgSetCropMode('rect')"
            style="flex:1;padding:5px;border-radius:6px;border:2px solid #4A3E3D;background:#E9D5FF;
            font-size:11px;font-weight:800;color:#4A3E3D;cursor:pointer;">⬜ 사각</button>
          <button id="dg-crop-lasso" onclick="dgSetCropMode('lasso')"
            style="flex:1;padding:5px;border-radius:6px;border:1px solid #8B7E7D;background:white;
            font-size:11px;font-weight:800;color:#4A3E3D;cursor:pointer;">🌀 올가미</button>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;">
          <button onclick="dgPresetCrop(0,0,1,0.42)" style="padding:5px;border-radius:6px;
            border:1px solid #4A3E3D;background:#FEF3C7;font-size:11px;font-weight:700;cursor:pointer;">🔝 상단</button>
          <button onclick="dgPresetCrop(0,0.3,1,0.42)" style="padding:5px;border-radius:6px;
            border:1px solid #4A3E3D;background:#FEF3C7;font-size:11px;font-weight:700;cursor:pointer;">⬛ 중단</button>
          <button onclick="dgPresetCrop(0,0.58,1,0.42)" style="padding:5px;border-radius:6px;
            border:1px solid #4A3E3D;background:#FEF3C7;font-size:11px;font-weight:700;cursor:pointer;">🔽 하단</button>
          <button onclick="dgPresetCrop(0,0,1,1)" style="padding:5px;border-radius:6px;
            border:1px solid #4A3E3D;background:#E9D5FF;font-size:11px;font-weight:700;cursor:pointer;">📋 전체</button>
        </div>
        <button onclick="dgAddToPocket()" style="width:100%;margin-top:6px;padding:8px;background:#C084FC;
          color:white;border:2px solid #4A3E3D;border-radius:8px;font-size:12px;font-weight:900;cursor:pointer;
          box-shadow:2px 2px 0px #4A3E3D;">+ 보관함에 추가 ♡</button>
      </div>

      <!-- Panel 2: Goods Preview Canvas -->
      <div style="background:white;border:2px solid #4A3E3D;border-radius:12px;padding:10px;">
        <div style="font-size:13px;font-weight:900;color:#4A3E3D;margin-bottom:8px;">🎨 굿즈 프리뷰</div>
        <div style="display:flex;justify-content:center;align-items:center;
          background:#F8F5F0;border-radius:8px;border:1px dashed #8B7E7D;
          padding:12px;position:relative;min-height:270px;">
          <canvas id="dg-goods-canvas" style="max-width:100%;max-height:310px;cursor:crosshair;display:block;"></canvas>
          <div id="dg-sticker-ctrl" style="display:none;position:absolute;bottom:4px;left:4px;right:4px;
            background:white;border:1.5px solid #C084FC;border-radius:8px;padding:6px 8px;">
            <div style="display:flex;align-items:center;gap:6px;">
              <span style="font-size:11px;font-weight:700;color:#4A3E3D;white-space:nowrap;">크기</span>
              <input type="range" id="dg-st-scale" min="10" max="80" value="30"
                oninput="dgScaleSticker(this.value)"
                style="flex:1;accent-color:#C084FC;cursor:pointer;height:4px;">
              <button onclick="dgDeleteSticker()" style="background:#4A3E3D;color:white;border:none;
                border-radius:4px;padding:3px 8px;font-size:11px;cursor:pointer;white-space:nowrap;">✕ 삭제</button>
            </div>
          </div>
        </div>
        <div style="margin-top:8px;">
          <div style="font-size:11px;font-weight:700;color:#7C6E6D;margin-bottom:5px;">제품 컬러</div>
          <div style="display:flex;gap:5px;flex-wrap:wrap;align-items:center;">
            <button onclick="dgColor('#FDE2E4')" style="width:22px;height:22px;border-radius:50%;background:#FDE2E4;border:2.5px solid #4A3E3D;cursor:pointer;" title="핑크"></button>
            <button onclick="dgColor('#D8F3DC')" style="width:22px;height:22px;border-radius:50%;background:#D8F3DC;border:2px solid #8B7E7D;cursor:pointer;" title="민트"></button>
            <button onclick="dgColor('#E0F2FE')" style="width:22px;height:22px;border-radius:50%;background:#E0F2FE;border:2px solid #8B7E7D;cursor:pointer;" title="스카이"></button>
            <button onclick="dgColor('#FEF08A')" style="width:22px;height:22px;border-radius:50%;background:#FEF08A;border:2px solid #8B7E7D;cursor:pointer;" title="레몬"></button>
            <button onclick="dgColor('#E2D5F0')" style="width:22px;height:22px;border-radius:50%;background:#E2D5F0;border:2px solid #8B7E7D;cursor:pointer;" title="라벤더"></button>
            <button onclick="dgColor('#FFFFFF')" style="width:22px;height:22px;border-radius:50%;background:#FFFFFF;border:2px solid #8B7E7D;cursor:pointer;" title="화이트"></button>
            <button onclick="dgColor('#1E1B1B')" style="width:22px;height:22px;border-radius:50%;background:#1E1B1B;border:2px solid #8B7E7D;cursor:pointer;" title="블랙"></button>
          </div>
        </div>
        <div style="display:flex;align-items:center;justify-content:space-between;margin-top:8px;padding:0 2px;">
          <span style="font-size:11px;font-weight:700;color:#4A3E3D;">📐 인쇄 가이드</span>
          <label style="position:relative;display:inline-block;width:38px;height:20px;cursor:pointer;">
            <input type="checkbox" id="dg-guide-toggle" checked
              style="opacity:0;width:0;height:0;position:absolute;" onchange="dgToggleGuide(this.checked)">
            <div id="dg-guide-track"
              style="position:absolute;inset:0;background:#86EFAC;border-radius:20px;border:1.5px solid #4A3E3D;transition:background 0.2s;">
              <div id="dg-guide-knob" style="position:absolute;top:1px;left:19px;width:14px;height:14px;
                background:white;border-radius:50%;transition:left 0.2s;pointer-events:none;"></div>
            </div>
          </label>
        </div>
        <button onclick="dgExportGoods()" style="width:100%;margin-top:8px;padding:8px;
          background:#2D2727;color:white;border:2px solid #4A3E3D;border-radius:8px;
          font-size:12px;font-weight:900;cursor:pointer;box-shadow:2px 2px 0px #4A3E3D;">⬇ 굿즈 디자인 저장</button>
      </div>

      <!-- Panel 3: Sticker Pocket -->
      <div style="background:white;border:2px solid #4A3E3D;border-radius:12px;padding:10px;">
        <div style="font-size:13px;font-weight:900;color:#4A3E3D;margin-bottom:8px;">🗂 스티커 보관함</div>
        <div id="dg-pocket" style="display:grid;grid-template-columns:1fr 1fr;gap:4px;
          min-height:90px;max-height:210px;overflow-y:auto;">
          <div id="dg-pocket-empty" style="color:#8B7E7D;font-size:11px;font-weight:600;
            text-align:center;grid-column:1/-1;padding:20px 4px;line-height:1.7;">
            크롭 후 추가하면<br>여기 모여요 ♡</div>
        </div>
        <div style="background:#FEF3C7;border:1px solid #4A3E3D;border-radius:6px;
          padding:6px 8px;margin-top:6px;font-size:10.5px;color:#4A3E3D;font-weight:600;line-height:1.6;">
          💡 스티커를 굿즈로<br>드래그해서 배치!<br>클릭 후 이동·크기 조절
        </div>
        <button onclick="dgClearPocket()" style="width:100%;margin-top:6px;padding:6px;
          background:white;color:#7C6E6D;border:1px solid #8B7E7D;border-radius:6px;
          font-size:11px;font-weight:700;cursor:pointer;">🗑 보관함 비우기</button>
      </div>
    </div>

    <hr style="border:none;border-top:2px dashed #8B7E7D;margin:10px 0;">
    <div style="display:flex;gap:8px;">
      <button onclick="dgCloseModal()" style="flex:1;padding:0.65rem;background:#C084FC;color:white;
        border:2px solid #4A3E3D;font-weight:900;font-size:0.85rem;border-radius:12px;
        box-shadow:3px 3px 0px #4A3E3D;cursor:pointer;">💌 완성 ♡</button>
      <button onclick="dgCloseModal()" style="padding:0.65rem 1rem;background:white;color:#4A3E3D;
        border:2px solid #4A3E3D;font-weight:700;font-size:0.85rem;border-radius:12px;
        box-shadow:2px 2px 0px #4A3E3D;cursor:pointer;">돌아가기</button>
    </div>
  </div>
</div>
"""

# ── Global JS (runs after Gradio mounts; defines all onclick handler functions) ─

_CUSTOM_JS = """
function dgClickRadio(groupId, idx) {
  var g = document.getElementById(groupId);
  if (!g) return;
  var labels = g.querySelectorAll('label');
  if (labels[idx]) { labels[idx].click(); return; }
  var inputs = g.querySelectorAll('input[type="radio"]');
  if (inputs[idx]) {
    inputs[idx].checked = true;
    inputs[idx].click();
    inputs[idx].dispatchEvent(new Event('change', {bubbles:true}));
    inputs[idx].dispatchEvent(new Event('input', {bubbles:true}));
  }
}

function setGenMode(btn, idx) {
  document.querySelectorAll('#gen-mode-grid button').forEach(function(b) {
    b.style.background = 'white';
    b.style.border = '2px solid #4A3E3D';
    b.style.color = '#4A3E3D';
    b.style.boxShadow = '1px 1px 0px #4A3E3D';
    b.style.transform = 'none';
  });
  btn.style.background = '#E9D5FF';
  btn.style.border = '2px solid #4A3E3D';
  btn.style.color = '#4A3E3D';
  btn.style.boxShadow = '3px 3px 0px #4A3E3D';
  btn.style.transform = 'translate(-1px,-1px)';
  dgClickRadio('mode-radio-group', idx);
}

function setQuality(btn, idx) {
  document.querySelectorAll('#quality-tabs button').forEach(function(b) {
    b.style.background = 'transparent';
    b.style.border = 'none';
    b.style.boxShadow = 'none';
    b.style.color = '#7C6E6D';
  });
  btn.style.background = 'white';
  btn.style.border = '1px solid #4A3E3D';
  btn.style.boxShadow = '1px 1px 0px #4A3E3D';
  btn.style.color = '#4A3E3D';
  dgClickRadio('quality-radio-group', idx);
}

function setStyleCard(el, style) {
  document.querySelectorAll('.dg-style-card').forEach(function(c) {
    c.style.border = '2px solid #8B7E7D';
    c.style.boxShadow = '1px 1px 0px #8B7E7D';
    c.style.background = 'white';
  });
  el.style.border = '2px solid #4A3E3D';
  el.style.boxShadow = '2px 2px 0px #4A3E3D';
  el.style.background = '#FCFAF6';
}

// ── Goods Studio ──────────────────────────────────────────────────────────────
var _dgSt = {
  type:'phone', color:'#FDE2E4', removeBg:false,
  stickers:[], pocket:[], selId:null,
  cropMode:'rect',
  cropRect:null, cropDrag:false, cropStart:null,
  lassoPts:[], lassoDrag:false, lassoAnimId:null, lassoOff:0,
  pocketDragIdx:null,
  goodsDrag:false, goodsDragId:null, goodsDragOx:0, goodsDragOy:0,
  nextId:0, imgTs:0, mainImg:null,
  showGuide:true
};

var _dgCfg = {
  phone:   {cw:190, ch:340, ax:22, ay:60, aw:146, ah:220},
  griptok: {cw:220, ch:220, ax:35, ay:35, aw:150, ah:150},
  mug:     {cw:250, ch:260, ax:22, ay:55, aw:168, ah:170},
  keyring: {cw:190, ch:250, ax:24, ay:56, aw:142, ah:168},
  tshirt:  {cw:260, ch:260, ax:55, ay:76, aw:150, ah:148}
};

function _dgPos(canvas, e) {
  var r = canvas.getBoundingClientRect();
  return {x:(e.clientX-r.left)/r.width*canvas.width, y:(e.clientY-r.top)/r.height*canvas.height};
}

function dgOpenModal() {
  var m = document.getElementById('dg-modal');
  if (!m) return;
  m.style.display = 'flex';
  _dgSt.stickers = []; _dgSt.selId = null; _dgSt.mainImg = null;
  _dgSt.imgTs = Date.now();
  var ctrl = document.getElementById('dg-sticker-ctrl');
  if (ctrl) ctrl.style.display = 'none';
  setTimeout(function() {
    _dgSetupGoods();  // attach canvas events once, before any render
    var img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = function() { _dgSt.mainImg = img; _dgInitCrop(); _dgRender(); };
    img.onerror = function() { _dgInitCrop(); _dgRender(); };
    img.src = '/current-image?t=' + _dgSt.imgTs;
  }, 150);
}

function dgCloseModal() {
  var m = document.getElementById('dg-modal');
  if (m) m.style.display = 'none';
}

function dgSetType(t) {
  _dgSt.type = t;
  ['phone','griptok','mug','keyring','tshirt'].forEach(function(x) {
    var b = document.getElementById('dg-bt-' + x);
    if (!b) return;
    b.style.border = x===t ? '2px solid #4A3E3D' : '2px solid #8B7E7D';
    b.style.background = x===t ? '#E9D5FF' : 'white';
    b.style.boxShadow = x===t ? '2px 2px 0px #4A3E3D' : 'none';
  });
  _dgRender();
}

function dgColor(c) { _dgSt.color = c; _dgRender(); }

function dgToggleBg(on) {
  _dgSt.removeBg = on;
  var tr = document.getElementById('dg-bg-track');
  var kn = document.getElementById('dg-bg-knob');
  if (tr) tr.style.background = on ? '#C084FC' : '#D1D5DB';
  if (kn) kn.style.transform = on ? 'translateX(20px)' : 'translateX(0)';
  _dgRender();
}

// ── Sticker Cropper ────────────────────────────────────────────────────────────
function _dgInitCrop() {
  var cc = document.getElementById('dg-crop-canvas');
  var cs = document.getElementById('dg-crop-sel');
  if (!cc || !cs) return;
  var sz = cc.parentElement.offsetWidth || 220;
  cc.width = sz; cc.height = sz; cs.width = sz; cs.height = sz;
  if (_dgSt.mainImg) {
    var img = _dgSt.mainImg, ctx = cc.getContext('2d');
    var sc = Math.min(sz/img.width, sz/img.height);
    ctx.clearRect(0,0,sz,sz);
    ctx.drawImage(img, (sz-img.width*sc)/2, (sz-img.height*sc)/2, img.width*sc, img.height*sc);
  }
  if (cs._dgBound) return;
  cs._dgBound = true;
  function cp(e) {
    var r=cs.getBoundingClientRect(), t=e.touches?e.touches[0]:e;
    return {x:(t.clientX-r.left)/r.width*cs.width, y:(t.clientY-r.top)/r.height*cs.height};
  }
  function onDown(e) {
    var p=cp(e);
    if (_dgSt.cropMode==='lasso') {
      _dgSt.lassoDrag=true; _dgSt.lassoPts=[p]; _dgSt.cropRect=null;
      _dgStartLassoAnim();
    } else {
      _dgSt.cropDrag=true; _dgSt.cropStart=p; _dgSt.cropRect=null;
      _dgSt.lassoPts=[];
    }
  }
  function onMove(e) {
    var p=cp(e);
    if (_dgSt.cropMode==='lasso' && _dgSt.lassoDrag) {
      _dgSt.lassoPts.push(p);
    } else if (_dgSt.cropMode==='rect' && _dgSt.cropDrag) {
      var s=_dgSt.cropStart;
      _dgSt.cropRect={x:Math.min(s.x,p.x),y:Math.min(s.y,p.y),w:Math.abs(p.x-s.x),h:Math.abs(p.y-s.y)};
      _dgDrawSel();
    }
  }
  function onUp() { _dgSt.cropDrag=false; _dgSt.lassoDrag=false; }
  cs.onmousedown=onDown; cs.onmousemove=onMove; cs.onmouseup=onUp; cs.onmouseleave=onUp;
  cs.addEventListener('touchstart',function(e){e.preventDefault();onDown(e);},{passive:false});
  cs.addEventListener('touchmove', function(e){e.preventDefault();onMove(e);},{passive:false});
  cs.addEventListener('touchend',  function(e){e.preventDefault();onUp();},{passive:false});
}

function dgSetCropMode(mode) {
  _dgSt.cropMode = mode;
  var rb = document.getElementById('dg-crop-rect'), lb = document.getElementById('dg-crop-lasso');
  if (rb) { rb.style.background=mode==='rect'?'#E9D5FF':'white'; rb.style.border=mode==='rect'?'2px solid #4A3E3D':'1px solid #8B7E7D'; }
  if (lb) { lb.style.background=mode==='lasso'?'#E9D5FF':'white'; lb.style.border=mode==='lasso'?'2px solid #4A3E3D':'1px solid #8B7E7D'; }
  _dgSt.cropRect=null; _dgSt.lassoPts=[];
  var cs=document.getElementById('dg-crop-sel');
  if (cs) cs.getContext('2d').clearRect(0,0,cs.width,cs.height);
  if (mode==='lasso') _dgStartLassoAnim();
  else if (_dgSt.lassoAnimId) { cancelAnimationFrame(_dgSt.lassoAnimId); _dgSt.lassoAnimId=null; }
}

function _dgStartLassoAnim() {
  if (_dgSt.lassoAnimId) cancelAnimationFrame(_dgSt.lassoAnimId);
  function step() {
    _dgSt.lassoOff = (_dgSt.lassoOff+0.5)%9;
    _dgDrawLasso();
    _dgSt.lassoAnimId = requestAnimationFrame(step);
  }
  _dgSt.lassoAnimId = requestAnimationFrame(step);
}

function _dgDrawLasso() {
  var cs=document.getElementById('dg-crop-sel');
  if (!cs) return;
  var ctx=cs.getContext('2d');
  ctx.clearRect(0,0,cs.width,cs.height);
  if (_dgSt.lassoPts.length<2) return;
  ctx.beginPath();
  ctx.moveTo(_dgSt.lassoPts[0].x, _dgSt.lassoPts[0].y);
  _dgSt.lassoPts.forEach(function(p){ctx.lineTo(p.x,p.y);});
  if (!_dgSt.lassoDrag) ctx.closePath();
  ctx.fillStyle='rgba(236,72,153,0.15)'; ctx.fill();
  ctx.strokeStyle='#EC4899'; ctx.lineWidth=2;
  ctx.setLineDash([6,3]); ctx.lineDashOffset=-_dgSt.lassoOff;
  ctx.stroke(); ctx.setLineDash([]);
}

function _dgDrawSel() {
  var cs=document.getElementById('dg-crop-sel');
  if (!cs) return;
  var ctx=cs.getContext('2d'), r=_dgSt.cropRect;
  ctx.clearRect(0,0,cs.width,cs.height);
  if (!r) return;
  ctx.fillStyle='rgba(192,132,252,0.18)'; ctx.fillRect(r.x,r.y,r.w,r.h);
  ctx.strokeStyle='#C084FC'; ctx.lineWidth=2; ctx.setLineDash([4,3]);
  ctx.strokeRect(r.x,r.y,r.w,r.h); ctx.setLineDash([]);
}

function dgPresetCrop(nx,ny,nw,nh) {
  var cs=document.getElementById('dg-crop-sel');
  if (!cs) return;
  _dgSt.cropRect={x:nx*cs.width,y:ny*cs.height,w:nw*cs.width,h:nh*cs.height};
  _dgDrawSel();
}

function dgAddToPocket() {
  var cc=document.getElementById('dg-crop-canvas');
  if (!cc) return;
  var tmp, pw, ph;
  if (_dgSt.cropMode==='lasso' && _dgSt.lassoPts.length>5) {
    var pts=_dgSt.lassoPts;
    var xs=pts.map(function(p){return p.x;}), ys=pts.map(function(p){return p.y;});
    var minX=Math.max(0,Math.min.apply(null,xs)), minY=Math.max(0,Math.min.apply(null,ys));
    var maxX=Math.min(cc.width,Math.max.apply(null,xs)), maxY=Math.min(cc.height,Math.max.apply(null,ys));
    pw=Math.round(maxX-minX); ph=Math.round(maxY-minY);
    if (pw<4||ph<4) return;
    tmp=document.createElement('canvas'); tmp.width=pw; tmp.height=ph;
    var ctx2=tmp.getContext('2d');
    ctx2.beginPath();
    ctx2.moveTo(pts[0].x-minX, pts[0].y-minY);
    pts.forEach(function(p){ctx2.lineTo(p.x-minX, p.y-minY);});
    ctx2.closePath(); ctx2.clip();
    ctx2.drawImage(cc, minX, minY, pw, ph, 0, 0, pw, ph);
  } else {
    var r=_dgSt.cropRect||{x:0,y:0,w:cc.width,h:cc.height};
    pw=Math.max(4,Math.round(r.w)); ph=Math.max(4,Math.round(r.h));
    if (pw<4||ph<4) { r={x:0,y:0,w:cc.width,h:cc.height}; pw=cc.width; ph=cc.height; }
    tmp=document.createElement('canvas'); tmp.width=pw; tmp.height=ph;
    tmp.getContext('2d').drawImage(cc,r.x,r.y,r.w,r.h,0,0,pw,ph);
  }
  _dgSt.pocket.push(tmp.toDataURL('image/png'));
  _dgSt.cropRect=null; _dgSt.lassoPts=[]; _dgDrawSel();
  _dgUpdatePocket();
}

function _dgUpdatePocket() {
  var el=document.getElementById('dg-pocket');
  var emp=document.getElementById('dg-pocket-empty');
  if (!el) return;
  if (emp) emp.style.display=_dgSt.pocket.length?'none':'block';
  Array.from(el.children).forEach(function(c){ if(c.id!=='dg-pocket-empty') el.removeChild(c); });
  _dgSt.pocket.forEach(function(du,idx) {
    var div=document.createElement('div');
    div.style.cssText='position:relative;border:1.5px solid #C084FC;border-radius:6px;overflow:hidden;aspect-ratio:1/1;background:white;cursor:grab;';
    div.draggable=true;
    div.ondragstart=function(e){ _dgSt.pocketDragIdx=idx; e.dataTransfer.effectAllowed='copy'; };
    var img=document.createElement('img');
    img.src=du; img.style.cssText='width:100%;height:100%;object-fit:contain;pointer-events:none;';
    var del=document.createElement('button');
    del.textContent='×';
    del.style.cssText='position:absolute;top:1px;right:1px;background:rgba(74,62,61,0.85);color:white;border:none;border-radius:3px;width:15px;height:15px;font-size:9px;cursor:pointer;padding:0;line-height:1;';
    del.onclick=function(e){ e.stopPropagation(); _dgSt.pocket.splice(idx,1); _dgUpdatePocket(); };
    var sav=document.createElement('button');
    sav.textContent='↓';
    sav.style.cssText='position:absolute;bottom:1px;right:1px;background:rgba(192,132,252,0.9);color:white;border:none;border-radius:3px;width:15px;height:15px;font-size:9px;cursor:pointer;padding:0;line-height:1;';
    sav.onclick=function(e){
      e.stopPropagation();
      var a=document.createElement('a'); a.href=du; a.download='sticker_'+idx+'.png'; a.click();
    };
    div.appendChild(img); div.appendChild(del); div.appendChild(sav);
    el.appendChild(div);
  });
}

function dgClearPocket() { _dgSt.pocket=[]; _dgUpdatePocket(); }

// ── Goods Canvas Rendering ─────────────────────────────────────────────────────
function _dgRR(ctx,x,y,w,h,r) {
  ctx.beginPath();
  ctx.moveTo(x+r,y); ctx.lineTo(x+w-r,y); ctx.quadraticCurveTo(x+w,y,x+w,y+r);
  ctx.lineTo(x+w,y+h-r); ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);
  ctx.lineTo(x+r,y+h); ctx.quadraticCurveTo(x,y+h,x,y+h-r);
  ctx.lineTo(x,y+r); ctx.quadraticCurveTo(x,y,x+r,y); ctx.closePath();
}

function _dgShape(ctx,t,col) {
  var c=_dgCfg[t], w=c.cw, h=c.ch; ctx.save();
  if (t==='phone') {
    _dgRR(ctx,6,6,w-12,h-12,22); ctx.fillStyle='#1E1B1B'; ctx.fill();
    _dgRR(ctx,11,11,w-22,h-22,18); ctx.fillStyle=col; ctx.fill();
    ctx.fillStyle='rgba(0,0,0,0.06)'; ctx.fillRect(11,11,w-22,16);
    _dgRR(ctx,16,16,26,26,5); ctx.fillStyle='#1E1B1B'; ctx.fill();
  } else if (t==='griptok') {
    var R=Math.min(w,h)/2-6;
    ctx.beginPath(); ctx.arc(w/2,h/2,R,0,Math.PI*2); ctx.fillStyle='#4A3E3D'; ctx.fill();
    ctx.beginPath(); ctx.arc(w/2,h/2,R-4,0,Math.PI*2); ctx.fillStyle=col; ctx.fill();
  } else if (t==='mug') {
    _dgRR(ctx,14,26,w-56,h-44,9); ctx.fillStyle=col; ctx.fill();
    ctx.strokeStyle='#4A3E3D'; ctx.lineWidth=2.5; ctx.stroke();
    ctx.beginPath(); ctx.arc(w-30,h/2+4,24,-1.2,1.2); ctx.strokeStyle='#4A3E3D'; ctx.lineWidth=7; ctx.stroke();
    _dgRR(ctx,14,26,w-56,14,4); ctx.fillStyle='#4A3E3D'; ctx.fill();
  } else if (t==='keyring') {
    _dgRR(ctx,14,48,w-28,h-62,12); ctx.fillStyle='rgba(255,255,255,0.9)'; ctx.fill();
    ctx.strokeStyle='#4A3E3D'; ctx.lineWidth=2.5; ctx.stroke();
    ctx.beginPath(); ctx.arc(w/2,25,13,0,Math.PI*2); ctx.strokeStyle='#94A3B8'; ctx.lineWidth=2.5; ctx.stroke();
    ctx.beginPath(); ctx.moveTo(w/2,12); ctx.lineTo(w/2,48); ctx.strokeStyle='#94A3B8'; ctx.lineWidth=2.5; ctx.stroke();
  } else if (t==='tshirt') {
    ctx.beginPath();
    ctx.moveTo(62,16); ctx.lineTo(18,16); ctx.lineTo(2,58); ctx.lineTo(42,72);
    ctx.lineTo(42,h-14); ctx.lineTo(w-42,h-14); ctx.lineTo(w-42,72);
    ctx.lineTo(w-2,58); ctx.lineTo(w-18,16); ctx.lineTo(w-62,16);
    ctx.quadraticCurveTo(w/2,48,62,16);
    ctx.fillStyle=col; ctx.fill(); ctx.strokeStyle='#4A3E3D'; ctx.lineWidth=2.5; ctx.stroke();
  }
  ctx.restore();
}

function _dgBgRemove(src) {
  var tmp=document.createElement('canvas');
  tmp.width=src.width; tmp.height=src.height;
  var ctx=tmp.getContext('2d'); ctx.drawImage(src,0,0);
  var id=ctx.getImageData(0,0,tmp.width,tmp.height), d=id.data;
  for (var i=0;i<d.length;i+=4) { if(d[i]>228&&d[i+1]>228&&d[i+2]>228) d[i+3]=0; }
  ctx.putImageData(id,0,0); return tmp;
}

function _dgDrawGuides(ctx, cfg) {
  var ax=cfg.ax, ay=cfg.ay, aw=cfg.aw, ah=cfg.ah;
  ctx.save();
  ctx.strokeStyle='rgba(239,68,68,0.75)'; ctx.lineWidth=1;
  ctx.setLineDash([4,2]); ctx.strokeRect(ax-4,ay-4,aw+8,ah+8); ctx.setLineDash([]);
  ctx.strokeStyle='rgba(59,130,246,0.75)';
  ctx.setLineDash([4,2]); ctx.strokeRect(ax+6,ay+6,aw-12,ah-12); ctx.setLineDash([]);
  ctx.font='7px sans-serif';
  ctx.fillStyle='rgba(239,68,68,0.85)'; ctx.fillText('재단선',ax-3,ay-6);
  ctx.fillStyle='rgba(59,130,246,0.85)'; ctx.fillText('안전선',ax+8,ay+17);
  if (_dgSt.type==='keyring') {
    ctx.strokeStyle='rgba(239,68,68,0.75)'; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.arc(cfg.cw/2,ay-14,7,0,Math.PI*2); ctx.stroke();
    ctx.fillStyle='rgba(239,68,68,0.85)'; ctx.fillText('고리홀',cfg.cw/2-9,ay-23);
  }
  ctx.restore();
}

function dgToggleGuide(on) {
  _dgSt.showGuide=on;
  var tr=document.getElementById('dg-guide-track'), kn=document.getElementById('dg-guide-knob');
  if (tr) tr.style.background=on?'#86EFAC':'#D1D5DB';
  if (kn) kn.style.left=on?'19px':'1px';
  _dgRender();
}

function _dgRender() {
  var canvas=document.getElementById('dg-goods-canvas');
  if (!canvas) return;
  var t=_dgSt.type, cfg=_dgCfg[t];
  if (canvas.width!==cfg.cw || canvas.height!==cfg.ch) {
    canvas.width=cfg.cw; canvas.height=cfg.ch;
  }
  var ctx=canvas.getContext('2d');
  ctx.clearRect(0,0,cfg.cw,cfg.ch);
  _dgShape(ctx,t,_dgSt.color);
  _dgSt.stickers.forEach(function(st) {
    if (!st._img) return;
    var sz=(st.scale/100)*Math.min(cfg.cw,cfg.ch);
    var src=_dgSt.removeBg?_dgBgRemove(st._img):st._img;
    ctx.save(); ctx.translate(st.cx,st.cy);
    ctx.drawImage(src,-sz/2,-sz/2,sz,sz);
    if (_dgSt.selId===st.id) {
      ctx.strokeStyle='#C084FC'; ctx.lineWidth=2; ctx.setLineDash([3,2]);
      ctx.strokeRect(-sz/2-3,-sz/2-3,sz+6,sz+6); ctx.setLineDash([]);
    }
    ctx.restore();
  });
  if (_dgSt.showGuide) _dgDrawGuides(ctx, cfg);
}

function _dgSetupGoods() {
  var canvas=document.getElementById('dg-goods-canvas');
  if (!canvas) return;
  // Always (re)attach — called once per modal open, not inside render loop
  canvas.ondragover = function(e) { e.preventDefault(); };
  canvas.ondrop = function(e) {
    e.preventDefault();
    if (_dgSt.pocketDragIdx===null) return;
    var p=_dgPos(canvas,e), du=_dgSt.pocket[_dgSt.pocketDragIdx];
    var st={id:_dgSt.nextId++,dataUrl:du,cx:p.x,cy:p.y,scale:30,_img:null};
    var im=new Image(); im.onload=function(){st._img=im;_dgRender();}; im.src=du;
    _dgSt.stickers.push(st); _dgSt.pocketDragIdx=null;
  };
  canvas.onmousedown = function(e) {
    var cfg=_dgCfg[_dgSt.type], p=_dgPos(canvas,e);
    _dgSt.selId=null; _dgSt.goodsDrag=false;
    for (var i=_dgSt.stickers.length-1;i>=0;i--) {
      var st=_dgSt.stickers[i], sz=(st.scale/100)*Math.min(cfg.cw,cfg.ch);
      if (Math.abs(p.x-st.cx)<sz/2&&Math.abs(p.y-st.cy)<sz/2) {
        _dgSt.selId=st.id; _dgSt.goodsDrag=true;
        _dgSt.goodsDragId=st.id; _dgSt.goodsDragOx=p.x-st.cx; _dgSt.goodsDragOy=p.y-st.cy;
        var ctrl=document.getElementById('dg-sticker-ctrl');
        if (ctrl) { ctrl.style.display='block'; document.getElementById('dg-st-scale').value=st.scale; }
        break;
      }
    }
    if (!_dgSt.selId) { var ctrl=document.getElementById('dg-sticker-ctrl'); if(ctrl) ctrl.style.display='none'; }
    _dgRender();
  };
  canvas.onmousemove = function(e) {
    if (!_dgSt.goodsDrag) return;
    var p=_dgPos(canvas,e);
    _dgSt.stickers.forEach(function(s){if(s.id===_dgSt.goodsDragId){s.cx=p.x-_dgSt.goodsDragOx;s.cy=p.y-_dgSt.goodsDragOy;}});
    _dgRender();
  };
  canvas.onmouseup = canvas.onmouseleave = function() { _dgSt.goodsDrag=false; };
  function _t2e(e) { var t=e.touches[0]||e.changedTouches[0]; return {clientX:t.clientX,clientY:t.clientY}; }
  canvas.addEventListener('touchstart',function(e){e.preventDefault();canvas.onmousedown(_t2e(e));},{passive:false});
  canvas.addEventListener('touchmove', function(e){e.preventDefault();canvas.onmousemove(_t2e(e));},{passive:false});
  canvas.addEventListener('touchend',  function(e){e.preventDefault();_dgSt.goodsDrag=false;},{passive:false});
}

function dgScaleSticker(v) {
  _dgSt.stickers.forEach(function(s){if(s.id===_dgSt.selId)s.scale=parseInt(v);});
  _dgRender();
}

function dgDeleteSticker() {
  _dgSt.stickers=_dgSt.stickers.filter(function(s){return s.id!==_dgSt.selId;});
  _dgSt.selId=null;
  var ctrl=document.getElementById('dg-sticker-ctrl'); if(ctrl) ctrl.style.display='none';
  _dgRender();
}

function dgExportGoods() {
  var c=document.getElementById('dg-goods-canvas');
  if (!c) return;
  var a=document.createElement('a');
  a.href=c.toDataURL('image/png');
  a.download='goods_'+_dgSt.type+'_'+Date.now()+'.png';
  a.click();
}

function toggleStyleRef() {
  var det = document.querySelector('#style-ref-box details');
  if (det) { det.open = !det.open; return; }
  var btn = document.querySelector('#style-ref-box button[aria-expanded], #style-ref-box summary');
  if (btn) btn.click();
}

function dgSaveImage() {
  fetch('/current-image')
    .then(function(r) { return r.blob(); })
    .then(function(blob) {
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url;
      a.download = 'doodle_' + Date.now() + '.png';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });
}

function dgResetImage() {
  var btn = document.getElementById('dg-reset-hidden');
  if (btn) btn.click();
}

document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') dgCloseModal();
});
"""

# ── UI Layout ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="AI Doodle Character Sheet Generator", js=_CUSTOM_JS) as demo:

    # Goods modal at DOM root — must NOT be inside any element with CSS transform
    # (position:fixed breaks when ancestor has transform applied)
    gr.HTML(_GOODS_MODAL_HTML)

    with gr.Column(elem_id="app-header"):
        gr.Markdown(
            "# ✏️ AI Doodle Character Sheet 🎨✨\n"
            "캐릭터 사진 한 장으로 완성하는 고품질 손그림 다이어리 콜라주 ♡"
        )

    gr.Markdown("---")

    with gr.Row(equal_height=False):

        with gr.Column(scale=1, elem_id="left-panel"):
            gr.HTML("""<div style="margin-bottom:12px;">
  <span style="background:#FCE7F3;border:2px solid #4A3E3D;color:#4A3E3D;font-size:12px;
    font-weight:800;padding:4px 14px;border-radius:999px;display:inline-flex;align-items:center;gap:6px;
    box-shadow:1px 1px 0px #4A3E3D;">
    ✏️ 작업 지시서 작성</span>
</div>
<span style="background:#FEE2E2;border:2px solid #4A3E3D;color:#4A3E3D;font-size:10px;
  font-weight:700;padding:2px 8px;border-radius:6px;display:inline-block;margin-bottom:8px;
  box-shadow:2px 2px 0px #4A3E3D;">
  1단계: 스케치 원본 올리기</span>""")
            image_input = gr.Image(
                type="pil",
                label="",
                show_label=False,
                height=280,
            )
            mode_selector = gr.Radio(
                choices=[
                    ("🌸 풀 캐릭터 시트", "Full Character Sheet"),
                    ("🖼️ 포트레이트 낙서", "Portrait Doodle"),
                    ("🧜‍♀️ 상반신 캐릭터", "Upper Body Character"),
                    ("🍀 치비 스티커", "Chibi Sticker"),
                    ("✨ 심플 클린 포트레이트", "Simple Clean Portrait"),
                ],
                value="Full Character Sheet",
                label="",
                show_label=False,
                elem_id="mode-radio-group",
            )
            quality_selector = gr.Radio(
                choices=[
                    ("Low", "low"),
                    ("Medium", "medium"),
                    ("High", "high"),
                ],
                value="high",
                label="",
                show_label=False,
                elem_id="quality-radio-group",
            )
            gr.HTML(_MODE_BUTTONS_HTML)
            gr.HTML(_QUALITY_BUTTONS_HTML)
            generate_btn = gr.Button(
                "✨  낙서 시트 그리기  ♡",
                variant="primary",
                elem_id="generate-btn",
            )
            gr.Markdown(
                "💡 **Tip:** 정면 사진일수록 잘 나와요!  \n"
                "애니 캐릭터, 게임 캐릭터, 실제 사람 모두 OK ʕ•ᴥ•ʔ",
                elem_classes="tip-box",
            )

        with gr.Column(scale=1, elem_id="right-panel"):
            gr.HTML("""<div style="margin-bottom:10px;display:flex;align-items:center;justify-content:space-between;">
  <span style="background:#FEF08A;border:2px solid #4A3E3D;color:#4A3E3D;font-size:12px;
    font-weight:800;padding:4px 14px;border-radius:999px;display:inline-flex;align-items:center;gap:6px;
    box-shadow:1px 1px 0px #4A3E3D;">
    🎨 완성된 낙서 스케치북</span>
  <span style="background:#C084FC;color:white;font-size:9px;font-weight:700;
    padding:2px 10px;border-radius:999px;border:1px solid #4A3E3D;box-shadow:1px 1px 0px #4A3E3D;">AI Output Canvas</span>
</div>""")
            with gr.Row(visible=True) as empty_state_row:
                gr.HTML("""<div id="dg-output-empty">
  <div style="width:48px;height:48px;background:white;border:2px solid #4A3E3D;border-radius:50%;
    display:flex;align-items:center;justify-content:center;font-size:22px;
    box-shadow:2px 2px 0px #4A3E3D;margin-bottom:12px;">🎨</div>
  <div style="font-weight:800;font-size:15px;color:#7C6E6D;margin-bottom:8px;">스케치북이 비어있어요</div>
  <div style="font-size:12px;color:#8B7E7D;text-align:center;line-height:1.7;font-weight:600;">
    왼쪽 보드에 원본 사진을 넣고,<br>"낙서 시트 그리기" 버튼을 꼭 눌러주세요.</div>
</div>""")
            with gr.Row(visible=False) as loading_row:
                gr.HTML("""<div id="dg-loading">
  <div class="dg-spin">✏️</div>
  <div style="font-weight:800;font-size:15px;color:#7C6E6D;margin-bottom:8px;">AI가 낙서를 그리는 중...</div>
  <div style="font-size:12px;color:#8B7E7D;text-align:center;line-height:1.9;font-weight:600;">
    캐릭터 분석 → 프롬프트 빌드 → 이미지 생성<br>약 30~60초 걸려요 ♡ 잠깐만요!</div>
</div>""")
            image_display = gr.HTML("", elem_id="dg-image-display")
            reset_btn = gr.Button("↺", visible=False, elem_id="dg-reset-hidden")
            with gr.Row(visible=False) as goods_link_row:
                gr.HTML(_ACTION_BUTTONS_HTML)

    # ── Bottom tabs ───────────────────────────────────────────────────────────
    with gr.Tabs(elem_id="bottom-tabs"):

        with gr.Tab("🎀 낙서 스타일 장착실"):
            gr.HTML("""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px;">
  <div class="dg-style-card" onclick="setStyleCard(this,'pastel_watercolor')"
    style="cursor:pointer;border:2px solid #4A3E3D;border-radius:12px;background:#FCFAF6;
    box-shadow:2px 2px 0px #4A3E3D;overflow:hidden;transition:all 0.2s;"
    onmouseover="this.style.transform='translateY(-2px)'"
    onmouseout="this.style.transform='none'">
    <div style="background:#FFF0F5;height:60px;display:flex;align-items:center;justify-content:center;font-size:28px;">🍦</div>
    <div style="padding:8px 10px;text-align:center;">
      <div style="font-size:11px;font-weight:800;color:#4A3E3D;">파스텔 수채화</div>
    </div>
  </div>
  <div class="dg-style-card" onclick="setStyleCard(this,'pencil_sketch')"
    style="cursor:pointer;border:2px solid #8B7E7D;border-radius:12px;background:white;
    box-shadow:1px 1px 0px #8B7E7D;overflow:hidden;transition:all 0.2s;"
    onmouseover="this.style.transform='translateY(-2px)'"
    onmouseout="this.style.transform='none'">
    <div style="background:#F8FAFC;height:60px;display:flex;align-items:center;justify-content:center;font-size:28px;">✏️</div>
    <div style="padding:8px 10px;text-align:center;">
      <div style="font-size:11px;font-weight:800;color:#4A3E3D;">연필 스케치</div>
    </div>
  </div>
  <div class="dg-style-card" onclick="setStyleCard(this,'kawaii_sticker')"
    style="cursor:pointer;border:2px solid #8B7E7D;border-radius:12px;background:white;
    box-shadow:1px 1px 0px #8B7E7D;overflow:hidden;transition:all 0.2s;"
    onmouseover="this.style.transform='translateY(-2px)'"
    onmouseout="this.style.transform='none'">
    <div style="background:#E0F2FE;height:60px;display:flex;align-items:center;justify-content:center;font-size:28px;">🎟️</div>
    <div style="padding:8px 10px;text-align:center;">
      <div style="font-size:11px;font-weight:800;color:#4A3E3D;">스티커 팩</div>
    </div>
  </div>
  <div class="dg-style-card" onclick="setStyleCard(this,'crayon_doodle')"
    style="cursor:pointer;border:2px solid #8B7E7D;border-radius:12px;background:white;
    box-shadow:1px 1px 0px #8B7E7D;overflow:hidden;transition:all 0.2s;"
    onmouseover="this.style.transform='translateY(-2px)'"
    onmouseout="this.style.transform='none'">
    <div style="background:#FEF3C7;height:60px;display:flex;align-items:center;justify-content:center;font-size:28px;">🖍️</div>
    <div style="padding:8px 10px;text-align:center;">
      <div style="font-size:11px;font-weight:800;color:#4A3E3D;">크레용 낙서</div>
    </div>
  </div>
</div>
""")
            with gr.Accordion("📝 Style Reference 설정", open=False, elem_id="style-ref-box"):
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

        with gr.Tab("🖼️ 낙서 전시관"):
            gr.Markdown(
                f"test/ 폴더에서 {len(_gallery_images)}장을 불러왔어요 ♡",
                elem_classes="tip-box",
            )
            gr.Gallery(
                value=_gallery_images,
                label="",
                columns=3,
                height=600,
                object_fit="contain",
                show_label=False,
            )

        with gr.Tab("⚙️ 백엔드 설계실"):
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

    gr.HTML("""
<div style="text-align:center;padding:0.5rem 0 0.75rem;">
  <span style="background:#FEE2E2;border:2px solid #4A3E3D;color:#4A3E3D;font-size:11px;
    font-weight:800;padding:3px 14px;border-radius:6px;display:inline-block;box-shadow:2px 2px 0px #4A3E3D;">
    ✨ AI 낙서 생성 파이프라인 소개</span>
</div>
<div style="background:rgba(255,255,255,0.6);border:2px dashed #8B7E7D;border-radius:16px;
  padding:16px;margin-bottom:1rem;">
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;">
    <div style="display:flex;align-items:flex-start;gap:10px;">
      <div style="background:#FCE7F3;border:1px solid #4A3E3D;border-radius:50%;
        width:24px;height:24px;min-width:24px;
        display:flex;align-items:center;justify-content:center;
        font-weight:900;font-size:11px;color:#4A3E3D;margin-top:1px;">1</div>
      <div>
        <div style="font-weight:900;color:#4A3E3D;font-size:13px;margin-bottom:3px;">1단계 · 분석 🔍</div>
        <div style="font-size:11px;color:#7C6E6D;line-height:1.55;font-weight:600;">GPT-4o가 머릿결, 헤어, 패션 등 고유 특징 정보 파싱</div>
      </div>
    </div>
    <div style="display:flex;align-items:flex-start;gap:10px;">
      <div style="background:#FEF08A;border:1px solid #4A3E3D;border-radius:50%;
        width:24px;height:24px;min-width:24px;
        display:flex;align-items:center;justify-content:center;
        font-weight:900;font-size:11px;color:#4A3E3D;margin-top:1px;">2</div>
      <div>
        <div style="font-weight:900;color:#4A3E3D;font-size:13px;margin-bottom:3px;">2단계 · 빌드 🔥</div>
        <div style="font-size:11px;color:#7C6E6D;line-height:1.55;font-weight:600;">선택한 스타일 정보와 원본 데이터를 결합하여 프롬프트 정렬</div>
      </div>
    </div>
    <div style="display:flex;align-items:flex-start;gap:10px;">
      <div style="background:#D9F99D;border:1px solid #4A3E3D;border-radius:50%;
        width:24px;height:24px;min-width:24px;
        display:flex;align-items:center;justify-content:center;
        font-weight:900;font-size:11px;color:#4A3E3D;margin-top:1px;">3</div>
      <div>
        <div style="font-weight:900;color:#4A3E3D;font-size:13px;margin-bottom:3px;">3단계 · 그리기 🖌️</div>
        <div style="font-size:11px;color:#7C6E6D;line-height:1.55;font-weight:600;">gpt-image-1 드로잉 AI가 수채 채색 다꾸 디자인 완성</div>
      </div>
    </div>
  </div>
</div>
""")

    gr.HTML("""
<hr style="border:none;border-top:2px dashed #8B7E7D;margin:0.5rem 0;">
<p style="text-align:center;color:#7C6E6D;font-size:11px;font-weight:600;padding:0.5rem 0 1rem;">
  Premium Single-File SPA · Gradio Layout Rework · Powered by GPT-4o &amp; gpt-image-1
</p>
""")

    # ── Button wiring ──────────────────────────────────────────────────────────
    # Step 1: instant UI switch to loading state
    # Step 2: run pipeline (image_output is always in DOM — Gradio won't skip it)
    generate_btn.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        inputs=[],
        outputs=[empty_state_row, loading_row],
    ).then(
        fn=run_pipeline,
        inputs=[image_input, mode_selector, quality_selector],
        outputs=[image_display, analysis_out, prompt_out, token_out, style_ref_out, goods_link_row, empty_state_row, loading_row],
    )

    reset_btn.click(
        fn=lambda: (
            "", "", "", "", "",
            gr.update(visible=False),  # goods_link_row
            gr.update(visible=True),   # empty_state_row
            gr.update(visible=False),  # loading_row
        ),
        inputs=None,
        outputs=[image_display, analysis_out, prompt_out, token_out, style_ref_out, goods_link_row, empty_state_row, loading_row],
    )

    style_ref_btn.click(
        fn=run_generate_style_reference,
        inputs=[],
        outputs=[style_ref_content, style_ref_status],
    )



# ── Launch with custom routes ──────────────────────────────────────────────────

from fastapi.responses import HTMLResponse, Response as FastAPIResponse

_gradio_app, _, _ = demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    prevent_thread_lock=True,
    theme=gr.themes.Soft(
        primary_hue="pink",
        secondary_hue="purple",
        neutral_hue="pink",
        font=[gr.themes.GoogleFont("Nunito"), "sans-serif"],
        radius_size=gr.themes.sizes.radius_lg,
    ),
    css=CSS,
)


@_gradio_app.get("/goods-page", response_class=HTMLResponse)
async def _goods_page():
    p = pathlib.Path("goods_page.html")
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("<p>goods_page.html not found</p>", status_code=404)


@_gradio_app.get("/current-image")
async def _current_image():
    img_path = pathlib.Path("/tmp/.current_goods.png")
    if not img_path.exists():
        return FastAPIResponse(status_code=404)
    return FastAPIResponse(content=img_path.read_bytes(), media_type="image/png")


@_gradio_app.get("/goods-simulate")
async def _goods_simulate(type: str = "📸 포토카드"):
    img_path = pathlib.Path("/tmp/.current_goods.png")
    if not img_path.exists():
        return FastAPIResponse(status_code=404)
    image = Image.open(img_path).convert("RGB")
    result = simulate_goods(image, type)
    if result is None:
        return FastAPIResponse(status_code=400)
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return FastAPIResponse(content=buf.getvalue(), media_type="image/png")


demo.block_thread()
