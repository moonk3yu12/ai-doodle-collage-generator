import gradio as gr
import openai
import base64
import os
import io
import pathlib
import requests
from PIL import Image


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


_HELP_HTML = """
<div class="dg-help-wrap" id="dg-help-wrap">
  <button class="dg-help-icon" id="dg-help-btn" aria-label="생성 팁">ⓘ</button>
  <div class="dg-help-pop" id="dg-help-pop">
    <div class="dg-help-title">💡 생성 팁</div>
    <ul class="dg-help-list">
      <li>정면 캐릭터일수록 잘 나와요</li>
      <li>얼굴이 크게 보이는 이미지가 좋아요</li>
      <li>머리 장식이 선명할수록 특징 보존이 잘 됩니다</li>
      <li>캐릭터가 화면에서 차지하는 비율이 클수록 좋아요</li>
      <li>Full Character Sheet는 생성마다 결과가 달라질 수 있어요</li>
    </ul>
  </div>
</div>
<style>
.dg-help-wrap {
  position: relative;
  display: flex;
  justify-content: flex-end;
  margin-bottom: 6px;
}
.dg-help-icon {
  background: linear-gradient(135deg, #f9a8d4, #d8b4fe);
  border: none;
  border-radius: 50%;
  width: 26px;
  height: 26px;
  font-size: 13px;
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 8px rgba(192,132,252,0.35);
  transition: transform 0.15s, box-shadow 0.15s;
  font-weight: 700;
  padding: 0;
  line-height: 1;
}
.dg-help-icon:hover {
  transform: scale(1.15);
  box-shadow: 0 4px 14px rgba(192,132,252,0.55);
}
.dg-help-pop {
  display: none;
  position: absolute;
  bottom: calc(100% + 8px);
  right: 0;
  width: 255px;
  background: white;
  border: 2px solid #e9d5ff;
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 8px 28px rgba(192,132,252,0.2);
  z-index: 9999;
}
.dg-help-wrap:hover .dg-help-pop,
.dg-help-pop.open { display: block; }
.dg-help-title {
  font-weight: 800;
  color: #c084fc !important;
  font-size: 0.88rem;
  margin-bottom: 8px;
}
.dg-help-list {
  margin: 0;
  padding-left: 16px;
  color: #9333ea !important;
  font-size: 0.82rem;
  font-weight: 600;
  line-height: 1.65;
}
.dg-help-list li {
  color: #9333ea !important;
}
</style>
<script>
(function(){
  var wrap = document.getElementById('dg-help-wrap');
  var btn  = document.getElementById('dg-help-btn');
  var pop  = document.getElementById('dg-help-pop');
  if(!btn||!pop) return;
  btn.addEventListener('click', function(e){
    e.stopPropagation();
    pop.classList.toggle('open');
  });
  document.addEventListener('click', function(e){
    if(wrap && !wrap.contains(e.target)) pop.classList.remove('open');
  });
})();
</script>
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
                    ("💸 Low", "low"),
                    ("⚡ Medium", "medium"),
                    ("✨ High", "high"),
                ],
                value="medium",
                label="🖼️ 이미지 퀄리티",
            )
            gr.HTML(_HELP_HTML)
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
