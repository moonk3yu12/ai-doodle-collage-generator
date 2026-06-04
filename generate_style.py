#!/usr/bin/env python3
"""
Run once to generate style_reference.txt from images in styles/.

Usage:
    Windows PowerShell:  $env:OPENAI_API_KEY="sk-..."; python generate_style.py
    macOS/Linux:         OPENAI_API_KEY="sk-..." python generate_style.py
"""

import openai
import base64
import os
import io
import pathlib
from PIL import Image

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


def pil_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        print("  Windows: $env:OPENAI_API_KEY='sk-...'")
        print("  macOS/Linux: export OPENAI_API_KEY='sk-...'")
        return

    client = openai.OpenAI(api_key=api_key)

    if not STYLES_DIR.exists():
        print(f"ERROR: {STYLES_DIR}/ folder not found.")
        return

    image_paths = sorted([
        *STYLES_DIR.glob("*.png"),
        *STYLES_DIR.glob("*.jpg"),
        *STYLES_DIR.glob("*.jpeg"),
    ])[:10]

    if not image_paths:
        print(f"ERROR: No PNG/JPG images found in {STYLES_DIR}/")
        return

    print(f"Found {len(image_paths)} sample image(s). Starting style analysis...\n")

    analyses = []
    for i, img_path in enumerate(image_paths):
        print(f"  [{i+1}/{len(image_paths)}] Analyzing {img_path.name}...")
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
        result = resp.choices[0].message.content.strip()
        analyses.append(result)
        print(f"     tokens used: {resp.usage.total_tokens}")

    print(f"\nSynthesizing common style from {len(analyses)} analyses...")
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
    print(f"  tokens used: {resp.usage.total_tokens}")

    STYLE_REF_FILE.write_text(style_ref, encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"style_reference.txt saved:\n")
    print(style_ref)
    print(f"{'='*60}")
    print(f"\nNext step: commit and push")
    print(f"  git add style_reference.txt styles/")
    print(f"  git commit -m 'feat: add style reference and sample images'")
    print(f"  git push origin main")


if __name__ == "__main__":
    main()
