#!/usr/bin/env python3
"""
Downloads only the curated set of Piper TTS voice models needed for agents.
Each voice represents a distinct accent/region for agent personality.
"""
import os
import urllib.request

BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"

# ── Curated voice list ────────────────────────────────────────────
# Each entry: (model_name, description)
# All medium quality (~60MB each, 22.05kHz) — best balance of quality + performance
CURATED_VOICES = [
    # American English
    ("en_US-amy-medium",        "American female"),
    ("en_US-ryan-medium",       "American male"),
    ("en_US-lessac-medium",     "American neutral (default)"),
    # British English
    ("en_GB-alan-medium",       "British male"),
    ("en_GB-alba-medium",       "Scottish/British female"),
    # Indian
    ("hi_IN-pratham-medium",    "Indian male (Hindi)"),
    ("hi_IN-priyamvada-medium", "Indian female (Hindi)"),
    # European
    ("fr_FR-tom-medium",        "French male"),
    ("de_DE-thorsten-medium",   "German male"),
    # South American
    ("pt_BR-faber-medium",      "Brazilian Portuguese male"),
    # African
    ("sw_CD-lanfrica-medium",   "African (Swahili)"),
]


def download_file(url, dest):
    """Download a file if it doesn't already exist on disk."""
    if os.path.exists(dest):
        print(f"  Already exists: {os.path.basename(dest)}")
        return True
    try:
        print(f"  Downloading {os.path.basename(dest)}...")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  Failed to download {os.path.basename(dest)}: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def get_onnx_path(voice_name):
    """Build the HuggingFace path for a voice model's ONNX file."""
    # Piper voices are organized: {lang_code}/{name_without_quality}/{quality}/{name}.onnx
    # e.g. en/en_US/amy/medium/en_US-amy-medium.onnx
    parts = voice_name.rsplit("-", 1)  # ["en_US-amy", "medium"]
    name_part = parts[0]  # "en_US-amy"
    quality = parts[1]    # "medium"

    lang_parts = name_part.split("-", 1)  # ["en_US", "amy"]
    lang_code = lang_parts[0]            # "en_US"
    speaker = lang_parts[1]              # "amy"

    # Language family is the first two chars (en, hi, fr, de, pt, sw, zh)
    lang_family = lang_code.split("_")[0]

    return f"{lang_family}/{lang_code}/{speaker}/{quality}/{voice_name}"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download curated Piper TTS voice models")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "data", "piper_models"),
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Download] Output directory: {args.output_dir}")
    print(f"[Download] Downloading {len(CURATED_VOICES)} curated voice models...\n")

    success_count = 0
    for voice_name, description in CURATED_VOICES:
        print(f"[{voice_name}] — {description}")
        rel_path = get_onnx_path(voice_name)

        onnx_url = f"{BASE_URL}{rel_path}.onnx"
        json_url = f"{BASE_URL}{rel_path}.onnx.json"

        onnx_dest = os.path.join(args.output_dir, f"{voice_name}.onnx")
        json_dest = os.path.join(args.output_dir, f"{voice_name}.onnx.json")

        ok_onnx = download_file(onnx_url, onnx_dest)
        ok_json = download_file(json_url, json_dest)

        if ok_onnx and ok_json:
            success_count += 1
        print()

    print(f"[Download] Complete: {success_count}/{len(CURATED_VOICES)} voices downloaded successfully.")


if __name__ == "__main__":
    main()
