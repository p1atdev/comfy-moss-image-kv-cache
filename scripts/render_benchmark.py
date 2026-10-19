"""Render the BF16 timing chart and the seed-42 long-prompt comparison."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def render(source: Path, image_root: Path, output: Path) -> None:
    data = json.loads(source.read_text())
    output.mkdir(parents=True, exist_ok=True)
    variants = ("joint", "kv", "kv_cal100")
    labels = ("Joint attention", "Text KV cache", "KV + correction LoRA")
    colors = ("#53657f", "#46a99b", "#126c60")
    baseline = data["measurements"]["long_joint"]["median_seconds"]
    corrected = data["measurements"]["long_kv_cal100"]["median_seconds"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    fig.subplots_adjust(left=0.25, right=0.91, top=0.75, bottom=0.22, hspace=0.65)
    fig.patch.set_facecolor("#ffffff")
    fig.text(
        0.04, 0.94, "MOSS-Image: text KV caching in ComfyUI", size=19, weight="bold"
    )
    fig.text(
        0.04,
        0.89,
        "BF16 · DMD 0.7 · Euler/simple · 8 steps · CFG 1 · 1024 × 1024",
        color="#53657f",
    )
    fig.text(
        0.04,
        0.81,
        f"Long prompt: {baseline / corrected:.2f}× speedup vs joint  |  "
        f"{100 * (1 - corrected / baseline):.0f}% less time",
        size=15,
        color="#126c60",
        weight="bold",
    )
    for ax, case in zip(axes, data["prompts"], strict=True):
        rows = [data["measurements"][f"{case['id']}_{v}"] for v in variants]
        values = [r["median_seconds"] for r in rows]
        error = (
            [r["median_seconds"] - r["min_seconds"] for r in rows],
            [r["max_seconds"] - r["median_seconds"] for r in rows],
        )
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#e8ebee")
        ax.barh(range(3), values, color=colors, height=0.60, xerr=error, capsize=3)
        ax.set_yticks(range(3), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 9.3)
        ax.set_title(
            f"{case['label']} · {case['reference_tokens']:,} tokens",
            loc="left",
            size=12,
            pad=10,
        )
        for index, value in enumerate(values):
            ax.text(value + 0.20, index, f"{value:.2f} s", va="center", weight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0)
    axes[-1].set_xlabel("Seconds per saved image  ·  lower is faster", labelpad=9)
    fig.text(
        0.04,
        0.05,
        "RTX 6000 Ada · median of 3 runs after warmup · whiskers: min–max\n"
        "Includes KV construction + sampling + VAE + PNG saving; excludes text encoding and initial model loading.",
        size=9,
        color="#53657f",
    )
    fig.savefig(output / "kv_cache_benchmark.png", dpi=180)
    fig.savefig(output / "kv_cache_benchmark.svg")
    svg = output / "kv_cache_benchmark.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(fig)

    tile, gap, top, bottom = 512, 16, 84, 44
    sheet = Image.new("RGB", (3 * tile + 4 * gap, top + tile + bottom), "white")
    draw = ImageDraw.Draw(sheet)
    title = ImageFont.load_default(size=24)
    font = ImageFont.load_default(size=18)
    draw.text(
        (gap, 12),
        "Same long prompt and seed 42 · BF16 · DMD 0.7 · 8 steps",
        font=title,
        fill="#172637",
    )
    for index, (variant, label) in enumerate(zip(variants, labels, strict=True)):
        image = data["measurements"][f"long_{variant}"]["runs"][0]["images"][0]
        x = gap + index * (tile + gap)
        draw.text((x, 50), label, font=font, fill="#172637")
        with Image.open(
            image_root / image["subfolder"] / image["filename"]
        ) as original:
            sheet.paste(
                original.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS),
                (x, top),
            )
    draw.text(
        (gap, top + tile + 12),
        "KV changes attention behavior; the correction does not reproduce joint-attention pixels.",
        font=font,
        fill="#53657f",
    )
    sheet.save(output / "long_prompt_comparison.jpg", quality=92, optimize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.source, args.image_root, args.output)
