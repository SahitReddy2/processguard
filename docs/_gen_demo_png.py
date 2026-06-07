"""
Render the captured output of `python examples/synthetic_raw_loop_demo.py`
as a PNG into docs/demo.png. One-off script; rerun if the demo output
shape changes. Not packaged.
"""
from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT  = Path(__file__).parent / "demo.png"
TEXT = """\
$ python examples/synthetic_raw_loop_demo.py

[processguard] trace 75439536... started

=== ProcessGuard Raw Loop Demo ===

[step 01] web_search('RAG')
[step 02] web_search('RAG 2026')
[step 03] web_search('RAG latest')
[step 04] web_search('RAG newest')
[step 05] web_search('RAG most recent')
  [!] BEYOND-MAST no_progress_loop detected (confidence=1.00, action=steer)
      avg_novelty: 0.0
      novelty_threshold: 0.05
      window: 4
  -> steer injected: "Your recent tool calls are returning no new
     information. Try a different tool, different search terms, or
     a different approach."
  [step 06] read_paper(url='https://arxiv.org/abs/2503.13657')
  [step 07] writer.draft(...)
  [step 08] terminate (verified)

[processguard] trace 75439536... ended - 10 events, 1 detections

Total detections: 1
"""

# colours (dark terminal style)
BG          = (24, 26, 31)         # near-black
FG          = (220, 223, 228)      # warm white
PROMPT      = (130, 170, 255)      # blue $ prefix
COMMENT     = (130, 140, 150)      # grey [processguard] lines
DETECT      = (255, 180, 90)       # amber detection line
STEER       = (140, 220, 140)      # green steer line

# pick a monospace font that exists on Windows
def _font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in (
        "Consolas.ttf", "consola.ttf",
        "CascadiaCode.ttf", "CascadiaMono.ttf",
        "lucon.ttf",     # Lucida Console
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _line_colour(line: str) -> tuple[int, int, int]:
    s = line.lstrip()
    if line.startswith("$ "):
        return PROMPT
    if "[processguard]" in line or s.startswith("==="):
        return COMMENT
    if "[!]" in line or "detected" in line:
        return DETECT
    if "-> steer" in line or "information." in line or s.startswith("a different approach"):
        return STEER
    return FG


def render():
    font_size = 18
    font      = _font(font_size)
    pad_x     = 32
    pad_y     = 28

    lines = TEXT.splitlines()

    # measure
    # use a sample char to get monospace advance + line height
    bbox    = font.getbbox("M")
    char_w  = bbox[2] - bbox[0]
    line_h  = (bbox[3] - bbox[1]) + 6
    max_w   = max(len(l) for l in lines)

    img_w = pad_x * 2 + char_w * max_w
    img_h = pad_y * 2 + line_h * len(lines)

    img  = Image.new("RGB", (img_w, img_h), BG)
    draw = ImageDraw.Draw(img)

    y = pad_y
    for line in lines:
        draw.text((pad_x, y), line, font=font, fill=_line_colour(line))
        y += line_h

    img.save(OUT, "PNG", optimize=True)
    print(f"wrote {OUT} ({img_w}x{img_h})")


if __name__ == "__main__":
    render()
