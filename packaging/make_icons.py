"""Render the SVG icon and crisp PNG/ICO files using PIL FreeType:  python packaging/make_icons.py"""
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parent.parent / "kinetra" / "assets"


def generate_all():
    ASSETS.mkdir(parents=True, exist_ok=True)
    icon_path = ASSETS / "icon.png"
    if not icon_path.exists():
        print("icon.png missing!")
        return

    icon = Image.open(str(icon_path)).convert("RGBA")

    # Target banner size
    im = Image.new("RGBA", (1640, 488), (0, 0, 0, 0))

    # Paste emblem
    icon_resized = icon.resize((440, 440), Image.Resampling.LANCZOS)
    im.paste(icon_resized, (24, 24), icon_resized)

    # Draw typography with system TrueType font
    draw = ImageDraw.Draw(im)
    font_path = r"C:\Windows\Fonts\segoeuib.ttf"
    if not os.path.exists(font_path):
        font_path = r"C:\Windows\Fonts\arialbd.ttf"

    font_kinetra = ImageFont.truetype(font_path, 170)
    font_sub = ImageFont.truetype(font_path, 56)

    # KINETRA
    draw.text((510, 85), "KINETRA", fill=(226, 232, 240, 255), font=font_kinetra)

    # Amber accent line
    draw.rounded_rectangle([(515, 290), (660, 302)], radius=5, fill=(255, 179, 0, 255))

    # MOTION STUDIO with letter spacing
    sub_text = "MOTION STUDIO"
    sub_x = 515
    sub_y = 325
    spacing = 14
    curr_x = sub_x
    for ch in sub_text:
        draw.text((curr_x, sub_y), ch, fill=(0, 240, 255, 255), font=font_sub)
        bbox = font_sub.getbbox(ch)
        w = bbox[2] - bbox[0]
        curr_x += w + spacing

    bbox = im.getbbox()
    pad = 24
    cropped = im.crop((max(0, bbox[0] - pad), max(0, bbox[1] - pad), min(im.width, bbox[2] + pad), min(im.height, bbox[3] + pad)))
    logo_path = ASSETS / "logo.png"
    cropped.save(str(logo_path), "PNG")
    print("Rendered crisp logo.png to", logo_path, cropped.size)


if __name__ == "__main__":
    generate_all()
