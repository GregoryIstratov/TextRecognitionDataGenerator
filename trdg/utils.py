"""
Utility functions
"""

import os
import random
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

def debug(msg: str):
    dt = time.strftime("%H:%M:%S")
    print(f"[{dt}][DBG]: {msg}")
    pass

def add_image_noise(image: Image) -> Image:
    img = np.asarray(image).astype(np.float32)
    
    #c = random.randint(1, 1)
    c = 1

    noise = np.ones((img.shape[0] // c, img.shape[1] // c), dtype=np.float32)
    #noise = np.ones((img.shape[0] * c, img.shape[1] * c), dtype=np.float32)
    #cv2.randu(noise, 1, 3)
    cv2.randn(noise, 1, 0.2)
    #noise = cv2.resize(noise, dsize=(img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    #noise = cv2.blur(noise, (3,3))
        
    if len(img.shape) > 2 and img.shape[2] == 3:
        img[:,:, 0] = img[:,:, 0] * noise
        img[:,:, 1] = img[:,:, 1] * noise
        img[:,:, 2] = img[:,:, 2] * noise
    else:
        img = img * noise
    
    #img[img>255] = 255
    return Image.fromarray(img.astype(np.uint8))


def random_upper(s: str):
    p = random.random()
    if p <= 0.4:
        p2 = random.random()
        if p2 <= 0.6:
            return s.upper()
        
        return s[0].upper() + s[1:]
    return s


def load_dict(path: str) -> List[str]:
    """Read the dictionnary file and returns all words in it."""

    word_dict = []
    with open(
        path,
        "r",
        encoding="utf8",
        errors="ignore",
    ) as d:
        # word_dict = [l for l in d.read().splitlines() if len(l) > 0]
        word_dict = []
        for l in d.read().splitlines():
            if len(l) <= 0:
                continue
            word_dict.append(l.upper())

    return word_dict


def load_fonts(lang: str) -> List[str]:
    """Load all fonts in the fonts directories"""
    
    def __load_dir(p: Path):
        fonts = []
        for item in p.iterdir():
            if item.is_file() and item.name.lower().endswith((".ttf", ".otf")):
                fonts.append(item)
            elif item.is_dir():
                fonts += __load_dir(item)
            else:
                print(f"[load_fonts] Skipping {item}")
                continue
        return fonts

    if lang == "en":
        lang = "latin"
    
    fonts: list[Path] = []
    curdir = Path(os.path.dirname(__file__)) / "fonts"
    langs = [x for x in curdir.iterdir() if x.is_dir() and str(x.stem) == lang]
    if len(langs) > 0:
        fonts = __load_dir(langs[0])
    else:
        raise FileNotFoundError(f"{lang} not found")
    
    
    fonts = [str(x) for x in fonts if x.is_file()]
    return fonts


def mask_to_bboxes(mask: List[Tuple[int, int, int, int]], tess: bool = False):
    """Process the mask and turns it into a list of AABB bounding boxes"""

    mask_arr = np.array(mask)

    bboxes = []

    i = 0
    space_thresh = 1
    while True:
        try:
            color_tuple = ((i + 1) // (255 * 255),
                           (i + 1) // 255, (i + 1) % 255)
            letter = np.where(np.all(mask_arr == color_tuple, axis=-1))
            if space_thresh == 0 and letter:
                x1 = min(bboxes[-1][2] + 1, np.min(letter[1]) - 1)
                y1 = (
                    min(bboxes[-1][3] + 1, np.min(letter[0]) - 1)
                    if not tess
                    else min(
                        mask_arr.shape[0] -
                        np.min(letter[0]) + 2, bboxes[-1][1] - 1
                    )
                )
                x2 = max(bboxes[-1][2] + 1, np.min(letter[1]) - 2)
                y2 = (
                    max(bboxes[-1][3] + 1, np.min(letter[0]) - 2)
                    if not tess
                    else max(
                        mask_arr.shape[0] -
                        np.min(letter[0]) + 2, bboxes[-1][1] - 1
                    )
                )
                bboxes.append((x1, y1, x2, y2))
                space_thresh += 1
            bboxes.append(
                (
                    max(0, np.min(letter[1]) - 1),
                    max(0, np.min(letter[0]) - 1)
                    if not tess
                    else max(0, mask_arr.shape[0] - np.max(letter[0]) - 1),
                    min(mask_arr.shape[1] - 1, np.max(letter[1]) + 1),
                    min(mask_arr.shape[0] - 1, np.max(letter[0]) + 1)
                    if not tess
                    else min(
                        mask_arr.shape[0] - 1, mask_arr.shape[0] -
                        np.min(letter[0]) + 1
                    ),
                )
            )
            i += 1
        except Exception as ex:
            if space_thresh == 0:
                break
            space_thresh -= 1
            i += 1

    return bboxes


def draw_bounding_boxes(
    img: Image, bboxes: List[Tuple[int, int, int, int]], color: str = "green"
) -> None:
    d = ImageDraw.Draw(img)

    for bbox in bboxes:
        d.rectangle(bbox, outline=color)


def make_filename_valid(value: str, allow_unicode: bool = False) -> str:
    """
    Code adapted from: https://docs.djangoproject.com/en/4.0/_modules/django/utils/text/#slugify

    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[^\w\s-]", "", value)

    # Image names will be shortened to avoid exceeding the max filename length
    return value[:200]


def get_text_width(image_font: ImageFont, text: str) -> int:
    """
    Get the width of a string when rendered with a given font
    """
    return round(image_font.getlength(text))


def get_text_height(image_font: ImageFont, text: str) -> int:
    """
    Get the width of a string when rendered with a given font
    """
    return image_font.getsize(text)[1]
