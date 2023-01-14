import json
import os
import random
from typing import List, Tuple

from trdg.data_generator import FakeTextDataGenerator
from trdg.utils import load_dict, load_fonts

# support RTL
from arabic_reshaper import ArabicReshaper
from bidi.algorithm import get_display


class GeneratorFromGenerator:
    """Generator that uses a given generator"""

    def __init__(
        self,
        generator,
        fonts: List[str] = [],
        language: str = "en",
        size: int = 32,
        skewing_angle: int = 0,
        random_skew: bool = False,
        blur: int = 0,
        random_blur: bool = False,
        background_type: int = 0,
        distorsion_type: int = 0,
        distorsion_orientation: int = 0,
        is_handwritten: bool = False,
        width: int = -1,
        alignment: int = 1,
        text_color: str = "#282828",
        orientation: int = 0,
        space_width: float = 1.0,
        character_spacing: int = 0,
        margins: Tuple[int, int, int, int] = (5, 5, 5, 5),
        fit: bool = False,
        output_mask: bool = False,
        word_split: bool = False,
        image_dir: str = os.path.join(
            "..", os.path.split(os.path.realpath(__file__))[0], "images"
        ),
        stroke_width: int = 0,
        stroke_fill: str = "#282828",
        image_mode: str = "RGB",
        output_bboxes: int = 0,
        rtl: bool = False,
        aug_opts: dict = {}
    ):
        self.generator = generator
        self.fonts = fonts
        if len(fonts) == 0:
            self.fonts = load_fonts(language)
        self.rtl = rtl
        self.orig_strings = []
        if self.rtl:
            if language == "ckb":
                ar_reshaper_config = {"delete_harakat": True, "language": "Kurdish"}
            else:
                ar_reshaper_config = {"delete_harakat": False}
            self.rtl_shaper = ArabicReshaper(configuration=ar_reshaper_config)
            # save a backup of the original strings before arabic-reshaping
            self.orig_strings = self.strings
            # reshape the strings
            self.strings = self.reshape_rtl(self.strings, self.rtl_shaper)
        self.language = language
        self.size = size
        self.skewing_angle = skewing_angle
        self.random_skew = random_skew
        self.blur = blur
        self.random_blur = random_blur
        self.background_type = background_type
        self.distorsion_type = distorsion_type
        self.distorsion_orientation = distorsion_orientation
        self.is_handwritten = is_handwritten
        self.width = width
        self.alignment = alignment
        self.text_color = text_color
        self.orientation = orientation
        self.space_width = space_width
        self.character_spacing = character_spacing
        self.margins = margins
        self.fit = fit
        self.output_mask = output_mask
        self.word_split = word_split
        self.image_dir = image_dir
        self.output_bboxes = output_bboxes
        self.generated_count = 0
        self.stroke_width = stroke_width
        self.stroke_fill = stroke_fill
        self.image_mode = image_mode
        self.aug_opts = aug_opts

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        self.generated_count += 1
        text = next(self.generator)
        font = self.fonts[random.randint(0, len(self.fonts) - 1)]
        
        return (
            FakeTextDataGenerator.generate(
                index=self.generated_count,
                text=text,
                font=font,
                out_dir=None,
                size=self.size,
                extension=None,
                skewing_angle=self.skewing_angle,
                random_skew=self.random_skew,
                blur=self.blur,
                random_blur=self.random_blur,
                background_type=self.background_type,
                distorsion_type=self.distorsion_type,
                distorsion_orientation=self.distorsion_orientation,
                is_handwritten=self.is_handwritten,
                name_format=0,
                width=self.width,
                alignment=self.alignment,
                text_color=self.text_color,
                orientation=self.orientation,
                space_width=self.space_width,
                character_spacing=self.character_spacing,
                margins=self.margins,
                fit=self.fit,
                output_mask=self.output_mask,
                word_split=self.word_split,
                image_dir=self.image_dir,
                stroke_width=self.stroke_width,
                stroke_fill=self.stroke_fill,
                image_mode=self.image_mode,
                output_bboxes=self.output_bboxes,
                dummy=-1,
                settings=self.aug_opts,
            ),
            text
        )

    def reshape_rtl(self, strings: list, rtl_shaper: ArabicReshaper):
        # reshape RTL characters before generating any image
        rtl_strings = []
        for string in strings:
            reshaped_string = rtl_shaper.reshape(string)
            rtl_strings.append(get_display(reshaped_string))
        return rtl_strings


if __name__ == "__main__":
    from trdg.generators.from_wikipedia import GeneratorFromWikipedia

    s = GeneratorFromWikipedia("test")
    next(s)
