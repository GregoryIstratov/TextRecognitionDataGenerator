from pathlib import Path
import numpy as np
import cv2
import csv

from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromWikipedia
)

from trdg.utils import load_fonts


fonts_ru = load_fonts('ru')
fonts_en = load_fonts('en')

def union_fonts(a: list, b: list):
    a = {str(Path(x).name): x for x in a}
    b = {str(Path(x).name): x for x in b}
    c: dict = a | b
    r = [x for x in c.values()]
    return r

fonts_all = union_fonts(fonts_ru, fonts_en)

generator_sym = GeneratorFromRandom(count=-1, length=3, allow_variable=True, fonts=fonts_ru, language="ru",
                                    use_letters=True,
                                    size=64, random_blur=True, blur=3, skewing_angle=4, random_skew=True, background_type=4,
                                    image_dir='trdg/images'
                                    )

def create_dict_generator(lang, fonts):
    return GeneratorFromDict(count=1000, fonts=fonts, length=3, language=lang, 
                                random_blur=True, blur=3, allow_variable=True,
                                skewing_angle=4,
                                random_skew=True,
                                background_type=4,
                                image_dir='trdg/images',
                                size=64
                                )

def create_wiki_generator(lang, fonts):
    return GeneratorFromWikipedia(count=-1, fonts=fonts, language=lang, 
                                random_blur=True, blur=3,
                                skewing_angle=4,
                                random_skew=True,
                                background_type=4,
                                image_dir='/home/greg/PycharmProjects/TextRecognitionDataGenerator/trdg/images',
                                size=64
                                )

generator_ru = create_dict_generator(lang='ru', fonts=fonts_ru)
generator_en = create_dict_generator(lang='en', fonts=fonts_all)

create_dataset = True

dataset_root = Path("./dataset")
dataset_root.mkdir(exist_ok=True, parents=True)

if create_dataset:
    csv_file = (dataset_root / "labels.csv").open("w", encoding="utf-8")
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["filename", "words"])

widths = []
for i in range(1000):

    c = i % 3
    #c = 1
    if c == 0:
        img, lbl = next(generator_ru)
    elif c == 1:
        img, lbl = next(generator_en)
    else:
        img, lbl = next(generator_sym)

    # Do something with the pillow images here.
    print(f"[{i}] Text: {lbl}")
    print(f"Image size: {np.asarray(img).shape}")
    widths.append(np.asarray(img).shape[1])
    
    print(f"Avg. width: {np.average(widths)}")
    
    if create_dataset:
        fname = f"{i}.png"
        fpath = dataset_root / fname
        img.save(str(fpath))
        csv_writer.writerow([str(fname), lbl])
    else:
        cv2.imshow(f"main", np.asarray(img))
        cv2.waitKey()

if create_dataset:
    csv_writer = None
    csv_file.close()
    
print("End")