from pathlib import Path
import random
import numpy as np
import cv2
import csv

from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromWikipedia,
    GeneratorFromStrings
)

from trdg.utils import load_fonts

def union_fonts(a: list, b: list):
    a = {str(Path(x).name): x for x in a}
    b = {str(Path(x).name): x for x in b}
    c: dict = a | b
    r = [x for x in c.values()]
    return r

class Generator:
    grayscale = True
    NUM = -1

    def __init__(self) -> None:
        self.fonts_ru = load_fonts('ru')
        self.fonts_en = load_fonts('en')
        self.fonts_all = union_fonts(fonts_ru, fonts_en)


def numeric_strings_gen(count=NUM):
    
    strings = []
    for i in range(int(count/2)):
        d = random.randint(0, 99)
        m = random.randint(0, 99)
        y = random.randint(0, 9999)
        
        s = f"{d:02d}.{m:02d}.{y:04d}"
        
        strings.append(s)
        
    for i in range(int(count/2)):
        a = random.randint(0, 99)
        b = random.randint(0, 99)
        c = random.randint(1000000000, 9999999999)
        
        s = f"{a:02d} {b:02d} {c:010d}"
        
        strings.append(s)
        
    for i in range(int(count/2)):
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        
        s = f"{a:03d}-{b:03d}"
        
        strings.append(s)
        
    return strings
        
    
generator_num = GeneratorFromStrings(numeric_strings_gen(), fonts=fonts_all, size=64, random_blur=True, blur=1.5, skewing_angle=4, random_skew=True, background_type=4, distorsion_type=1)
    

generator_sym_ru = GeneratorFromRandom(count=-1, length=2, allow_variable=True, fonts=fonts_ru, language="ru",
                                    use_letters=True,
                                    size=64, random_blur=False, blur=2, skewing_angle=4, random_skew=True, background_type=4,
                                    image_dir='trdg/images'
                                    )

generator_sym_en = GeneratorFromRandom(count=-1, length=2, allow_variable=True, fonts=fonts_en, language="en",
                                    use_letters=True,
                                    size=64, random_blur=True, blur=4, skewing_angle=4, random_skew=True, background_type=4,
                                    image_dir='trdg/images'
                                    )

def create_dict_generator(lang, fonts):
    return GeneratorFromDict(count=NUM, fonts=fonts, length=2, language=lang, 
                                random_blur=True, blur=1.5, allow_variable=True,
                                skewing_angle=4,
                                random_skew=True,
                                background_type=4,
                                distorsion_type=1,                                
                                size=64
                                )

def create_wiki_generator(lang, fonts):
    return GeneratorFromWikipedia(count=-1, fonts=fonts, language=lang, 
                                random_blur=True, blur=3,
                                skewing_angle=4,
                                random_skew=True,
                                background_type=4,
                                image_dir='trdg/images',
                                size=64
                                )

generator_ru = create_dict_generator(lang='ru', fonts=fonts_ru)
generator_en = create_dict_generator(lang='en', fonts=fonts_all)

create_dataset = True

dataset_root = Path("~/text_dataset_ru_gray_10k3")
dataset_root.mkdir(exist_ok=True, parents=True)

if create_dataset:
    csv_file = (dataset_root / "labels.csv").open("w", encoding="utf-8")
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["filename", "words"])

widths = []
for i in range(NUM):

    c = i % 2
    #c = 0
    if c == 0:
        img, lbl = next(generator_ru)
    elif c == 1:
        #img, lbl = next(generator_sym_ru)
        img, lbl = next(generator_num)
    elif c == 2:
        img, lbl = next(generator_en)
    else:
        img, lbl = next(generator_sym_en)

    # Do something with the pillow images here.
    print(f"[{i}] Text: {lbl}")
    print(f"Image size: {np.asarray(img).shape}")
    widths.append(np.asarray(img).shape[1])
    
    print(f"Avg. width: {np.average(widths)}")
    
    if grayscale:
        img = img.convert('L')
    
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
