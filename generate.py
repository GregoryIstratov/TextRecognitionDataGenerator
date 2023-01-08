import io
from pathlib import Path
import random
import string
import numpy as np
import cv2
import csv
from PIL import Image
import traceback

from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromWikipedia,
    GeneratorFromStrings,
    GeneratorFromGenerator
)

from trdg.utils import load_fonts, add_image_noise, apply_random_overexposure, debug

def union_fonts(a: list, b: list):
    a = {str(Path(x).name): x for x in a}
    b = {str(Path(x).name): x for x in b}
    c: dict = a | b
    r = [x for x in c.values()]
    return r

def random_numeric_string():  
    def gen1():
        d = random.randint(0, 99)
        m = random.randint(0, 99)
        y = random.randint(0, 9999)
        
        s = f"{d:02d}.{m:02d}.{y:04d}"
        
        return s
        
    def gen2():
        return ''.join(random.choices(population='0123456789.', k=random.randint(8,32))).strip()
        
    def gen3():
        return ''.join(random.choices(population=(string.ascii_uppercase*3 + '0123456789'*3 + 
                                                  "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ №"), k=random.randint(8,32))).strip()
    
    def gen4():
        return ''.join(random.choices(population='0123456789 №', k=random.randint(8,32))).strip()
    
    def gen5():
        return ''.join(random.choices(population='0123456789 ', k=random.randint(8,32))).strip()
    
    def gen6():
        return ''.join(random.choices(population=(string.ascii_uppercase+ ' '), k=random.randint(8,32))).strip()
    
    def gen7():
        return ''.join(random.choices(population="ЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮЁ ", k=random.randint(8,32))).strip()
    
    f = [gen1, gen2, gen3, gen4, gen5, gen6, gen7]
    
    c = random.randint(0, len(f) - 1)
        
    return f[c]()

def random_numeric_string_gen():
    while True:
        yield random_numeric_string()
        
     

class Generator:
    grayscale: bool = True
    NUM: int = -1
    height: int
    index: int = 0 
    blur: int
    skew_angle: int
    length: int
    rgb: bool
    sensitive: bool
    image_mode: str
    distortion_type: int = 0
    random_skew: bool = False
    random_blur: bool = False

    def __init__(self, max_len: int = -1, height: int = 64, blur: float = 1, random_blur: bool = True,
                 skew_angle: int = 0, length: int = 2, rgb: bool = False, sensitive: bool = False, aug_opts: dict = {}) -> None:
        self.height = height
        self.blur = blur
        self.random_blur = random_blur
        self.skew_angle = skew_angle
        self.length = length
        self.rgb = rgb
        self.sensitive = sensitive
        self.max_len = max_len
        self.image_dir = Path(__file__).parent / "trdg" / "images"
        self.aug_opts = aug_opts
        
        if len(self.aug_opts) == 0:
            raise RuntimeError("aug_options are empty")
        
        if not self.rgb:
            self.image_mode = 'L'
        else:
            self.image_mode = 'RGB'
        
        self.fonts_ru = load_fonts('ru')
        self.fonts_en = load_fonts('en')
        self.fonts_all = union_fonts(self.fonts_ru, self.fonts_en)
        
        self.generator_num = GeneratorFromGenerator(random_numeric_string_gen(), fonts=self.fonts_ru, size=self.height, 
                                                    random_blur=self.random_blur, blur=self.blur, skewing_angle=self.skew_angle, 
                                                    random_skew=self.random_skew,
                                                    image_dir=self.image_dir,
                                                    distorsion_type=self.distortion_type, image_mode=self.image_mode,
                                                    aug_opts=self.aug_opts)

        self.generator_ru = self.__create_dict_generator('ru', self.height, self.fonts_ru)
        self.generator_en = self.__create_dict_generator('en', self.height, self.fonts_all)
        self.generator_sym = GeneratorFromRandom(count=-1, length=self.length + 2, allow_variable=True, fonts=self.fonts_all, language="ru",
                                                use_letters=True, size=self.height, random_blur=self.random_blur, blur=self.blur, 
                                                skewing_angle=self.skew_angle, random_skew=self.random_skew, 
                                                distorsion_type=self.distortion_type,
                                                image_dir=self.image_dir,
                                                image_mode=self.image_mode,
                                                aug_opts=self.aug_opts
                                                )
        
        #self.gens = [self.generator_num, self.generator_ru, self.generator_en, self.generator_sym]
        self.gens = [self.generator_num, self.generator_ru, self.generator_en]
        #self.gens = [self.generator_ru]
        pass
        
    def __create_dict_generator(self, lang: str, height: int, fonts: list[str]):
        return GeneratorFromDict(count=-1, fonts=fonts, length=self.length, language=lang, 
                                    random_blur=self.random_blur, blur=self.blur, allow_variable=True,
                                    skewing_angle=self.skew_angle,
                                    random_skew=self.random_skew,
                                    distorsion_type=self.distortion_type,                                
                                    size=height,
                                    image_dir=self.image_dir,
                                    image_mode=self.image_mode,
                                    aug_opts=self.aug_opts
                                    )       
    def __next__(self):
        while True:
            try:
                c = self.index % len(self.gens)
                self.index = self.index + 1
                img, lbl = next(self.gens[c])
                
                if img is None:
                    print(f"Empty image generated from gen#{c} {str(self.gens[c])}, trying again...")
                    continue
                    
                if not self.sensitive:
                    lbl = lbl.upper()
                    
                if self.max_len != -1 and len(lbl) > self.max_len:
                    continue
                
                return img, lbl
            except Exception as e:
                #print(f"[TextGenerator] Failed to get new sample: \n{traceback.format_exc()}")
                print(f"[TextGenerator] Failed to get new sample: {e}")
                


if __name__ == "__main__":
    gen = Generator()
    
    create_dataset = False
    benchmark = False

    dataset_root = Path("text_dataset_100k_new")
    dataset_root.mkdir(exist_ok=True, parents=True)
    
    if benchmark:
        for i in range(500):
            _,_ = next(gen)
            
        exit()
    
    if create_dataset:
        csv_file = (dataset_root / "labels.csv").open("w", encoding="utf-8")
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["filename", "words"])    

    cv2.namedWindow("main")
    for i in range(100000):
        
        if cv2.getWindowProperty('main', 0) < 0:
            print("exit...")
            break
            
        img, lbl = next(gen)
        # cv2.imshow(f"main", np.asarray(img))
        # cv2.waitKey()
        debug(f"[{i}] {img.size} Text({len(lbl)}): {lbl}")
        
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