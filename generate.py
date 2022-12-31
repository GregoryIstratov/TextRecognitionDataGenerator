from pathlib import Path
import random
import numpy as np
import cv2
import csv
from PIL import Image

from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom,
    GeneratorFromWikipedia,
    GeneratorFromStrings,
    GeneratorFromGenerator
)

from trdg.utils import load_fonts, add_image_noise

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
        a = random.randint(0, 99)
        b = random.randint(0, 99)
        c = random.randint(100000, 999999)
        
        s = f"{a:02d} {b:02d}  {c:06d}"
        
        return s
        
    def gen3():
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        
        s = f"{a:03d}-{b:03d}"
        
        return s
    
    def gen4():
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        c = random.randint(100000, 999999)
        
        s = f"{a:03d},{b:03d}: ({c:06d})"
        
        return s    
    
    f = [gen1, gen2, gen3, gen4]
    
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

    def __init__(self, height: int = 96, blur: float = 2.5, skew_angle: int = 3, length: int = 2, rgb: bool = False, sensitive: bool = False) -> None:
        self.height = height
        self.blur = blur
        self.skew_angle = skew_angle
        self.length = length
        self.rgb = rgb
        self.sensitive = sensitive
        
        if not self.rgb:
            self.image_mode = 'L'
        else:
            self.image_mode = 'RGB'
        
        self.fonts_ru = load_fonts('ru')
        self.fonts_en = load_fonts('en')
        self.fonts_all = union_fonts(self.fonts_ru, self.fonts_en)
        
        self.generator_num = GeneratorFromGenerator(random_numeric_string_gen(), fonts=self.fonts_all, size=self.height, 
                                                    random_blur=True, blur=self.blur, skewing_angle=self.skew_angle, random_skew=True, 
                                                    distorsion_type=1, image_mode=self.image_mode)

        self.generator_ru = self.__create_dict_generator('ru', self.height, self.fonts_ru)
        self.generator_en = self.__create_dict_generator('en', self.height, self.fonts_all)
        self.generator_sym = GeneratorFromRandom(count=-1, length=self.length, allow_variable=True, fonts=self.fonts_all, language="en",
                                                use_letters=False, size=self.height, random_blur=True, blur=self.blur, 
                                                skewing_angle=self.skew_angle, random_skew=True, 
                                                distorsion_type=1,
                                                image_mode=self.image_mode
                                                )
        
        #self.gens = [self.generator_num, self.generator_ru, self.generator_en, self.generator_sym]
        #self.gens = [self.generator_num, self.generator_ru, self.generator_en]
        self.gens = [self.generator_ru]
        pass
        
    def __create_dict_generator(self, lang: str, height: int, fonts: list[str]):
        return GeneratorFromDict(count=-1, fonts=fonts, length=self.length, language=lang, 
                                    random_blur=True, blur=self.blur, allow_variable=True,
                                    skewing_angle=self.skew_angle,
                                    random_skew=True,
                                    distorsion_type=1,                                
                                    size=height,
                                    image_mode=self.image_mode
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
                
                enable_downsample = random.choices([True, False], weights=[0.75, 0.25], k=1)[0]
                
                if enable_downsample:
                    dfactor = random.choice([2,3,4])
                    img_np = np.asarray(img)
                    img_ds = cv2.resize(img_np, dsize=(img_np.shape[1] // dfactor, img_np.shape[0] // dfactor), interpolation=cv2.INTER_NEAREST)
                    img_np = cv2.resize(img_ds, dsize=(img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_LINEAR)
                    img = Image.fromarray(img_np)
                    
                if not self.sensitive:
                    lbl = lbl.upper()
                
                return img, lbl
            except Exception as e:
                print(f"[TextGenerator] Failed to get new sample: {str(e)}")


if __name__ == "__main__":
    gen = Generator()
    
    create_dataset = False

    dataset_root = Path("~/text_dataset_100k_new")
    dataset_root.mkdir(exist_ok=True, parents=True)
    
    if create_dataset:
        csv_file = (dataset_root / "labels.csv").open("w", encoding="utf-8")
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["filename", "words"])    

    for i in range(100000):
        img, lbl = next(gen)
        print(f"Len: {len(lbl)}")
        # cv2.imshow(f"main", np.asarray(img))
        # cv2.waitKey()
        print(f"[{i}] Text({len(lbl)}): {lbl}")
        
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