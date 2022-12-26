from trdg.generators import (
    GeneratorFromDict,
    GeneratorFromRandom
)

import cv2
import numpy as np

generator = GeneratorFromDict(count=10000, length=4, language='ru', random_blur=True, blur=2, allow_variable=True)

i = 0
for img, lbl in generator:
    # Do something with the pillow images here.
    cv2.imshow(f"main{0}", np.asarray(img))
    i = i + 1

    if i % 1 == 0:
        cv2.waitKey()