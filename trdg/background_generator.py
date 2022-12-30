import cv2
import math
import os
import random as rnd
import numpy as np
import numba
from numba import jit
from PIL import Image, ImageDraw, ImageFilter
from dataclasses import dataclass

@jit
def clouds(arr: np.array, x: int, y: int, size: float) -> np.float32:
    value: np.float32 = 0.0
    initial_size: np.float32 = size
    
    while size >= 1.0:
        value = value + arr[int(y // size), int(x // size)] * size
        size = size / 2.0
        
    return 128 * value / initial_size

@dataclass
class MyNoise_CLUSTER:
    size: int

@dataclass
class MyNoise_INTER:
    size: int

@dataclass
class MyNoise_CLOUD:
    size: int
    cloud_size: int

MyNoise_BASIC = MyNoise_INTER | MyNoise_CLUSTER
MyNoiseType = MyNoise_BASIC | MyNoise_CLOUD

def my_noise(height: int, width: int, type: MyNoiseType) -> Image:
    print("My noise")
    
    def __create_noise(size, inter):
        noise = np.ones((height//size, width//size), dtype=np.float32)
        cv2.randu(noise, 0.3, 1)
        return cv2.resize(noise, dsize=(width, height), interpolation=inter)

    def __do_noise():
        match type:
            case MyNoise_CLUSTER(size):
                noise = __create_noise(size, cv2.INTER_NEAREST)
                return noise * 255
            case MyNoise_INTER(size):
                noise = __create_noise(size, cv2.INTER_LINEAR)
                return noise * 255
            case MyNoise_CLOUD(size, cloud_size):
                noise = np.ones((height//size, width//size), dtype=np.float32)
                cv2.randu(noise, 0.3, 1)                
                image = np.ones((height, width), dtype=np.float32)
                noise = cv2.resize(noise, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
                for y in range(0, height):
                    for x in range(0, width):
                        image[y, x] = clouds(noise, x, y, cloud_size)
                return image


    return Image.fromarray(__do_noise()).convert("RGBA")

def gaussian_noise(height: int, width: int) -> Image:
    """
    Create a background with Gaussian noise (to mimic paper)
    """

    print("Gauss noise")
    # We create an all white image
    image = np.ones((height, width))

    # We add gaussian noise
    cv2.randn(image, 200, 45)

    return Image.fromarray(image).convert("RGBA")


def plain_white(height: int, width: int) -> Image:
    """
    Create a plain white background
    """

    return Image.new("L", (width, height), 255).convert("RGBA")


@jit
def __quasicrystal(height: int, width: int) -> np.array:
    """
    Create a background with quasicrystal (https://en.wikipedia.org/wiki/Quasicrystal)
    """
        
    image = np.ones((height, width))

    frequency = rnd.random() * 10 + 5  # frequency
    phase = rnd.random() * 2 * math.pi  # phase
    rotation_count = rnd.randint(5, 15)  # of rotations

    for kw in range(width):
        y = float(kw) / (width - 1) * 4 * math.pi - 2 * math.pi
        for kh in range(height):
            x = float(kh) / (height - 1) * 4 * math.pi - 2 * math.pi
            z = 0.0
            for i in range(rotation_count):
                r = math.hypot(x, y)
                a = math.atan2(y, x) + i * math.pi * 2.0 / rotation_count
                z += math.cos(r * math.sin(a) * frequency + phase)
            c = int(255 - round(255 * z / rotation_count))
            image[kh, kw] = c  # grayscale
            
    return image

def quasicrystal(height: int, width: int) -> Image:

    image = __quasicrystal(height, width)
            
    return Image.fromarray(image).convert("RGBA")


def image(height: int, width: int, image_dir: str) -> Image:
    """
    Create a background with a image
    """
    images = os.listdir(image_dir)

    if len(images) > 0:
        pic = Image.open(
            os.path.join(image_dir, images[rnd.randint(0, len(images) - 1)])
        )

        if pic.size[0] < width:
            pic = pic.resize(
                [width, int(pic.size[1] * (width / pic.size[0]))],
                Image.Resampling.LANCZOS,
            )
        if pic.size[1] < height:
            pic = pic.resize(
                [int(pic.size[0] * (height / pic.size[1])), height],
                Image.Resampling.LANCZOS,
            )

        if pic.size[0] == width:
            x = 0
        else:
            x = rnd.randint(0, pic.size[0] - width)
        if pic.size[1] == height:
            y = 0
        else:
            y = rnd.randint(0, pic.size[1] - height)

        return pic.crop((x, y, x + width, y + height))
    else:
        raise Exception("No images where found in the images folder!")
