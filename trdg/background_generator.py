import cv2
import math
import os
import random as rnd
import numpy as np
import numba
from numba import jit
from PIL import Image, ImageDraw, ImageFilter
from dataclasses import dataclass
from scipy.ndimage._filters import gaussian_filter

@jit
def clouds(arr: np.array, x: int, y: int, size: float, m:int = 128) -> np.float32:
    value: np.float32 = 0.0
    initial_size: np.float32 = size
    
    while size >= 1.0:
        value = value + arr[int(y // size), int(x // size)] * size
        size = size / 2.0
        
    return m * value / initial_size

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
    
@dataclass
class MyNoise_MARBLE:
    pass

MyNoise_BASIC = MyNoise_INTER | MyNoise_CLUSTER
MyNoiseType = MyNoise_BASIC | MyNoise_CLOUD

@jit
def marble(width: int, height: int, dfactor: int = 1) -> np.array:
    noise_min = 0.0
    noise_max = 1.0
        
    noise_width = width // dfactor
    noise_height = height // dfactor
    x_period = rnd.uniform(-16, 16)
    y_period = rnd.uniform(-8, 8)
    turb_power = rnd.uniform(0.5, 4)
    turb_size = rnd.uniform(0, 8)
    
    noise: np.array = np.random.uniform(noise_min, noise_max, (noise_height, noise_width))
    image = np.ones((height, width), dtype=np.uint8)
    
    col_avg = rnd.randint(120, 190)
    col_dev = 256 - col_avg
    
    for x in range(width):
        for y in range(height):
            xy_value = x * x_period / noise_width + y * y_period / noise_height + turb_power * clouds(noise, x, y, turb_size) / 256
            sine_value = col_avg * abs(math.sin(xy_value * math.pi)) + col_dev
            image[y, x] = np.uint8(sine_value)
            
    return image
            

def my_noise(height: int, width: int, type: MyNoiseType) -> Image:
    noise_min = 0.25
    noise_max = 0.75
    
    def __create_noise(size, inter):
        noise = np.ones((height//size, width//size), dtype=np.float32)
        cv2.randu(noise, noise_min, noise_max)
        #cv2.randn(noise, 0.75, 0.25)
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
                cv2.randu(noise, noise_min, noise_max)                
                image = np.ones((height, width), dtype=np.float32)
                noise = cv2.resize(noise, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
                for y in range(0, height):
                    for x in range(0, width):
                        image[y, x] = clouds(noise, x, y, cloud_size)
                return image
            case MyNoise_MARBLE:                
                image = marble(width, height)   
                image = gaussian_filter(image, sigma=1)
                return image              


    return Image.fromarray(__do_noise()).convert("RGBA")

def gaussian_noise(height: int, width: int) -> Image:
    """
    Create a background with Gaussian noise (to mimic paper)
    """

    # We create an all white image
    image = np.ones((height, width))

    # We add gaussian noise
    mean = 190
    std = 255 - mean
    cv2.randn(image, mean, std)

    return Image.fromarray(image).convert("RGBA")


def plain_color(height: int, width: int) -> Image:
    """
    Create a plain color background
    """
    
    color = rnd.randint(100, 235)

    return Image.new("L", (width, height), color).convert("RGBA")


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
            c = int(255 - round(200 * z / rotation_count))
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
