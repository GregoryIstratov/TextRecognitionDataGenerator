import io
import json
import math
import os
import random as rnd
import traceback

from PIL import Image, ImageFilter, ImageStat, ImageEnhance

from trdg import computer_text_generator, background_generator, distorsion_generator
from trdg.utils import mask_to_bboxes, make_filename_valid, add_image_noise, apply_random_overexposure, debug, error, AttrDict
from trdg.background_generator import MyNoise_INTER, MyNoise_CLUSTER, MyNoise_CLOUD, MyNoise_MARBLE

import numpy as np
import cv2
import random as rnd
from enum import Enum
import graphlib

class DistortType(Enum):
    NONE = -1
    TEXT_BLUR = 0
    TEXT_NOISE = 1
    BLUR = 2
    MOTION_BLUR = 3
    SHARPEN = 4
    DOWNSAMPLE = 5
    JPEG_ARTIFACT = 6
    SIN_COS = 7
    NOISE = 8

def randfloat(a: float, b: float) -> float:
    return np.interp(rnd.random(), [0, 1], [a, b])

def cv_interpolation_from_string(s: str):
    match s.lower():
        case "nearest":
            return cv2.INTER_NEAREST
        case "linear":
            return cv2.INTER_LINEAR
        case "cubic":
            return cv2.INTER_CUBIC
        case _:
            raise RuntimeError(f"Unknown interpolation type: {s}")

class Augmentation(object):
    def apply(self, image: Image) -> Image:
        raise NotImplementedError
    
class AugmentationNone(Augmentation):
    def apply(self, image: Image) -> Image:
        return image 
    
class AugmentationSinCos(Augmentation):
    def __init__(self, opt:AttrDict) -> None:
        super().__init__()
        self.opt = AttrDict(opt) 
        
    def apply(self, image: Image) -> Image:
        max_offset = int(image.height**self.opt.max_offset)
        vertical = True
        horizontal = False
        
        func_sin = lambda x: int(math.sin(math.radians(x)) * max_offset)
        func_cos = lambda x: int(math.cos(math.radians(x)) * max_offset)
        func_rand = lambda x: rnd.randint(0, max_offset)
        
        func = rnd.choice([func_sin, func_cos, func_rand])
        
        image = distorsion_generator.apply_func_distorsion1(
            image,
            vertical,
            horizontal,
            max_offset,
            func,
        )
        
        return image
    
class AugmentationBlur(Augmentation):
    def __init__(self, opt:AttrDict) -> None:
        super().__init__()
        self.opt = AttrDict(opt)
        
    def apply(self, image: Image) -> Image:
        r = randfloat(self.opt.radius_min, self.opt.radius_max)
        blur = ImageFilter.GaussianBlur(radius=r)
        
        return image.filter(blur)
    
class AugmentationMotionBlur(Augmentation):
    def __init__(self, opt:AttrDict) -> None:
        super().__init__()
        self.opt = AttrDict(opt) 
        
    def apply(self, image: Image) -> Image:
        size = rnd.randint(self.opt.size_min, self.opt.size_max)
        img = np.asarray(image)
        k = np.zeros((size, size), dtype=np.float32)
        k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
        angle = rnd.randint(0, 360)
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )  
        k = k * ( 1.0 / np.sum(k) )        
        return Image.fromarray(cv2.filter2D(img, -1, k))
    
class AugmentationJpegArtifact(Augmentation):
    def __init__(self, opt:AttrDict) -> None:
        super().__init__()
        self.opt = AttrDict(opt) 
        
    def apply(self, image: Image) -> Image:
        q = rnd.randint(self.opt.quality_min, self.opt.quality_max)
        with io.BytesIO() as buff:
            image.save(buff, format='JPEG', quality=q)
            image = Image.open(buff, formats=["JPEG"]).copy()
            return image
        
class AugmentationDownsample(Augmentation):
    def __init__(self, opt:AttrDict) -> None:
        super().__init__()
        self.opt = AttrDict(opt) 
        
    def apply(self, image: Image) -> Image:
        dfactor = rnd.randint(self.opt.factor_min, self.opt.factor_max)
        img_np = np.asarray(image)
        img_ds = cv2.resize(img_np, dsize=(img_np.shape[1] // dfactor, img_np.shape[0] // dfactor), interpolation=cv2.INTER_NEAREST)
        img_np = cv2.resize(img_ds, dsize=(img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        return Image.fromarray(img_np)
    
class AugmentationSharpen(Augmentation):
    def __init__(self, opt:AttrDict) -> None:
        super().__init__()
        self.opt = AttrDict(opt)
        
    def apply(self, image: Image) -> Image:
        sharp = ImageEnhance.Sharpness(image)
        factor = randfloat(self.opt.sharpness_min, self.opt.sharpness_max)
        return sharp.enhance(factor)
    
class AugmentationNoise(Augmentation):
    def __init__(self, opt:AttrDict) -> None:
        super().__init__()
        self.opt = AttrDict(opt) 
        
    def apply(self, image: Image) -> Image:
        img = np.asarray(image).astype(np.float32)
        
        #c = random.randint(1, 1)
        c = rnd.randint(self.opt.downsample_factor_min, self.opt.downsample_factor_max)
        inter = cv_interpolation_from_string(self.opt.interpolation)

        noise = np.ones((img.shape[0] // c, img.shape[1] // c), dtype=np.float32)
        #noise = np.ones((img.shape[0] * c, img.shape[1] * c), dtype=np.float32)
        #cv2.randu(noise, 1, 3)
        stddev = randfloat(self.opt.stddev_min, self.opt.stddev_max)
        cv2.randn(noise, self.opt.mean, stddev)
        if c > 1:
            noise = cv2.resize(noise, dsize=(img.shape[1], img.shape[0]), interpolation=inter)
        #noise = cv2.blur(noise, (3,3))
            
        if len(img.shape) > 2 and (img.shape[2] == 3 or img.shape[2] == 4):
            img[:,:, 0] = cv2.multiply(img[:,:, 0], noise)
            img[:,:, 1] = cv2.multiply(img[:,:, 1], noise)
            img[:,:, 2] = cv2.multiply(img[:,:, 2], noise)
        else:
            img = cv2.multiply(img, noise)
        
        img[img>255] = 255
        return Image.fromarray(img.astype(np.uint8))
    
class AugmentationPipeline(Augmentation):
    def append(self, aug: Augmentation):
        raise NotImplementedError    
    
class AugmentationPipelineRandomUniform(AugmentationPipeline):    
    def __init__(self, k: int) -> None:
        super().__init__()
        self.augs: list[Augmentation] = []
        self.k = k
        
    def append(self, aug: Augmentation):
        self.augs.append(aug)
        
    def apply(self, image: Image) -> Image:
        indices = list(range(0, len(self.augs)))
        chosen = rnd.choices(population=indices, k=self.k)
        chosen = sorted(chosen)
        for i in chosen:
            image = self.augs[i].apply(image)
            
        return image
    
    
class AugmentationPipelineRandomWeighted(AugmentationPipeline):
    def __init__(self, k: int, weights) -> None:
        super().__init__()
        self.augs: list[Augmentation] = []        
        self.k = k
        self.weights = weights
        
    def append(self, aug: Augmentation):
        self.augs.append(aug)
        
    def apply(self, image: Image) -> Image:
        indices = list(range(0, len(self.augs)))
        chosen = rnd.choices(population=indices, weights=self.weights, k=self.k)
        chosen = sorted(chosen)
        for i in chosen:
            image = self.augs[i].apply(image)
            
        return image
    
class AugmentationPipelineSequential(AugmentationPipeline):    
    def __init__(self) -> None:
        self.augs: list[Augmentation] = []
        
    def append(self, aug: Augmentation):
        self.augs.append(aug)
        
    def apply(self, image: Image) -> Image:
        for aug in self.augs:
            image = aug.apply(image)
            
        return image
    

def create_augmentation_instance(instance_opt: AttrDict) -> Augmentation:
    t = instance_opt.type.lower()
    match t:
        case "blur":
            return AugmentationBlur(instance_opt.opt)
        case "motion_blur":
            return AugmentationMotionBlur(instance_opt.opt)
        case "sharpen":
            return AugmentationSharpen(instance_opt.opt)
        case "downsample":
            return AugmentationDownsample(instance_opt.opt)
        case "jpeg_artifact":
            return AugmentationJpegArtifact(instance_opt.opt)
        case "noise":
            return AugmentationNoise(instance_opt.opt)
        case "sin_cos":
            return AugmentationSinCos(instance_opt.opt)
        case "none":
            return AugmentationNone()
        case _:
            raise RuntimeError(f"Unknown augmentation type '{t}'")

def build_augmentation_pipeline(opt_augs: AttrDict) -> dict[str, AugmentationPipeline]:
    def build_instances(instances_opt: AttrDict) -> dict[str, Augmentation]:
        instances = dict[str, Augmentation]()
        for name, v in instances_opt.items():
            instances[name] = create_augmentation_instance(AttrDict(v))
        return instances
    
    def build_pipeline(pipeline_opt: AttrDict, instances: dict[str, Augmentation]):
        t = pipeline_opt.type.lower()
        pipeline: AugmentationPipeline = None
        match t:
            case "random_uniform":
                pipeline = AugmentationPipelineRandomUniform(pipeline_opt.k)
            case "random_weighted":
                pipeline = AugmentationPipelineRandomWeighted(pipeline_opt.k, pipeline_opt.weights)
            case "sequential":
                pipeline = AugmentationPipelineSequential()
            case _:
                raise RuntimeError(f"Unknown pipiline type '{t}'")
        for aug_name in pipeline_opt.augs:
            if aug_name not in instances.keys():
                raise RuntimeError(f"instance '{aug_name}' does not exist")
            
            aug = instances[aug_name]
            pipeline.append(aug)
            
        return pipeline
    
    def get_build_order(pipelines_opt: AttrDict, instances: dict[str, Augmentation]):
        G = {}
        for name, v in pipelines_opt.items():
            G[name] = {aug_name for aug_name in AttrDict(v).augs if aug_name not in instances.keys()}
        ts = graphlib.TopologicalSorter(G)
        order = list(ts.static_order())
        return order
    
    instances = build_instances(opt_augs.instances)
    order = get_build_order(opt_augs.pipelines, instances)
    
    pipelines = {}
    for pl_name in order:
        pl = build_pipeline(AttrDict(opt_augs.pipelines[pl_name]), instances)
        instances[pl_name] = pl
        pipelines[pl_name] = pl
    
    if "main" not in pipelines.keys():
        raise RuntimeError(f"'main' is not in pipelines")
    
    return pipelines
    

class FakeTextDataGenerator(object):
    @classmethod
    def generate_from_tuple(cls, t):
        """
        Same as generate, but takes all parameters as one tuple
        """

        cls.generate(*t)

    @classmethod
    def generate(
        cls,
        index: int,
        text: str,
        font: str,
        out_dir: str,
        size: int,
        extension: str,
        skewing_angle: int,
        random_skew: bool,
        blur: int,
        random_blur: bool,
        background_type: int,
        distorsion_type: int,
        distorsion_orientation: int,
        is_handwritten: bool,
        name_format: int,
        width: int,
        alignment: int,
        text_color: str,
        orientation: int,
        space_width: int,
        character_spacing: int,
        margins: int,
        fit: bool,
        output_mask: bool,
        word_split: bool,
        image_dir: str,
        stroke_width: int = 0,
        stroke_fill: str = "#282828",
        image_mode: str = "RGB",
        output_bboxes: int = 0,
        dummy: int = 0,
        settings: dict = {},
    ) -> Image:
        image = None

        if len(settings) == 0:
            raise RuntimeError("Aug opts are empty")
        
        margin_top, margin_left, margin_bottom, margin_right = margins
        horizontal_margin = margin_left + margin_right
        vertical_margin = margin_top + margin_bottom
        
        aug_pipelines = build_augmentation_pipeline(settings)
        
        ##########################
        # Create picture of text #
        ##########################
        text_image = computer_text_generator.generate_horizontal_text1(
            text=text,
            font=font,
            text_color=text_color,
            font_size=size,
            space_width=space_width,
            character_spacing=character_spacing,
            word_split=word_split,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill
        )
        # if len(text) > 50:
        #     random_angle = rnd.randint(0 - skewing_angle // 3, skewing_angle // 3)
        # elif len(text) > 32:
        #     random_angle = rnd.randint(0 - skewing_angle // 2, skewing_angle // 2)
        # else:
        #     random_angle = rnd.randint(0 - skewing_angle, skewing_angle)

        # rotated_img = image.rotate(
        #     skewing_angle if not random_skew else random_angle, expand=1
        # )

        # rotated_mask = mask.rotate(
        #     skewing_angle if not random_skew else random_angle, expand=1
        # )
        
        if "text" in aug_pipelines:
            text_image = aug_pipelines["text"].apply(text_image)

        ##################################
        # Resize image to desired format #
        ##################################

        # Horizontal text
        new_width = int(text_image.size[0] * (float(size - vertical_margin) / float(text_image.size[1])))
        text_image = text_image.resize(
            (new_width, size - vertical_margin), Image.Resampling.LANCZOS
        )
        # resized_mask = distorted_mask.resize(
        #     (new_width, size - vertical_margin), Image.Resampling.NEAREST
        # )
        background_width = width if width > 0 else new_width + horizontal_margin
        background_height = size
   

        #############################
        # Generate background image #
        #############################
        def generate_backgound():
            #c = rnd.choices(population=[0,1,2], weights=[0.45, 0.10, 0.45], k=1)[0]
            #c = rnd.randint(0, 7)
            w = settings.inp_weights
            type = rnd.choices(population=["image", "noise", "plain"], weights=w, k=1)[0]
            
            c = -1
            match type:
                case "image":
                    c = 3
                case "noise":
                    c = rnd.choice([0, 4, 5, 6, 7])
                case "plain":
                    c = 1                    
                    
            #c = 3
            match c:
                case 0:
                    return background_generator.gaussian_noise(
                        background_height, background_width
                    )
                case 1:
                    return background_generator.plain_color(
                        background_height, background_width
                    )
                case 2:
                    return background_generator.quasicrystal(
                        background_height, background_width
                    )
                case 3:                
                    return background_generator.image(
                        background_height, background_width, image_dir
                    )
                case 4:
                    return background_generator.my_noise(background_height, background_width, MyNoise_CLUSTER(12))
                case 5:
                    return background_generator.my_noise(background_height, background_width, MyNoise_INTER(12))
                case 6:
                    return background_generator.my_noise(background_height, background_width, MyNoise_CLOUD(8, 8))
                case 7:
                    return background_generator.my_noise(background_height, background_width, MyNoise_MARBLE())
                case _:
                    raise RuntimeError("Unknown background type")


        background_img = generate_backgound()

        ##############################################################
        # Comparing average pixel value of text and background image #
        ##############################################################
        # if 0:
        #     try:
        #         resized_img_st = ImageStat.Stat(resized_img, resized_mask.split()[2])
        #         background_img_st = ImageStat.Stat(background_img)

        #         resized_img_px_mean = sum(resized_img_st.mean[:3]) / 3
        #         background_img_px_mean = sum(background_img_st.mean[:3]) / 3
        #         df = abs(resized_img_px_mean - background_img_px_mean)

        #         debug(f"Avg: bg:{background_img_px_mean} font: {resized_img_px_mean} df={df} angle={random_angle} blur={text_blur_fact}")
        #         if df < 15:
        #             debug("value of mean pixel is too similar. Ignore this image")

        #             debug("resized_img_st \n {}".format(resized_img_st.mean))
        #             debug("background_img_st \n {}".format(background_img_st.mean))

        #             return
        #     except Exception as err:
        #         #error(f"Exception: {str(err)}")
        #         raise err

        #############################
        # Place text with alignment #
        #############################

        new_text_width, _ = text_image.size
        
        background_img = background_img.convert('L')
        text_image = text_image.convert('LA')

        text_mask = np.asarray(text_image).copy()
        blend_factor = randfloat(settings.blend_factor_min, settings.blend_factor_max)
        text_mask[:,:, 1] = text_mask[:,:,1] * blend_factor
        text_mask = Image.fromarray(text_mask)
        background_img.paste(text_image, (margin_left, margin_top), text_mask)
        # if alignment == 0 or width == -1:
        #     background_img.paste(text_image, (margin_left, margin_top), text_image)
        # elif alignment == 1:
        #     background_img.paste(
        #         text_image,
        #         (int(background_width / 2 - new_text_width / 2), margin_top),
        #         text_image,
        #     )
        # else:
        #     background_img.paste(
        #         text_image,
        #         (background_width - new_text_width - margin_right, margin_top),
        #         text_image,
        #     )

        ############################################
        # Change image mode (RGB, grayscale, etc.) #
        ############################################

        #background_img = background_img.convert(image_mode)
        
        final_image = background_img  
            
        def invert_image(img: Image):
            return Image.fromarray(cv2.bitwise_not(np.asarray(img)))
             
        #######################
        # Apply augmentations #
        #######################

        final_image = aug_pipelines["main"].apply(final_image)            
                        
        if rnd.random() < settings.invert_chance:
            final_image = invert_image(final_image)                                
            
        if rnd.random() < settings.overexposure_chance:
            n = rnd.randint(1, settings.overexposure_k)
            for i in range(n):
                final_image = Image.fromarray(apply_random_overexposure(np.asarray(final_image)))


        if output_mask == 1:
            return final_image, None
        return final_image
