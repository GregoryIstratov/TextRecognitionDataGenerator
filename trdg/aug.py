import io
import numpy as np
import random as rnd
import math
from PIL import Image, ImageFilter, ImageStat, ImageEnhance
import cv2
from trdg.utils import AttrDict
from trdg import distorsion_generator
import graphlib

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
    
    if not isinstance(opt_augs, AttrDict):
        opt_augs = AttrDict(opt_augs)
        
    opt_augs = AttrDict(opt_augs)
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
    