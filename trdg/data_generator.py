import io
import os
import random as rnd
import traceback

from PIL import Image, ImageFilter, ImageStat

from trdg import computer_text_generator, background_generator, distorsion_generator
from trdg.utils import mask_to_bboxes, make_filename_valid, add_image_noise, apply_random_overexposure, debug, error, AttrDict
from trdg.background_generator import MyNoise_INTER, MyNoise_CLUSTER, MyNoise_CLOUD, MyNoise_MARBLE

import numpy as np
import cv2
import random as rnd
from enum import Enum

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

try:
    from trdg import handwritten_text_generator
except ImportError as e:
    print("Missing modules for handwritten text generation.")


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
        aug_opts: dict = {}
    ) -> Image:
        image = None

        if len(aug_opts) == 0:
            raise RuntimeError("Aug opts are empty")
        
        aug_opts = AttrDict(aug_opts)
        margin_top, margin_left, margin_bottom, margin_right = margins
        horizontal_margin = margin_left + margin_right
        vertical_margin = margin_top + margin_bottom
        
        def get_enabled_augs(aug_opts: dict) -> list[DistortType]:
            def str2aug(s: str):
                match s.lower():
                    case "text_blur":
                        return DistortType.TEXT_BLUR
                    case "text_noise":
                        return DistortType.TEXT_NOISE
                    case "blur":
                        return DistortType.BLUR
                    case "motion_blur":
                        return DistortType.MOTION_BLUR
                    case "sharpen":
                        return DistortType.SHARPEN
                    case "downsample":
                        return DistortType.DOWNSAMPLE
                    case "jpeg_artifact":
                        return DistortType.JPEG_ARTIFACT
                    case "sin_cos":
                        return DistortType.SIN_COS
                    case "noise":
                        return DistortType.NOISE
                    case "none":
                        return DistortType.NONE
                    case _:
                        raise RuntimeError(f"Unknown augmentation type '{s}'")
                    
            ss: list[str] = aug_opts.allow_list
            
            if "all" in ss:
                return list(DistortType)
            
            augs = [str2aug(s) for s in ss]
            return augs
                
        def gen_augs(allowed_augs, k):
            augs = rnd.sample(population=allowed_augs, k=k)
            return augs
        
        def filter_blurs(augs):
            blurs = set([DistortType.BLUR, DistortType.MOTION_BLUR, DistortType.TEXT_BLUR])
            d = set(augs) & blurs
            if len(d) >= 2:
                return filter_blurs(gen_augs(get_enabled_augs(aug_opts), aug_opts.k))
            else:
                return augs

        augs = filter_blurs(gen_augs(get_enabled_augs(aug_opts), aug_opts.k))

        
        debug(f"Augs={augs}")
        
        ##########################
        # Create picture of text #
        ##########################
        image, mask = computer_text_generator.generate(
            text,
            font,
            text_color,
            size,
            orientation,
            space_width,
            character_spacing,
            fit,
            word_split,
            stroke_width,
            stroke_fill,
        )
        if len(text) > 50:
            random_angle = rnd.randint(0 - skewing_angle // 3, skewing_angle // 3)
        elif len(text) > 32:
            random_angle = rnd.randint(0 - skewing_angle // 2, skewing_angle // 2)
        else:
            random_angle = rnd.randint(0 - skewing_angle, skewing_angle)

        rotated_img = image.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )

        rotated_mask = mask.rotate(
            skewing_angle if not random_skew else random_angle, expand=1
        )

        #############################
        # Apply distorsion to image #
        #############################
        distorsion_type = 0
        if DistortType.SIN_COS in augs:
            distorsion_type = rnd.randint(1, 3)
        
        if distorsion_type == 0:
            distorted_img = rotated_img  # Mind = blown
            distorted_mask = rotated_mask
        elif distorsion_type == 1:
            distorted_img, distorted_mask = distorsion_generator.sin(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        elif distorsion_type == 2:
            distorted_img, distorted_mask = distorsion_generator.cos(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )
        else:
            distorted_img, distorted_mask = distorsion_generator.random(
                rotated_img,
                rotated_mask,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2),
            )

        ##################################
        # Resize image to desired format #
        ##################################

        # Horizontal text
        if orientation == 0:
            new_width = int(
                distorted_img.size[0]
                * (float(size - vertical_margin) / float(distorted_img.size[1]))
            )
            resized_img = distorted_img.resize(
                (new_width, size - vertical_margin), Image.Resampling.LANCZOS
            )
            resized_mask = distorted_mask.resize(
                (new_width, size - vertical_margin), Image.Resampling.NEAREST
            )
            background_width = width if width > 0 else new_width + horizontal_margin
            background_height = size
        # Vertical text
        elif orientation == 1:
            new_height = int(
                float(distorted_img.size[1])
                * (float(size - horizontal_margin) / float(distorted_img.size[0]))
            )
            resized_img = distorted_img.resize(
                (size - horizontal_margin, new_height), Image.Resampling.LANCZOS
            )
            resized_mask = distorted_mask.resize(
                (size - horizontal_margin, new_height), Image.Resampling.NEAREST
            )
            background_width = size
            background_height = new_height + vertical_margin
        else:
            raise ValueError("Invalid orientation")
        
        ################################
        # Apply noise over text image  #
        ################################
        
        text_blur_fact = 0
        if DistortType.TEXT_BLUR in augs:
            text_blur_fact = aug_opts.text_blur if not aug_opts.text_blur_rnd else rnd.random() * aug_opts.text_blur
            #text_blur_fact = blur if not random_blur else rnd.random() * blur
            gaussian_filter = ImageFilter.GaussianBlur(
                radius=text_blur_fact
            )
            resized_img = resized_img.filter(gaussian_filter)
            
        if DistortType.TEXT_NOISE in augs:
            resized_img = add_image_noise(resized_img)
        

        #############################
        # Generate background image #
        #############################
        def generate_backgound():
            #c = rnd.choices(population=[0,1,2], weights=[0.45, 0.10, 0.45], k=1)[0]
            #c = rnd.randint(0, 7)
            w = aug_opts.inp_weights
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

        background_mask = Image.new(
            "RGB", (background_width, background_height), (0, 0, 0)
        )

        ##############################################################
        # Comparing average pixel value of text and background image #
        ##############################################################
        if 0:
            try:
                resized_img_st = ImageStat.Stat(resized_img, resized_mask.split()[2])
                background_img_st = ImageStat.Stat(background_img)

                resized_img_px_mean = sum(resized_img_st.mean[:3]) / 3
                background_img_px_mean = sum(background_img_st.mean[:3]) / 3
                df = abs(resized_img_px_mean - background_img_px_mean)

                debug(f"Avg: bg:{background_img_px_mean} font: {resized_img_px_mean} df={df} angle={random_angle} blur={text_blur_fact}")
                if df < 15:
                    debug("value of mean pixel is too similar. Ignore this image")

                    debug("resized_img_st \n {}".format(resized_img_st.mean))
                    debug("background_img_st \n {}".format(background_img_st.mean))

                    return
            except Exception as err:
                #error(f"Exception: {str(err)}")
                raise err

        #############################
        # Place text with alignment #
        #############################

        new_text_width, _ = resized_img.size

        if alignment == 0 or width == -1:
            background_img.paste(resized_img, (margin_left, margin_top), resized_img)
            background_mask.paste(resized_mask, (margin_left, margin_top))
        elif alignment == 1:
            background_img.paste(
                resized_img,
                (int(background_width / 2 - new_text_width / 2), margin_top),
                resized_img,
            )
            background_mask.paste(
                resized_mask,
                (int(background_width / 2 - new_text_width / 2), margin_top),
            )
        else:
            background_img.paste(
                resized_img,
                (background_width - new_text_width - margin_right, margin_top),
                resized_img,
            )
            background_mask.paste(
                resized_mask,
                (background_width - new_text_width - margin_right, margin_top),
            )

        ############################################
        # Change image mode (RGB, grayscale, etc.) #
        ############################################

        background_img = background_img.convert(image_mode)
        background_mask = background_mask.convert(image_mode)
        
        final_image = background_img
        final_mask = background_mask        
            
        def downsample(img: Image):
            #dfactor = random.choice([2,3,4])
            dfactor = aug_opts.downsample_factor
            img_np = np.asarray(img)
            img_ds = cv2.resize(img_np, dsize=(img_np.shape[1] // dfactor, img_np.shape[0] // dfactor), interpolation=cv2.INTER_NEAREST)
            img_np = cv2.resize(img_ds, dsize=(img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            return Image.fromarray(img_np)
            
        def invert_image(img: Image):
            return Image.fromarray(cv2.bitwise_not(np.asarray(img)))
        
        #size - in pixels, size of motion blur
        #angel - in degrees, direction of motion blur
        def apply_motion_blur(image: Image, size, angle):
            img = np.asarray(image)
            k = np.zeros((size, size), dtype=np.float32)
            k[ (size-1)// 2 , :] = np.ones(size, dtype=np.float32)
            k = cv2.warpAffine(k, cv2.getRotationMatrix2D( (size / 2 -0.5 , size / 2 -0.5 ) , angle, 1.0), (size, size) )  
            k = k * ( 1.0 / np.sum(k) )        
            return Image.fromarray(cv2.filter2D(img, -1, k))
        
        def apply_sharpen(image: Image):
            img = np.asarray(image)
            # Create the sharpening kernel
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            # Apply the sharpening kernel to the image using filter2D
            sharpened = cv2.filter2D(img, -1, kernel)
            return Image.fromarray(sharpened)
        
        def add_jpeg_artifact(img: Image):
            with io.BytesIO() as buff:
                #img.save(buff, format='JPEG', quality=random.randint(15, 35))
                img.save(buff, format='JPEG', quality=aug_opts.jpeg_artifact_q)
                img = Image.open(buff, formats=["JPEG"]).copy()
                return img                
            
        #######################
        # Apply augmentations #
        #######################

        if DistortType.BLUR in augs:
            blur_fact = aug_opts.blur if not aug_opts.blur_rnd else (0.25 + rnd.random() * 0.75) * aug_opts.blur
            gaussian_filter = ImageFilter.GaussianBlur(
                radius=blur_fact
            )
            final_image = final_image.filter(gaussian_filter)
            final_mask = final_mask.filter(gaussian_filter)             
                    
        if DistortType.MOTION_BLUR in augs:
            mn = aug_opts.motion_blur_sz_min
            mx = aug_opts.motion_blur_sz_max
            mb_sz = rnd.randint(mn, mx)
            mb_angle = rnd.randint(0, 360)
            final_image = apply_motion_blur(final_image, mb_sz, mb_angle)
            
        if DistortType.JPEG_ARTIFACT in augs:
            final_image = add_jpeg_artifact(final_image)
            
        if DistortType.DOWNSAMPLE in augs:
            final_image = downsample(final_image)
            
        if DistortType.SHARPEN in augs:
            final_image = apply_sharpen(final_image)
            
        if DistortType.NOISE in augs:
            final_image = add_image_noise(final_image, mean=aug_opts.noise_mean, stddev=aug_opts.noise_stddev)                 
                        
        if rnd.random() < aug_opts.invert_chance:
            final_image = invert_image(final_image)                                
            
        if rnd.random() < aug_opts.overexposure_chance:
            n = rnd.randint(1, aug_opts.overexposure_k)
            for i in range(n):
                final_image = Image.fromarray(apply_random_overexposure(np.asarray(final_image)))

        #####################################
        # Generate name for resulting image #
        #####################################
        # We remove spaces if space_width == 0
        if space_width == 0:
            text = text.replace(" ", "")
        if name_format == 0:
            name = "{}_{}".format(text, str(index))
        elif name_format == 1:
            name = "{}_{}".format(str(index), text)
        elif name_format == 2:
            name = str(index)
        else:
            print("{} is not a valid name format. Using default.".format(name_format))
            name = "{}_{}".format(text, str(index))

        name = make_filename_valid(name, allow_unicode=True)
        image_name = "{}.{}".format(name, extension)
        mask_name = "{}_mask.png".format(name)
        box_name = "{}_boxes.txt".format(name)
        tess_box_name = "{}.box".format(name)

        # Save the image
        if out_dir is not None:
            final_image.save(os.path.join(out_dir, image_name))
            if output_mask == 1:
                final_mask.save(os.path.join(out_dir, mask_name))
            if output_bboxes == 1:
                bboxes = mask_to_bboxes(final_mask)
                with open(os.path.join(out_dir, box_name), "w") as f:
                    for bbox in bboxes:
                        f.write(" ".join([str(v) for v in bbox]) + "\n")
            if output_bboxes == 2:
                bboxes = mask_to_bboxes(final_mask, tess=True)
                with open(os.path.join(out_dir, tess_box_name), "w") as f:
                    for bbox, char in zip(bboxes, text):
                        f.write(
                            " ".join([char] + [str(v) for v in bbox] + ["0"]) + "\n"
                        )
        else:
            if output_mask == 1:
                return final_image, final_mask
            return final_image
