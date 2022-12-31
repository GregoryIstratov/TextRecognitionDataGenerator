import os
import random as rnd

from PIL import Image, ImageFilter, ImageStat

from trdg import computer_text_generator, background_generator, distorsion_generator
from trdg.utils import mask_to_bboxes, make_filename_valid, add_image_noise
from trdg.background_generator import MyNoise_INTER, MyNoise_CLUSTER, MyNoise_CLOUD

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
    ) -> Image:
        image = None

        margin_top, margin_left, margin_bottom, margin_right = margins
        horizontal_margin = margin_left + margin_right
        vertical_margin = margin_top + margin_bottom

        ##########################
        # Create picture of text #
        ##########################
        if is_handwritten:
            if orientation == 1:
                raise ValueError("Vertical handwritten text is unavailable")
            image, mask = handwritten_text_generator.generate(text, text_color)
        else:
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
        if distorsion_type > 0:
            distorsion_type = rnd.randint(0, 2)
        
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
        
        
        #apply_text_noise = rnd.choice([True, False])
        apply_text_noise = False
        
        ################################
        # Apply noise over text image  #
        ################################
        
        if apply_text_noise:
            c = rnd.randint(0, 2)
            match c:
                case 0:
                    resized_img = add_image_noise(resized_img)
                case 1:
                    gaussian_filter = ImageFilter.GaussianBlur(
                        radius=blur if not random_blur else rnd.random() * blur
                    )
                    resized_img = resized_img.filter(gaussian_filter)
                case 2:
                    pass

        #############################
        # Generate background image #
        #############################
        def generate_backgound():
            #c = rnd.choices(population=[0,1,2], weights=[0.45, 0.10, 0.45], k=1)[0]
            c = rnd.randint(0, 6)
            match c:
                case 0:
                    return background_generator.gaussian_noise(
                        background_height, background_width
                    )
                case 1:
                    return background_generator.plain_white(
                        background_height, background_width
                    )
                case 2:
                    return background_generator.quasicrystal(
                        background_height, background_width
                    )
                case 3:
                    return background_generator.quasicrystal(
                        background_height, background_width
                    )                    
                    # return background_generator.image(
                    #     background_height, background_width, image_dir
                    # )
                case 4:
                    return background_generator.my_noise(background_height, background_width, MyNoise_CLUSTER(12))
                case 5:
                    return background_generator.my_noise(background_height, background_width, MyNoise_INTER(12))
                case 6:
                    return background_generator.my_noise(background_height, background_width, MyNoise_CLOUD(8, 8))
                case _:
                    raise RuntimeError("Unknown background type")


        background_img = generate_backgound()

        background_mask = Image.new(
            "RGB", (background_width, background_height), (0, 0, 0)
        )

        ##############################################################
        # Comparing average pixel value of text and background image #
        ##############################################################
        try:
            resized_img_st = ImageStat.Stat(resized_img, resized_mask.split()[2])
            background_img_st = ImageStat.Stat(background_img)

            resized_img_px_mean = sum(resized_img_st.mean[:3]) / 3
            background_img_px_mean = sum(background_img_st.mean[:3]) / 3
            df = abs(resized_img_px_mean - background_img_px_mean)

            print(f"Avg: bg:{background_img_px_mean} font: {resized_img_px_mean} df={df}")
            if df < 50:
                print("value of mean pixel is too similar. Ignore this image")

                print("resized_img_st \n {}".format(resized_img_st.mean))
                print("background_img_st \n {}".format(background_img_st.mean))

                return
        except Exception as err:
            return

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

        #######################
        # Apply gaussian blur #
        #######################

        if not apply_text_noise:
            gaussian_filter = ImageFilter.GaussianBlur(
                radius=blur if not random_blur else rnd.random() * blur
            )
            final_image = background_img.filter(gaussian_filter)
            final_mask = background_mask.filter(gaussian_filter)
        else:
            final_image = background_img
            final_mask = background_mask

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
