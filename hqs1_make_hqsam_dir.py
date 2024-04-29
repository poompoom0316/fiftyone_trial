import numpy as np
from PIL import Image
import os
import shutil


def prepare_training_folder():
    # for prefix in ["train", "val", "test"]:
    for prefix in ["train"]:
        input_image_folder = f"data/PhenoBench-v100/PhenoBench/{prefix}/images"
        input_annotation_folder = f"data/PhenoBench-v100/PhenoBench/{prefix}/plant_instances"
        output_folder_root = "analysis/hqsam_processed"
        output_folder = f"{output_folder_root}/{prefix}"
        output_folder_plant = f"{output_folder}/plant"

        os.makedirs(output_folder_root, exist_ok=True)
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(output_folder_plant, exist_ok=True)

        # convert png file in input_image_folder to jpg and save to output_folder_plant
        image_list = os.listdir(input_image_folder)
        # for image_name in image_list:
        #     im = Image.open(f"{input_image_folder}/{image_name}")
        #     im.save(f"{output_folder_plant}/{image_name.replace('.png', '.jpg')}")

        # copy png file in input_annotation_folder
        annotation_list = os.listdir(input_annotation_folder)
        for annotation_name in annotation_list:
            im = Image.open(f"{input_annotation_folder}/{annotation_name}")
            im_array = np.array(im)
            im_array[im_array > 0] = 255
            im_converted = Image.fromarray(im_array.astype(np.uint8))
            im_converted.save(f"{output_folder_plant}/{annotation_name}")


if __name__ == '__main__':
    prepare_training_folder()