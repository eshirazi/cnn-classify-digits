import os

import PIL.Image, PIL.ImageOps

from util.parse_data_files import read_image_file_header, read_label_file_header
from util.paths import data_path, temp_path, mkdir_recursive


def convert_to_image_files(image_batch_file, label_batch_file, skip_first_n_images=0, save_n_images=100):
    image_batch_file = open(data_path(image_batch_file), "rb")
    image_file_header = read_image_file_header(image_batch_file)
    single_image_size = image_file_header["num_rows"] * image_file_header["num_columns"]

    label_batch_file = open(data_path(label_batch_file), "rb")
    label_file_header = read_label_file_header(label_batch_file)

    image_batch_file.seek(skip_first_n_images * single_image_size, 1)
    label_batch_file.seek(skip_first_n_images, 1)

    mkdir_recursive(temp_path("vis"))

    for i in xrange(save_n_images):
        image_data = image_batch_file.read(single_image_size)
        label = ord(label_batch_file.read(1))

        image = PIL.Image.frombytes("L", (image_file_header["num_rows"], image_file_header["num_columns"]), image_data)
        image = PIL.ImageOps.invert(image)
        image.save(
            open(temp_path("vis", "image{}-label{}.png".format(i, label)), "wb"),
            "png"
        )


if __name__ == "__main__":
    convert_to_image_files("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")