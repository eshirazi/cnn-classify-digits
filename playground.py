import json

from util.parse_data_files import read_image_file_header, read_label_file_header
from util.paths import data_path


def playground():
    image_file = open(data_path("t10k-images-idx3-ubyte"), "rb")
    image_file_header = read_image_file_header(image_file)
    print json.dumps(image_file_header, indent=4)

    labels_file = open(data_path("t10k-images-idx3-ubyte"), "rb")
    labels_file_header = read_label_file_header(labels_file)
    print json.dumps(labels_file_header, indent=4)


if __name__ == "__main__":
    playground()