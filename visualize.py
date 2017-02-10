import os
import random

import PIL.Image, PIL.ImageOps
import leveldb
import caffe

import lmdb
import numpy

from util.parse_data_files import read_image_file_header, read_label_file_header
from util.paths import data_path, temp_path, mkdir_recursive


def convert_to_image_files(image_batch_file, label_batch_file, skip_first_n_images=0, save_n_images=100, step=1):
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
        # image = PIL.ImageOps.invert(image)
        if i % step == 0:
            image.save(
                open(temp_path("vis", "image{}-label{}.png".format(i, label)), "wb"),
                "png"
            )


def average_data_set(image_batch_file, sample_approx_n_images):
    image_batch_file = open(data_path(image_batch_file), "rb")
    image_file_header = read_image_file_header(image_batch_file)
    single_image_size = image_file_header["num_rows"] * image_file_header["num_columns"]

    total_images = image_file_header["num_items"]

    keep_chance = max(min(float(sample_approx_n_images) / float(total_images), 1.0), 0.0)

    images_to_avg = []

    for i in xrange(total_images):
        image_data = image_batch_file.read(single_image_size)

        if random.random() <= keep_chance:
            image = PIL.Image.frombytes("L", (image_file_header["num_rows"], image_file_header["num_columns"]), image_data)
            images_to_avg.append(image)

    avg = numpy.zeros((images_to_avg[0].size[0], images_to_avg[0].size[1]), numpy.float)
    n = len(images_to_avg)

    for image in images_to_avg:
        imarr = numpy.array(image, dtype=numpy.float)
        avg += imarr / n

    return numpy.average(avg)


def create_lmdb(lmdb_filename, image_batch_file, label_batch_file):
    image_batch_file = open(data_path(image_batch_file), "rb")
    image_file_header = read_image_file_header(image_batch_file)
    single_image_size = image_file_header["num_rows"] * image_file_header["num_columns"]

    label_batch_file = open(data_path(label_batch_file), "rb")
    label_file_header = read_label_file_header(label_batch_file)

    num_images = image_file_header["num_items"]

    # We need to prepare the database for the size. We'll set it 10 times
    # greater than what we theoretically need. There is little drawback to
    # setting this too big. If you still run into problem after raising
    # this, you might want to try saving fewer entries in a single
    # transaction.
    map_size = single_image_size * num_images * 10

    env = lmdb.open(data_path(lmdb_filename), map_size=map_size)

    size_x = image_file_header["num_columns"]
    size_y = image_file_header["num_rows"]

    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i in xrange(num_images):
            print (i + 1), "/", num_images
            image_data = image_batch_file.read(single_image_size)
            label = ord(label_batch_file.read(1))

            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = 1
            datum.height = size_y
            datum.width = size_x
            datum.data = image_data
            datum.label = label
            str_id = '{:08}'.format(i)

            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())


if __name__ == "__main__":
    convert_to_image_files("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", 0, 10000, 100)
    # print average_data_set("train-images-idx3-ubyte", 1000)
    # create_lmdb("train.lmdb", "train-images-idx3-ubyte", "train-labels-idx1-ubyte")