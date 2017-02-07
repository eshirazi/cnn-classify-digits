import struct


def read_int32(fp):
    return struct.unpack(">i", fp.read(4))[0]


def read_image_file_header(fp):
    return {
        "magic": read_int32(fp),
        "num_items": read_int32(fp),
        "num_rows": read_int32(fp),
        "num_columns": read_int32(fp),
    }


def read_label_file_header(fp):
    return {
        "magic": read_int32(fp),
        "num_items": read_int32(fp),
    }

