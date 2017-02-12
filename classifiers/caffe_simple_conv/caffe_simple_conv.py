import glob
import random
import re

import numpy as np
import os
import caffe

from util.paths import temp_path, data_path

SCRIPT_PATH = os.path.dirname(__file__)

caffe.set_mode_cpu()

model_def = os.path.join(SCRIPT_PATH, 'train_val.prototxt')
model_weights = os.path.join(SCRIPT_PATH, 'simple_train_iter_1000.caffemodel')

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
# transformer.set_channel_swap('data', (2, 1, 0))
# transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
# transformer.set_mean('data', np.array([33]))            # subtract the dataset-mean value in each channel
# transformer.set_mean('data', np.array([-330]))            # subtract the dataset-mean value in each channel

LABEL_REGEX = re.compile("([0123456789]+)\\..*?")

def test_path(path):
    correct = 0
    total = 0

    for file_path in glob.glob(os.path.join(path, "*.png")):
        image = caffe.io.load_image(file_path, color=False)
        cur_correct = int(LABEL_REGEX.findall(os.path.basename(file_path))[0])

        transformed_image = transformer.preprocess('data', image)
        # print image.shape, transformed_image.shape

        # print transformed_image
        # break

        # copy the image data into the memory allocated for the net
        net.blobs['data'].data[...] = transformed_image

        ### perform classification
        output = net.forward()

        # print output
        cur_predicted = output["fc8"][0].argmax()

        print os.path.basename(file_path) + " - " + str(cur_predicted)

        total += 1
        if cur_correct == cur_predicted:
            correct += 1
            # output_prob = output['fc8'][0]  # the output probability vector for the first image in the batch
        # print output
        # print 'predicted class is:', output_prob.argmax() + 1

    print "Identified {} of {}, {:.2%}".format(correct, total, float(correct) / float(total))

if __name__ == "__main__":
    # test_path(temp_path("vis"))
    test_path(os.path.join(data_path("my_samples")))
    # print LABEL_REGEX.findall("/Users/eranshirazi/Home/Dev/cnn-classify-digits/util/../temp/vis/image8900-label9.png")