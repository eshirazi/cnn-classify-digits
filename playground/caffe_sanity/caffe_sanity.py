import numpy as np
import os
import caffe

SCRIPT_PATH = os.path.dirname(__file__)
TEST_IMAGE_PATH = os.path.join(SCRIPT_PATH, "img", "hammer.jpg")
SYNSET_WORDS = os.path.join(SCRIPT_PATH, "synset_words.txt")

caffe.set_mode_cpu()

caffe_root = os.path.join(os.path.dirname(caffe.__file__), "..", "..") + "/"

model_def = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

image = caffe.io.load_image(TEST_IMAGE_PATH)
transformed_image = transformer.preprocess('data', image)

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image

### perform classification
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print 'predicted class is:', output_prob.argmax()

# load ImageNet labels

labels = np.loadtxt(SYNSET_WORDS, str, delimiter='\t')

print 'output label:', labels[output_prob.argmax()]
