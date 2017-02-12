import glob
import os
import re
from collections import OrderedDict

import PIL.Image
from torch import nn, torch, optim
from torch.autograd import Variable
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from util.paths import data_path

DEBUG_CHECKS = False

BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
EPOCHS = 3
#WEIGHT_DECAY = 0.005
BASE_LEARNING_RATE = 0.01
LEARNING_RATE_DECAY_PER_EPOCH = 0.01
LEARNING_MOMENTUM = 0.5
LOG_INTERVAL = 10

SCRIPT_PATH = os.path.dirname(__file__)
DATA_PATH = data_path()
MODEL_FILE = os.path.join(SCRIPT_PATH, "net.pt")


class SimpleConvNet(nn.Module):
    def __init__(self, debug=False):
        super(SimpleConvNet, self).__init__()
        self.debug = debug
        self.layers = OrderedDict([
            ("conv1", nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)),
            ("conv1-relu", nn.ReLU()),
            ("conv2", nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3)),
            ("conv2-relu", nn.ReLU()),
            ("pool1", nn.MaxPool2d(kernel_size=2)),
            ("conv3", nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)),
            ("conv3-relu", nn.ReLU()),
            ("conv4", nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)),
            ("conv4-relu", nn.ReLU()),
            ("view", lambda x: x.view(-1, 2048)),
            ("fc1", nn.Linear(in_features=2048, out_features=2048)),
            ("fc1-relu", nn.ReLU()),
            ("fc2", nn.Linear(in_features=2048, out_features=10)),
        ])

        for layer_name, layer in self.layers.iteritems():
            if isinstance(layer, nn.Module):
                self.add_module(layer_name, layer)

    def forward(self, x):
        if self.debug:
            print "--->", x.size()

        for layer_name, layer in self.layers.iteritems():
            x = layer(x)
            if self.debug:
                print "--->", layer_name
                print "--->", " x ".join(map(str, x.size()))

        return x


# class SimpleConvNet(nn.Module):
#     def __init__(self):
#         super(SimpleConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = F.relu(self.fc2(x))
#         return x #F.log_softmax(x)


def calculate_dataset_mean_and_std(dataset):
    as_huge_array = next(iter(DataLoader(dataset, batch_size=len(dataset))))[0]
    return as_huge_array.mean(), as_huge_array.std()


def load_normalized_datasets():
    train_dataset = datasets.MNIST(DATA_PATH, train=True, download=True, transform=transforms.ToTensor())

    mean, std = calculate_dataset_mean_and_std(train_dataset)

    print "Normalizing dataset with: mean={}, stdev={}".format(mean, std)

    train_dataset.transform = transforms.Compose([
        train_dataset.transform, transforms.Normalize([mean], [std])
    ])

    if DEBUG_CHECKS:
        print "After normalization: mean={}, stdev={}".format(*calculate_dataset_mean_and_std(train_dataset))

    test_dataset = datasets.MNIST(DATA_PATH, train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([mean], [std])
    ]))

    if DEBUG_CHECKS:
        print "Test data set with normalization: mean={}, stdev={}".format(*calculate_dataset_mean_and_std(test_dataset))

    return train_dataset, test_dataset


def train_step(net, train_dataset_loader, epoch):
    # set net to train mode
    net.train()

    learning_rate = BASE_LEARNING_RATE * LEARNING_RATE_DECAY_PER_EPOCH ** epoch
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=LEARNING_MOMENTUM)

    for i, (inputs, labels) in enumerate(train_dataset_loader):
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        output = net.forward(inputs)
        loss = CrossEntropyLoss().forward(output, labels)
        loss.backward()
        optimizer.step()

        if i % LOG_INTERVAL == 0:
            print('Train Epoch (LR: {}): {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                learning_rate,
                epoch, i * len(inputs), len(train_dataset_loader.dataset),
                100. * i / len(train_dataset_loader), loss.data[0]))


def test_training_accuracy(net, test_dataset_loader, epoch):
    # set net to test mode
    net.eval()

    test_loss = 0
    correct = 0
    for data, target in test_dataset_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = net.forward(data)
        test_loss += CrossEntropyLoss()(output, target).data[0]
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_dataset_loader)  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_dataset_loader.dataset),
        100. * correct / len(test_dataset_loader.dataset)))


def train():
    train_dataset, test_dataset = load_normalized_datasets()

    net = SimpleConvNet() #debug=True)

    train_dataset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataset_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)

    for epoch in xrange(0, EPOCHS):
        train_step(net, train_dataset_loader, epoch)
        test_training_accuracy(net, test_dataset_loader, epoch)
        torch.save(net.state_dict(), open(MODEL_FILE, "wb"))


def show_saved_net_accuracy():
    train_dataset, test_dataset = load_normalized_datasets()
    test_dataset_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)

    net = SimpleConvNet()
    net.forward(Variable(torch.FloatTensor(1, 1, 28, 28)))
    net.load_state_dict(torch.load(open(MODEL_FILE, "rb")))
    test_training_accuracy(net, test_dataset_loader, 0)


def test_on_image_directory(path):
    net = SimpleConvNet()
    net.forward(Variable(torch.FloatTensor(1, 1, 28, 28)))
    net.load_state_dict(torch.load(open(MODEL_FILE, "rb")))

    label_regex = re.compile("([0123456789]+)\\..*?")
    correct = 0
    total = 0

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    for file_path in glob.glob(os.path.join(path, "*.png")):
        cur_correct = int(label_regex.findall(os.path.basename(file_path))[0])
        image = PIL.Image.open(file_path).convert("L")
        transformed_image = transform(image)
        transformed_image = Variable(transformed_image.view(1, 1, 28, 28))

        cur_predicted = net.forward(transformed_image).data.max(1)[1][0][0]

        print os.path.basename(file_path) + " - " + str(cur_predicted)

        total += 1
        if cur_correct == cur_predicted:
            correct += 1

    print "Identified {} of {}, {:.2%}".format(correct, total, float(correct) / float(total))


def main():
    # train()
    show_saved_net_accuracy()
    test_on_image_directory(data_path("my_samples"))

if __name__ == "__main__":
    main()

