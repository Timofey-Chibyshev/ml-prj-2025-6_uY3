import numpy as np
import struct
from array import array

# https://www.kaggle.com/code/hojjatk/read-mnist-dataset
class MnistDataLoader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels

    def load_train_data(self):
        return self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
    
    def load_test_data(self):
        return self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)

    def load_data(self):
        return self.load_train_data(), self.load_test_data() 

loader = MnistDataLoader(
    "dataset/train-images.idx3-ubyte",
    "dataset/train-labels.idx1-ubyte",
    "dataset/t10k-images.idx3-ubyte",
    "dataset/t10k-labels.idx1-ubyte"
)
