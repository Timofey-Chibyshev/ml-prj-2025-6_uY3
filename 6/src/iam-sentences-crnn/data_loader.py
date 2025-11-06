import numpy as np
from preprocess_dataset import target_height, target_width

class IamSentencesDataLoader(object):
    def __init__(self):
        self.training_images_filepath = "dataset/train-images.idx3-ubyte"
        self.training_labels_filepath = "dataset/train-labels.idx1-U128"
        self.test_images_filepath = "dataset/t10k-images.idx3-ubyte"
        self.test_labels_filepath = "dataset/t10k-labels.idx1-U128"

    @staticmethod
    def load_images(path):
        images = np.fromfile(path, dtype=np.uint8)
        images = images.reshape(-1, target_height, target_width)
        return images
    
    @staticmethod
    def load_labels(path):
        labels = np.fromfile(path, dtype='<U128')
        return labels
    
    def load_train_data(self):
        return self.load_images(self.training_images_filepath), self.load_labels(self.training_labels_filepath)
    
    def load_test_data(self):
        return self.load_images(self.test_images_filepath), self.load_labels(self.test_labels_filepath)

    def load_data(self):
        return self.load_train_data(), self.load_test_data() 

loader = IamSentencesDataLoader()
