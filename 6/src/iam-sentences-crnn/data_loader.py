import numpy as np
from preprocess_dataset import target_height, target_width
from model import IamSentencesCRNN
import torch

class IamSentencesDataLoader(object):
    def __init__(self):
        self.training_images_filepath = "dataset/train-images.idx3-ubyte"
        self.training_labels_filepath = "dataset/train-labels.idx1-U128"
        self.test_images_filepath = "dataset/t10k-images.idx3-ubyte"
        self.test_labels_filepath = "dataset/t10k-labels.idx1-U128"
        self.model_filepath = "model/iam_sentences_crnn.ckpt"

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
    
    def load_model(self):
        model = IamSentencesCRNN()
        try:
            print("Loading model...")
            model.load_state_dict(
                torch.load(self.model_filepath)
            )
            print("Model has been loaded successfully!\n")
            return model
        except FileNotFoundError:
            print(f"No {self.model_filepath} file exists!\n")
            raise FileNotFoundError

loader = IamSentencesDataLoader()
