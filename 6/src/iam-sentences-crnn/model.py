import torch
from torch import nn

from preprocess_dataset import target_chunks, target_height, target_width
from alphabet import alphabet

from torchinfo import summary

class IamSentencesCRNN(nn.Module):
    def __init__(self):
        super(IamSentencesCRNN, self).__init__()

        self.convo_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.BatchNorm2d(num_features=32),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.BatchNorm2d(num_features=128),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))
        )

        self.hidden_size = 64
        self.lstm = nn.LSTM(input_size=128, hidden_size=self.hidden_size, 
                            batch_first=True, bidirectional=True, num_layers=2)

        self.linear_layer = nn.Linear(128, (len(alphabet) + 1)) # +1 for blank symbol

        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        out = self.convo_layers(x)

        assert out.shape[1:] == (128, 1, target_chunks)

        out = torch.reshape(out, (-1, 128, target_chunks))
        out = torch.transpose(out, 1, 2)
        out, (hn, cn) = self.lstm(out)

        out = self.linear_layer(out)
        out = self.softmax(out)

        return out
    
if __name__ == "__main__":
    model = IamSentencesCRNN()
    summary(model, input_size=(64, 1, target_height, target_width))
