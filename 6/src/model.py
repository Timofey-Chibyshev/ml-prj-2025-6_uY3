from torch import nn

class MnistNumbersCNN(nn.Module):
    def __init__(self):
        super(MnistNumbersCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.dropout = nn.Dropout()
        self.full1 = nn.Sequential(
            nn.Linear(7 * 7 * 8, 100),
            nn.ReLU()
        )
        self.full2 = nn.Linear(100, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.dropout(out)
        out = self.full1(out)
        out = self.full2(out)
        return out
