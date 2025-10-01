import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from model import MnistNumbersCNN

def validate_model(model, x_test, y_test, batch_size=100):
    x_test_normalized = np.array([np.array([x_test_elem]) for x_test_elem in x_test])

    x_test_tensor = torch.Tensor(x_test_normalized)
    y_test_tensor = torch.LongTensor(y_test)

    dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the test images: {} %'.format((correct / total) * 100))

if __name__ == "__main__":
    import torch
    import sys
    from save import dir
    from data_loader import loader

    filename = "mnist_numbers_cnn" if len(sys.argv) <= 1 else sys.argv[1] 

    model = MnistNumbersCNN()
    try:
        print("Loading model...")
        model.load_state_dict(
            torch.load(f"{dir}/{filename}.ckpt")
        )
        print("Model has been loaded successfully!\n")
    except FileNotFoundError:
        print(f"No {filename}.ckpt file exists!")
        exit(-1)

    print("Loading data...")
    x_test, y_test = loader.load_test_data()
    print("Training data has been loaded successfully!\n")

    print("Validating model...")
    validate_model(model, x_test, y_test)
    print("Model has been validated successfully!")
