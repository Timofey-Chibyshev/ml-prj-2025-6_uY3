import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, criterion, optimizer, x_train, y_train, batch_size=64, num_epochs=5):
    x_train_normalized = np.array([np.array([x_train_elem]) for x_train_elem in x_train])

    x_train_tensor = torch.Tensor(x_train_normalized)
    y_train_tensor = torch.LongTensor(y_train)

    dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_step = len(train_loader)
    #loss_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Прямой запуск
            outputs = model(images)
            loss = criterion(outputs, labels)
            #loss_list.append(loss.item())

            # Обратное распространение и оптимизатор
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 1 == 0:
                print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

if __name__ == "__main__":
    import sys
    from model import IamWordsCRNN
    from loss import MyCTCLoss
    from data_loader import loader
    from alphabet import encode_texts
    from torch import optim
    from save import save_model

    num_epochs = int(sys.argv[1]) if len(sys.argv) >= 2 else 5

    print("Loading data...")
    x_train, y_train = loader.load_train_data()
    y_train = encode_texts(y_train)
    print("Training data has been loaded successfully!\n")

    learning_rate = 1e-3

    model = torch.nn.Module()
    try:
        model = loader.load_model()
    except FileNotFoundError:
        model = IamWordsCRNN()

    criterion = MyCTCLoss()
    optimizer = optim.NAdam(model.parameters(), lr=learning_rate)

    filename = "iam_words_crnn"
    try:
        print("Training model...")
        train_model(model, criterion, optimizer, x_train, y_train, num_epochs=num_epochs)
        print("Model has been trained successfully!\n")
    except KeyboardInterrupt:
        print("Interrupted by user!")
    finally:
        save_model(model, f"{filename}.ckpt")
        print(f"Model saved to \"{filename}.ckpt\"")
