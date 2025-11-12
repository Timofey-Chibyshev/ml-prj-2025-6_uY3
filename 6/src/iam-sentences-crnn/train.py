import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time

def train_model(model, criterion, optimizer, x_train, y_train, x_test, y_test, batch_size=64, num_epochs=5):
    x_train_normalized = np.array([np.array([x_train_elem]) for x_train_elem in x_train])
    x_test_normalized = np.array([np.array([x_test_elem]) for x_test_elem in x_test])

    x_train_tensor = torch.Tensor(x_train_normalized)
    x_test_tensor = torch.Tensor(x_test_normalized)
    y_train_tensor = torch.LongTensor(y_train)
    y_test_tensor = torch.LongTensor(y_test)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    total_step = len(train_loader)
    train_loss_list = np.zeros(num_epochs)
    test_loss_list = np.zeros(num_epochs)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-5, patience=2)

    for epoch in range(num_epochs):
        training_loss = 0
        model.train()
        epoch_start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            # Прямой запуск
            outputs = model(images)
            loss = criterion(outputs, labels)
            training_loss += loss.item()

            # Обратное распространение и оптимизатор
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, LR: {}'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), scheduler.get_last_lr()[0]))
        
        # Валидация
        test_loss = 0
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
            # Прямой запуск
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        # Оцениваем лосс
        test_loss /= len(test_loader)
        training_loss /= total_step
        print(f"Epoch {epoch + 1}: {training_loss:.4f} training loss, {test_loss:.4f} validation loss, {epoch_time:.2f} seconds.")

        train_loss_list[epoch] = training_loss
        test_loss_list[epoch] = test_loss

        scheduler.step(test_loss)
    return train_loss_list, test_loss_list


if __name__ == "__main__":
    import sys
    from model import IamSentencesCRNN
    from loss import MyCTCLoss
    from data_loader import loader
    from alphabet import encode_texts
    from torch import optim
    from save import save_model
    import matplotlib.pyplot as plt

    num_epochs = int(sys.argv[1]) if len(sys.argv) >= 2 else 5

    print("Loading training data...")
    x_train, y_train = loader.load_train_data()
    y_train = encode_texts(y_train)
    print("Training data has been loaded successfully!\n")

    print("Loading validating data...")
    x_test, y_test = loader.load_test_data()
    y_test = encode_texts(y_test)
    print("Test data has been loaded successfully!\n")

    learning_rate = 0.01

    model = torch.nn.Module()
    try:
        model = loader.load_model()
    except FileNotFoundError:
        model = IamSentencesCRNN()

    criterion = MyCTCLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    filename = "iam_sentences_crnn"
    try:
        print("Training model...")
        train_loss, test_loss = \
            train_model(model, criterion, optimizer, x_train, y_train, x_test, y_test, num_epochs=num_epochs)
        print("Model has been trained successfully!\n")

        plt.plot(train_loss, label='loss')
        plt.plot(test_loss, label='val_loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    except KeyboardInterrupt:
        print("Interrupted by user!")
    finally:
        save_model(model, f"{filename}.ckpt")
        print(f"Model saved to \"{filename}.ckpt\"")
