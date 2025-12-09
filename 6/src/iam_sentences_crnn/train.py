import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

def train_model(model, criterion, accuracy, optimizer, x_train, y_train, x_test, y_test, batch_size=64, num_epochs=5):
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
    train_loss_list = np.zeros(num_epochs, dtype=np.float64)
    train_acc_list = np.zeros(num_epochs, dtype=np.float64)
    val_loss_list = np.zeros(num_epochs, dtype=np.float64)
    val_acc_list = np.zeros(num_epochs, dtype=np.float64)

    best_val_loss = 1000

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-5, patience=2)

    for epoch in range(num_epochs):
        training_loss = 0
        training_acc = 0
        model.train()
        epoch_start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            # Прямой запуск
            outputs = model(images)
            loss = criterion(outputs, labels)
            training_loss += loss.item()
            acc = accuracy(outputs, labels)
            training_acc += acc.item()

            # Обратное распространение и оптимизатор
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if (i + 1) % 1 == 0:
            #     print('Epoch [{}/{}], Batch [{}/{}], Acc: {:.4f}, LR: {}'
            #         .format(epoch + 1, num_epochs, i + 1, total_step, training_acc, optimizer.param_groups[0]['lr']))
        
        # Валидация
        val_loss = 0
        val_acc = 0
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
            # Прямой запуск
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                acc = accuracy(outputs, labels)
                val_acc += acc.item()

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        # Оцениваем лосс и точность
        val_loss /= len(test_loader)
        val_acc /= len(test_loader)
        training_loss /= total_step
        training_acc /= total_step

        print(f"Epoch {epoch + 1}: {training_loss:.4f} train loss, {training_acc:.1f}% train acc, {val_loss:.4f} val loss, {val_acc:.1f}% val acc, {epoch_time:.2f} seconds; LR: {optimizer.param_groups[0]['lr']}")

        train_loss_list[epoch] = training_loss
        train_acc_list[epoch] = training_acc
        val_loss_list[epoch] = val_loss
        val_acc_list[epoch] = val_acc

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Сохраняем лучшую модель
            save_model(model, "iam_sentences_crnn.ckpt")
        else:
            patience_counter += 1
            if patience_counter >= 10:  # 10 эпох без улучшений
                print("Early stopping!")
                break

        train_loss_list.tofile(os.path.join(script_dir, "model", "train_loss.f64"))
        train_acc_list.tofile(os.path.join(script_dir, "model", "train_acc.f64"))
        val_loss_list.tofile(os.path.join(script_dir, "model", "val_loss.f64"))
        val_acc_list.tofile(os.path.join(script_dir, "model", "val_acc.f64"))

        scheduler.step(val_loss)
    return


if __name__ == "__main__":
    import sys
    from .model import IamSentencesCRNN
    from .loss import MyCTCLoss, MyCERAccuracy
    from .data_loader import loader
    from .alphabet import encode_texts
    from torch import optim
    from .save import save_model
    import matplotlib.pyplot as plt

    num_epochs = 100

    print("Loading training data...")
    x_train, y_train = loader.load_train_data()
    y_train = encode_texts(y_train)
    print("Training data has been loaded successfully!\n")

    print("Loading validating data...")
    x_test, y_test = loader.load_test_data()
    y_test = encode_texts(y_test)
    print("Test data has been loaded successfully!\n")

    learning_rate = 0.001

    model = torch.nn.Module()
    try:
        model = loader.load_model()
    except FileNotFoundError:
        model = IamSentencesCRNN()

    criterion = MyCTCLoss()
    accuracy = MyCERAccuracy()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    filename = "iam_sentences_crnn"
    try:
        print("Training model...")
        train_model(model, criterion, accuracy, optimizer, x_train, y_train, x_test, y_test, num_epochs=num_epochs)
        print("Model has been trained successfully!\n")
    except KeyboardInterrupt:
        print("Interrupted by user!")
