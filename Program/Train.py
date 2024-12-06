import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision.models import mobilenet_v2  
from utils.getdata import Data

def main():
    BATCH_SIZE = 32
    EPOCH = 5
    LEARNING_RATE = 0.001
    NUM_CLASSES = 6
    VALID_SPLIT = 0.2  

    aug_path = "./Dataset/Augmented Images/Augmented Images/FOLDS_AUG/"
    orig_path = "./Dataset/Original Images/Original Images/FOLDS/"

    dataset = Data(base_folder_aug=aug_path, base_folder_orig=orig_path)
    train_data = dataset.dataset_train + dataset.dataset_aug

    train_size = int((1 - VALID_SPLIT) * len(train_data))
    valid_size = len(train_data) - train_size
    train_dataset, valid_dataset = random_split(train_data, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Menggunakan MobileNet
    model = mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # Training
    train_losses, valid_losses = train_model(train_loader, valid_loader, model, loss_fn, optimizer, EPOCH)

    torch.save(model.state_dict(), "trained_mobilenet_v2.pth")

    # Plotting grafik training dan validasi
    plt.figure(figsize=(10, 5))
    plt.plot(range(EPOCH), train_losses, color="#3399e6", label='Training Loss')
    plt.plot(range(EPOCH), valid_losses, color="#ff6666", label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./training_validation_mobilenet.png")

def train_model(train_loader, valid_loader, model, loss_fn, optimizer, epochs):
    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        # Training loop
        model.train()
        loss_train = 0.0
        correct_train = 0
        total_train = 0

        for src, trg in train_loader:
            src = src.permute(0, 3, 1, 2).float()
            trg = torch.argmax(trg, dim=1)

            pred = model(src)
            loss = loss_fn(pred, trg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            _, predicted = torch.max(pred, 1)
            total_train += trg.size(0)
            correct_train += (predicted == trg).sum().item()

        train_losses.append(loss_train / len(train_loader))
        train_accuracy = 100 * correct_train / total_train

        # Validation loop
        model.eval()
        loss_valid = 0.0
        correct_valid = 0
        total_valid = 0

        with torch.no_grad():
            for src, trg in valid_loader:
                src = src.permute(0, 3, 1, 2).float()
                trg = torch.argmax(trg, dim=1)

                pred = model(src)
                loss = loss_fn(pred, trg)

                loss_valid += loss.item()
                _, predicted = torch.max(pred, 1)
                total_valid += trg.size(0)
                correct_valid += (predicted == trg).sum().item()

        valid_losses.append(loss_valid / len(valid_loader))
        valid_accuracy = 100 * correct_valid / total_valid

        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {loss_train / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Valid Loss: {loss_valid / len(valid_loader):.4f}, Valid Accuracy: {valid_accuracy:.2f}%")

    return train_losses, valid_losses

if __name__ == "__main__":
    main()