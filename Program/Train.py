import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision import models
from utils.getdata import Data

def main():
    BATCH_SIZE = 32
    EPOCHS = 5
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

    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    train_losses, valid_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, targets in train_loader:
            inputs = inputs.permute(0, 3, 1, 2).float()  # Change the dimension order
            targets = torch.argmax(targets, dim=1)
            
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = torch.max(predictions, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        train_loss = running_train_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(train_loss)

        model.eval()
        running_valid_loss = 0.0
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs = inputs.permute(0, 3, 1, 2).float()
                targets = torch.argmax(targets, dim=1)

                predictions = model(inputs)
                loss = loss_fn(predictions, targets)

                running_valid_loss += loss.item()
                _, predicted = torch.max(predictions, 1)
                total_valid += targets.size(0)
                correct_valid += (predicted == targets).sum().item()

        valid_loss = running_valid_loss / len(valid_loader)
        valid_accuracy = 100 * correct_valid / total_valid
        valid_losses.append(valid_loss)

        print(f"Epoch {epoch+1}/{EPOCHS}: "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%")
        
    torch.save(model.state_dict(), "trained_mobilenet_v2.pth")
    plt.plot(range(EPOCHS), train_losses, label='Train Loss', color='blue')
    plt.plot(range(EPOCHS), valid_losses, label='Valid Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("training_validation_mobilenet.png")

if __name__ == "__main__":
    main()
