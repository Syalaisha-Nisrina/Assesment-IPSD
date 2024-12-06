import torch
from torch.utils.data import DataLoader, random_split
from torchvision.models import mobilenet_v2
from utils.getdata import Data
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(conf_matrix, class_names, normalized=False, output_file="confusion_matrix.png"):
    """
    Plot and save confusion matrix.

    Parameters:
    - conf_matrix: The confusion matrix to plot.
    - class_names: List of class names for the labels.
    - normalized: Whether to normalize the confusion matrix.
    - output_file: File name for saving the plot.
    """
    if normalized:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f' if normalized else 'd',
                cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalized else ''))
    plt.savefig("Confusion Matrix.png")
    plt.close()

def test_model(test_loader, model, num_classes, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for src, trg in test_loader:
            src = src.permute(0, 3, 1, 2).float()
            trg = torch.argmax(trg, dim=1)  # Convert one-hot labels to class indices

            logits = model(src)  # Output logits (raw scores before softmax)
            _, predicted = torch.max(logits, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(trg.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())  # Store logits for ROC-AUC

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)

    # Compute probabilities for ROC-AUC
    probabilities = torch.nn.functional.softmax(torch.tensor(all_logits), dim=1).numpy()

    # Compute metrics
    conf_matrix = confusion_matrix(all_labels, all_preds)
    accuracy = (all_preds == all_labels).mean() * 100
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    # ROC-AUC requires one-hot encoding for labels
    one_hot_labels = np.eye(num_classes)[all_labels]
    roc_auc = roc_auc_score(one_hot_labels, probabilities, multi_class='ovr')

    # Print metrics
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print(f"\nAccuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")
    print(f"ROC-AUC: {roc_auc:.2f}")

    # Plot normalized confusion matrix
    plot_confusion_matrix(conf_matrix, class_names, normalized=True, output_file="confusion_matrix_normalized.png")
    print("Normalized confusion matrix saved as 'confusion_matrix_normalized.png'.")

    # Plot ROC curve for each class
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(one_hot_labels[:, i], probabilities[:, i])
        plt.plot(fpr, tpr, label=f"Class {i} (AUC: {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.show()

def main():
    BATCH_SIZE = 32
    NUM_CLASSES = 6
    CLASS_NAMES = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4", "Class 5"]
    VALID_SPLIT = 0.2  
    TEST_SPLIT = 0.1

    aug_path = "./Dataset/Augmented Images/Augmented Images/FOLDS_AUG/"
    orig_path = "./Dataset/Original Images/Original Images/FOLDS/"

    dataset = Data(base_folder_aug=aug_path, base_folder_orig=orig_path)
    train_data = dataset.dataset_train + dataset.dataset_aug

    # Split dataset
    train_size = int((1 - VALID_SPLIT - TEST_SPLIT) * len(train_data))
    valid_size = int(VALID_SPLIT * len(train_data))
    test_size = len(train_data) - train_size - valid_size
    train_dataset, valid_dataset, test_dataset = random_split(
        train_data, [train_size, valid_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = mobilenet_v2(pretrained=True)
    model.classifier[1] = torch.nn.Linear(model.last_channel, NUM_CLASSES)
    model.load_state_dict(torch.load("trained_mobilenet_v2.pth"))
    model.eval()

    # Test model with metrics
    test_model(test_loader, model, NUM_CLASSES, CLASS_NAMES)

if __name__ == "__main__":
    main()

