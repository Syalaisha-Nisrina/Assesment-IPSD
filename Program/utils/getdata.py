import os
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, folder="./Dataset/"):
        """
        Inisialisasi dataset dengan memuat gambar dan label.
        """
        self.dataset = []
        onehot = np.eye(5) 

        # Iterasi melalui folder dataset
        for class_idx, class_name in enumerate(os.listdir(folder)):
            class_folder = os.path.join(folder, class_name)
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)

                # Membaca dan memproses gambar
                image = cv.resize(cv.imread(img_path), (32, 32) / 255  
                self.dataset.append([image, onehot[class_idx]])

        # Menampilkan dataset yang telah dimuat
        print("Dataset Loaded ")
        for example in self.dataset:
            print("Image Shape:", example[0].shape, "Label:", example[1])

    def __len__(self):
        """
        Mengembalikan jumlah data dalam dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Mengembalikan tuple (image, label) dari dataset.
        """
        features, label = self.dataset[idx]
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )


if __name__ == "__main__":
    data = Data(folder="./Dataset/")
