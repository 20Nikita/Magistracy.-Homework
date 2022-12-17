import torch
from torch.utils.data import Dataset
import pandas
import cv2
import numpy as np

class Retinopatia(Dataset):
    def __init__(self, annotation, root, N_class, transform=None, t1=None, t2=None, t3=None):
        self.landmarks_frame = pandas.read_csv(annotation)
        self.root = root
        self.transform = transform
        self.N_class = N_class
        self.t1=t1

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.root + "/" + self.t1 + "/" + self.landmarks_frame.iloc[idx, 2] + ".jpeg"
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        landmarks = self.landmarks_frame.iloc[idx, 3:]
        landmarks = np.array(landmarks)
        landmarks = landmarks.astype('int')
        classification_labels = landmarks[0]
        return image, classification_labels

def retinopatia(annotation, root, N_class, transform=None, t1=None, t2=None, t3=None):
    return(Retinopatia(annotation, root, N_class, transform, t1, t2, t3))

