# -*- coding: utf-8 -*-
import albumentations as A
import albumentations.pytorch as Ap


def get_ransforms(SIZE):
    val_transforms = A.Compose([
        A.Resize(SIZE, SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        Ap.transforms.ToTensorV2()
        ])
    train_transforms =A.Compose([
        A.Resize(SIZE, SIZE),
        A.Flip(p=0.5),#Отразите вход по горизонтали, вертикали или по горизонтали и вертикали.
        A.Rotate(p=0.5),#Поверните ввод на угол, случайно выбранный из равномерного распределения. 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        Ap.transforms.ToTensorV2()
        ])    
    return val_transforms, train_transforms
