import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import Config

class BrailleDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = self.load_annotations(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def load_annotations(self, annotations_file):
        # 어노테이션 파일 로드 및 파싱
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        return annotations

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # 이미지 및 레이블 로딩
        img_path = os.path.join(self.img_dir, self.img_labels[idx]['image'])
        image = np.array(Image.open(img_path).convert("RGB"))
        boxes = self.img_labels[idx]['boxes']
        labels = self.img_labels[idx]['labels']

        # 변환 적용
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.tensor(transformed['labels'], dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        return image, target

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ColorJitter(p=0.2),
        A.Resize(Config.image_height, Config.image_width),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_valid_transform():
    return A.Compose([
        A.Resize(Config.image_height, Config.image_width),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def collate_fn(batch):
    images = []
    targets = []

    for b in batch:
        images.append(b[0])
        targets.append(b[1])

    images = torch.stack(images)
    return images, targets

def create_dataloader(annotations_file, img_dir, batch_size, shuffle=True, num_workers=4, train=True):
    dataset = BrailleDataset(
        annotations_file=annotations_file,
        img_dir=img_dir,
        transform=get_train_transform() if train else get_valid_transform()
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return dataloader
