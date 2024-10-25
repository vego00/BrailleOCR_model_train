# config.py

import torch

class Config:
    # 데이터 관련 설정
    # 추후 몽고디비로 변경
    train_annotations = 'data/train_annotations.json'
    val_annotations = 'data/val_annotations.json'
    test_annotations = 'data/test_annotations.json'
    train_img_dir = 'data/train_images'
    val_img_dir = 'data/val_images'
    test_img_dir = 'data/test_images'

    # 이미지 크기
    image_height = 512
    image_width = 512

    # 학습 관련 설정
    num_classes = 64  # 브라유 문자의 가능한 조합 수
    batch_size = 8
    num_epochs = 20
    learning_rate = 1e-4
    num_workers = 4
    checkpoint_path = 'checkpoints/model.pth'
    resume = False

    # 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
