import torch
import torch.optim as optim
from model import create_model
from preprocess import create_dataloader
from config import Config
from utils import save_checkpoint, Averager
import time

def train():
    config = Config()
    device = torch.device(config.device)

    # 데이터로더 생성
    train_loader = create_dataloader(
        annotations_file=config.train_annotations,
        img_dir=config.train_img_dir,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        train=True
    )

    val_loader = create_dataloader(
        annotations_file=config.val_annotations,
        img_dir=config.val_img_dir,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        train=False
    )

    # 모델 생성
    model = create_model(num_classes=config.num_classes)
    model.to(device)

    # 옵티마이저 설정
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=config.learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 체크포인트 로드 (필요한 경우)
    start_epoch = 0
    if config.resume:
        checkpoint = torch.load(config.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # 학습 루프
    num_epochs = config.num_epochs
    loss_hist = Averager()

    for epoch in range(start_epoch, num_epochs):
        model.train()
        loss_hist.reset()

        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loss_hist.send(loss_value)

        # 학습률 스케줄러 업데이트
        lr_scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss_hist.value:.4f}")

        # 체크포인트 저장
        save_checkpoint(model, optimizer, epoch, config.checkpoint_path)

        # 검증 단계
        # validate(model, val_loader, device)

if __name__ == '__main__':
    train()
