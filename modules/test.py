import torch
from model import create_model
from preprocess import create_dataloader
from postprocess import decode_predictions, calculate_metrics
from config import Config

def test():
    config = Config()
    device = torch.device(config.device)

    # 데이터로더 생성
    test_loader = create_dataloader(
        annotations_file=config.test_annotations,
        img_dir=config.test_img_dir,
        batch_size=1,  # 평가 시에는 배치 크기를 1로 설정하는 것이 일반적입니다.
        shuffle=False,
        num_workers=config.num_workers,
        train=False
    )

    # 모델 로드
    model = create_model(num_classes=config.num_classes)
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = list(image.to(device) for image in images)
            outputs = model(images)

            predictions = decode_predictions(outputs, conf_threshold=0.5, iou_threshold=0.5)
            all_predictions.extend(predictions)
            all_ground_truths.extend(targets)

    # 평가 지표 계산
    metrics = calculate_metrics(all_predictions, all_ground_truths)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")

if __name__ == '__main__':
    test()
