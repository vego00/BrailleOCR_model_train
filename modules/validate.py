import torch
from model import create_model
from preprocess import create_dataloader
from postprocess import decode_predictions, calculate_metrics
from config import Config

def validate(model, val_loader, device):
    model.eval()
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = list(image.to(device) for image in images)
            outputs = model(images)

            predictions = decode_predictions(outputs, conf_threshold=0.5, iou_threshold=0.5)
            all_predictions.extend(predictions)
            all_ground_truths.extend(targets)

    # 평가 지표 계산
    metrics = calculate_metrics(all_predictions, all_ground_truths)
    print(f"Validation Precision: {metrics['precision']:.4f}")
    print(f"Validation Recall: {metrics['recall']:.4f}")
    print(f"Validation F1 Score: {metrics['f1_score']:.4f}")

    return metrics
