import torch
import numpy as np
from torchvision.ops import nms

def apply_nms(prediction, iou_threshold=0.5):
    # NMS 적용
    keep = nms(prediction['boxes'], prediction['scores'], iou_threshold)
    prediction['boxes'] = prediction['boxes'][keep]
    prediction['scores'] = prediction['scores'][keep]
    prediction['labels'] = prediction['labels'][keep]
    return prediction

def decode_predictions(outputs, conf_threshold=0.5, iou_threshold=0.5):
    # 모델 출력 후처리
    outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]
    results = []

    for output in outputs:
        pred_scores = output['scores']
        mask = pred_scores >= conf_threshold

        prediction = {}
        prediction['boxes'] = output['boxes'][mask]
        prediction['scores'] = pred_scores[mask]
        prediction['labels'] = output['labels'][mask]

        # NMS 적용
        prediction = apply_nms(prediction, iou_threshold)
        results.append(prediction)

    return results

def calculate_iou(box1, box2):
    # 두 박스 간의 IoU 계산
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max((x2 - x1), 0) * max((y2 - y1), 0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def calculate_metrics(predictions, ground_truths, iou_threshold=0.5):
    # 평가 지표 계산 (Precision, Recall, F1 Score)
    TP = 0
    FP = 0
    FN = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred['boxes'].numpy()
        pred_labels = pred['labels'].numpy()
        gt_boxes = gt['boxes'].numpy()
        gt_labels = gt['labels'].numpy()

        matched_gt = []

        for i in range(len(pred_boxes)):
            pred_box = pred_boxes[i]
            pred_label = pred_labels[i]
            match_found = False
            for j in range(len(gt_boxes)):
                if j in matched_gt:
                    continue
                iou = calculate_iou(pred_box, gt_boxes[j])
                if iou >= iou_threshold and pred_label == gt_labels[j]:
                    TP += 1
                    matched_gt.append(j)
                    match_found = True
                    break
            if not match_found:
                FP += 1

        FN += len(gt_boxes) - len(matched_gt)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    return metrics
