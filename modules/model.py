# model.py

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes):
    # 사전 학습된 Faster R-CNN 모델 불러오기
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # 클래스 수에 맞게 헤드 수정
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
