# api.py

from flask import Flask, request, jsonify
from train import train
from test import test
from validate import validate
from model_log import load_log, fetch_log_from_api, send_log_to_api
from config import Config

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train_model():
    # 백엔드 서버로부터 요청 데이터 받기
    data = request.get_json()
    train_annotations = data.get('train_annotations')
    val_annotations = data.get('val_annotations')
    img_dir = data.get('img_dir', Config.img_dir)
    num_epochs = data.get('num_epochs', Config.num_epochs)
    batch_size = data.get('batch_size', Config.batch_size)
    learning_rate = data.get('learning_rate', Config.learning_rate)
    # 기타 필요한 파라미터들을 추가로 받습니다.

    # 학습 파라미터 설정
    config = Config()
    config.train_annotations = train_annotations
    config.val_annotations = val_annotations
    config.img_dir = img_dir
    config.num_epochs = num_epochs
    config.batch_size = batch_size
    config.learning_rate = learning_rate
    # 필요한 파라미터를 config에 업데이트

    # 학습 시작
    train(config)
    return jsonify({'status': 'training started'})

# 나머지 엔드포인트는 동일하게 유지
