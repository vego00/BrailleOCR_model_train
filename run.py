import torch
from OCR.data_utils import data
from OCR.model.params import params, settings
from OCR import local_config
from pathlib import Path

import boto3
import os
from pymongo import MongoClient
import json
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, Response
from flasgger import Swagger
# http://localhost:5000/apidocs/

app = Flask(__name__)
swagger = Swagger(app)

bucket_name = os.environ['BUCKET_NAME']
model_path = os.environ['MODEL_PATH']

def download_from_s3(bucket_name, s3_key, local_path):
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, s3_key, local_path)
    
    
def download_from_docdb(train_local_path, train_list_file_names, val_list_file_names):
    # 환경 변수에서 연결 정보 가져오기
    docdb_host = os.environ.get('DOCDB_HOST')
    docdb_port = int(os.environ.get('DOCDB_PORT', 27017))
    docdb_user = os.environ.get('DOCDB_USER')
    docdb_password = os.environ.get('DOCDB_PASSWORD')
    docdb_database = os.environ.get('DOCDB_DATABASE')
    docdb_collection = os.environ.get('DOCDB_COLLECTION')

    # 연결 문자열 생성
    uri = f"mongodb://{docdb_user}:{docdb_password}@{docdb_host}:{docdb_port}/?ssl=true&retryWrites=false"

    # MongoClient 생성
    client = MongoClient(uri)

    # 데이터베이스와 컬렉션 선택
    db = client[docdb_database]
    collection = db[docdb_collection]

    # 학습 데이터와 검증 데이터 다운로드
    # train_list_file_names에 해당하는 데이터 가져오기
    for file_name in train_list_file_names:
        # 파일 이름으로 검색 (필요에 따라 키를 수정하세요)
        document = collection.find_one({'image_path': file_name})
        if document:
            # 문서를 파일로 저장 (예: JSON 형식)
            json_name = file_name.replace('.jpg', '.json')
            with open(f"{train_local_path}/{json_name}", 'w', encoding='utf-8') as f:
                json.dump(document, f, ensure_ascii=False)
        else:
            print(f"Train 파일 '{file_name}'을(를) 찾을 수 없습니다.")

    # val_list_file_names에 해당하는 데이터 가져오기
    for file_name in val_list_file_names:
        # 파일 이름으로 검색 (필요에 따라 키를 수정하세요)
        document = collection.find_one({'image_path': file_name})
        if document:
            # 문서를 파일로 저장 (예: JSON 형식)
            json_name = file_name.replace('.jpg', '.json')
            with open(f"{train_local_path}/{json_name}", 'w', encoding='utf-8') as f:
                json.dump(document, f, ensure_ascii=False)
        else:
            print(f"Validation 파일 '{file_name}'을(를) 찾을 수 없습니다.")

    print("DocumentDB에서 학습 및 검증 데이터를 다운로드했습니다.")
    


def load_data(params, collate_fn, train_list_file_names, val_list_file_names):
    train_local_path = Path(local_config.data_path) / 'data_train'
    with open(train_local_path / 'train_list.txt', 'w') as f:
        for item in train_list_file_names:
            f.write("%s\n" % item)
    with open(train_local_path / 'val_list.txt', 'w') as f:
        for item in val_list_file_names:
            f.write("%s\n" % item)
            
    download_from_s3(bucket_name, train_list_file_names, train_local_path)
    download_from_s3(bucket_name, val_list_file_names, train_local_path)
    download_from_docdb(train_local_path, train_list_file_names, val_list_file_names)
    
    train_loader = data.create_dataloader(params, collate_fn, list_file_names=train_list_file_names, shuffle=True)
    val_loaders = { k: data.create_dataloader(params, collate_fn, list_file_names=v, shuffle=False)
                    for k,v in val_list_file_names.items() }
    print('data loaded. train:{} batches'.format(len(train_loader)))
    for k,v in val_loaders.items():
        print('             {}:{} batches'.format(k, len(v)))
    return train_loader, val_loaders

def clean_up(train_local_path):
    for file_name in os.listdir(train_local_path):
        os.remove(os.path.join(train_local_path, file_name))
    os.rmdir(train_local_path)


@app.route('/train', methods=['POST'])
def run():

    max_epochs = request.args.get('max_epochs')
    train_list_file_names = request.args.get('train_list_file_names')
    val_list_file_names = request.args.get('val_list_file_names')
    
    model_local_path = Path(local_config.data_path) / params.load_model_from
    train_local_path = Path(local_config.data_path) / 'data_train'
    
    try:
        os.mkdir(model_local_path)
    except FileExistsError:
        pass
    
    try:
        os.mkdir(train_local_path)
    except FileExistsError:
        clean_up(train_local_path)
        os.mkdir(train_local_path)
    
    with open(train_local_path / 'train_list.txt', 'w') as f:
        for item in train_list_file_names:
            f.write("%s\n" % item)
    with open(train_local_path / 'val_list.txt', 'w') as f:
        for item in val_list_file_names:
            f.write("%s\n" % item)
    
    download_from_s3(bucket_name, model_path, model_local_path)
    download_from_s3(bucket_name, train_list_file_names, train_local_path)
    download_from_s3(bucket_name, val_list_file_names, train_local_path)
    download_from_docdb(train_local_path, train_list_file_names, val_list_file_names)
    
    from OCR.model import train
    
    return Response(status=200)

@app.route('/test', methods=['GET'])
def test():
    """
    Test API
    This is only a test.
    ---
    tags:
      - test
      responses:
        200:
          description: Test successful
    """
    return Response(status=200)