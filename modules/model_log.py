import requests
import json

def save_log(log_data, log_file):
    # 로그 데이터를 파일에 저장
    with open(log_file, 'w') as f:
        json.dump(log_data, f)

def load_log(log_file):
    # 로그 데이터를 파일에서 불러오기
    with open(log_file, 'r') as f:
        log_data = json.load(f)
    return log_data

def fetch_log_from_api(api_endpoint, params):
    # API를 통해 로그 데이터 조회
    response = requests.get(api_endpoint, params=params)
    if response.status_code == 200:
        log_data = response.json()
        return log_data
    else:
        print(f"Error fetching log: {response.status_code}")
        return None

def send_log_to_api(api_endpoint, log_data):
    # API를 통해 로그 데이터 전송
    response = requests.post(api_endpoint, json=log_data)
    if response.status_code == 200:
        print("Log data sent successfully")
    else:
        print(f"Error sending log: {response.status_code}")
