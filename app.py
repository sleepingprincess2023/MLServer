from flask import Flask, request, jsonify
from checkout_studying import CHECKOUT_STUDYING
from PIL import Image
import os
from io import BytesIO
import base64
import uuid

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect_studying():
    try:
        # 요청에서 이미지 데이터를 추출합니다
        data = request.json['image']
        
        # base64로 인코딩된 이미지를 디코딩합니다
        image_data = base64.b64decode(data)
        
        # 이미지 데이터를 PIL 이미지로 로드합니다
        image = Image.open(BytesIO(image_data))
        
        # 고유한 파일명을 생성합니다
        filename = f"{uuid.uuid4()}.jpg"
        
        # 이미지를 저장합니다
        image.save(filename)
        
        # 모델을 사용하여 이미지에서 점수를 계산합니다
        detection = CHECKOUT_STUDYING(5)
        score = detection.start_detection(filename)
        
        # 이미지 파일을 삭제합니다
        if os.path.exists(filename):
            os.remove(filename)

        # 점수를 반환합니다
        return jsonify({'score': score})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return "health"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
