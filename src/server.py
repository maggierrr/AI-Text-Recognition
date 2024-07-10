from flask import Flask, request, jsonify, send_from_directory, render_template
from PIL import Image
import io
import func as func
import cv2
import numpy

app = Flask(__name__)

def recognize_text(pil_image):

    open_cv_image = numpy.array(pil_image)
    # 将 RGB 转换为 BGR（OpenCV 使用的颜色空间顺序）
    # open_cv_image = open_cv_image[:, :, ::-1].copy()
    # 转换为灰度图像
    # gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    data = func.api_infer(open_cv_image)

    # 这是一个伪函数，模拟文本识别。你需要用实际的文本识别逻辑替换它。
    # 比如，使用 Tesseract 或其他 OCR 库。
    
    return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/samples/<filename>')
def get_sample(filename):
    return send_from_directory('samples', filename)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        image = Image.open(io.BytesIO(file.read())).convert("L")
        result = recognize_text(image)
        return jsonify({'msg': "Recognized text from the image", 'data': result})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
