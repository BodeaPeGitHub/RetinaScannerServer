from flask import Flask, request
from flask_cors import CORS
import os
from datetime import datetime
from received_image import ReceivedImage
from proprocessing import PrepocessingImages
from detector import Detector
import cv2
import numpy as np
import json
from flask import jsonify, send_file, make_response
from dataclasses import dataclass
from PIL import Image
import io
import base64
import tempfile
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, PageBreak
import reportlab.platypus as rep
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderPDF
from io import BytesIO
import webbrowser
import os
import matplotlib.pyplot as plt
import numpy as np

RECEIVED_IMAGES = os.path.join('images', 'received')

app = Flask(__name__)
CORS(app)

clients_pdf = {}
clients_images = {}
preprocessor = PrepocessingImages()
detector = Detector(os.path.join('model', 'best_model_v2_final.plk'))

def filestorage_to_cv2_image(filestorage):
    file_bytes = filestorage.read()
    np_array = np.frombuffer(file_bytes, np.uint8)
    cv2_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return cv2_image

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    sender_id = str(request.remote_addr)
    position = request.form.get('position')
    image_key = f'{str(sender_id)}_{str(position)}'
    
    if file.filename == '':
        return 'No file selected for uploading'
    
    if file:
        filename = str(sender_id) + '_' + position + '_' + str(datetime.now().strftime("%d_%b_%y_%H_%M_%S")) + '_' + file.filename
        file.save(os.path.join(RECEIVED_IMAGES, filename))
        clients_images[image_key] =  ReceivedImage(cv2.imread(os.path.join(RECEIVED_IMAGES, filename)), position, image_key)
        return 'File successfully uploaded'
    
def convert_image(img):
    img = Image.fromarray((img * 255).astype(np.uint8))
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


@app.route('/predict/<position>', methods=['GET'])
def predict(position):
    # Finding the key
    sender_id = str(request.remote_addr)
    image_key = f'{str(sender_id)}_{str(position)}'
    
    # Image preprocessing
    image_resized = preprocessor.preprocess_image(clients_images[image_key].image)
    predictions = detector.predict(image_resized)

    json_response = json.dumps(predictions)
    return jsonify(json_response)

@app.route('/analize/<position>', methods=['GET'])
def detect_images(position):
    # Finding the key
    sender_id = str(request.remote_addr)
    image_key = f'{str(sender_id)}_{str(position)}'
    
    # Image preprocessing
    image_resized = preprocessor.preprocess_image(clients_images[image_key].image)
    image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    heatmap = detector.generate_heatmap(image_resized)

    response = {
        'heatmap': convert_image(heatmap)
    }
    json_response = json.dumps(response)
    return jsonify(json_response)

tmp_path = os.path.join('images', 'tmp')

def get_image_component(req):
    img = base64.b64decode(req['img'].split(',')[1])
    exp = base64.b64decode(req['exp'].split(',')[1])
    img = Image.open(io.BytesIO(img))
    exp = Image.open(io.BytesIO(exp))
    img_temp = tempfile.NamedTemporaryFile(delete=True).name
    exp_temp = tempfile.NamedTemporaryFile(delete=True).name
    img.save(img_temp, 'jpeg')
    exp.save(exp_temp, 'jpeg')
    # img.save(os.path.join('images', 'tmp', 'img'), 'jpeg')
    # exp.save(os.path.join('images', 'tmp', 'exp'), 'jpeg')
    return img_temp, exp_temp, req['pred']

def generate_bar_chart(data):
    labels = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    values = data
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    temp = tempfile.NamedTemporaryFile(delete=True).name + '.png'
    plt.savefig(temp)
    plt.close()
    return temp

@app.route('/generate-pdf', methods=['POST'])
def generate_pdf():
    data = request.get_json()

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    story.append(Paragraph("RetinaScanner Report", styles['Title']))
    story.append(Spacer(1, 20))
    
    if data.get('left'):
        img_p, exp_p, pred = get_image_component(data.get('left'))
        story.append(Paragraph('Left eye', styles['Heading2']))
        story.append(Spacer(10, 10))
        im1 = rep.Image(img_p, 230, 230)
        im2 = rep.Image(exp_p, 230, 230)  
        images = [['Real image', Spacer(1, 10), 'Heatmap'], [im1, Spacer(1, 10), im2]]
        tbl = Table(images)
        
        pred_p = generate_bar_chart(pred)

        chart_im = rep.Image(pred_p, 450, 250)  # Adjusted size
        story.append(rep.KeepTogether([tbl, Spacer(1, 10), chart_im]))
        story.append(PageBreak())

    if data.get('right'):
        img_p, exp_p, pred = get_image_component(data.get('right'))
        story.append(Paragraph('Right eye', styles['Heading2']))
        story.append(Spacer(10, 10))
        im1 = rep.Image(img_p, 230, 230)
        im2 = rep.Image(exp_p, 230, 230)  
        images = [['Real image', Spacer(1, 10), 'Heatmap'], [im1, Spacer(1, 10), im2]]
        tbl = Table(images)
        
        pred_p = generate_bar_chart(pred)

        chart_im = rep.Image(pred_p, 450, 250)  # Adjusted size
        story.append(rep.KeepTogether([tbl, Spacer(1, 10), chart_im]))
        story.append(PageBreak())

    
    doc.build(story)
    buffer.seek(0)
    response = make_response(send_file(buffer, mimetype='application/pdf'))
    response.headers.set('Content-Disposition', 'attachment', filename='eye_analisys.pdf')
    return response

if __name__ == "__main__":
    app.run(debug=True)
