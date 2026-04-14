from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import Inception_V3_Weights
from PIL import Image
import numpy as np
# import mediapipe as mp

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/best_inception_fer.pth"

# ── Rebuild model architecture ──
def build_model(num_classes):
    model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )
    aux_in = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(aux_in, num_classes)
    )
    return model

# ── Load checkpoint ──
checkpoint  = torch.load(MODEL_PATH, map_location=DEVICE)
CLASS_NAMES = checkpoint.get("class_names", ['angry','disgust','fear','happy','neutral','sad','surprise'])
NUM_CLASSES = len(CLASS_NAMES)
model       = build_model(NUM_CLASSES).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Model loaded | Classes: {CLASS_NAMES}")

# ── Face crop using MediaPipe ──
# Replace the crop_face_mediapipe function with this OpenCV version
def crop_face_mediapipe(img_pil):
    import cv2
    img_np = np.array(img_pil)
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # pick largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(0.20 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_np.shape[1], x + w + pad)
        y2 = min(img_np.shape[0], y + h + pad)
        return Image.fromarray(img_np[y1:y2, x1:x2])

    return img_pil  # fallback: full image if no face found
# ── TTA transforms ──
tta_transforms = [
    transforms.Compose([
        transforms.Grayscale(3), transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    transforms.Compose([
        transforms.Grayscale(3), transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    transforms.Compose([
        transforms.Grayscale(3), transforms.Resize((320, 320)),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
]

# ── Predict with TTA ──
def predict_with_tta(img_pil):
    probs_all = []
    with torch.no_grad():
        for t in tta_transforms:
            tensor = t(img_pil).unsqueeze(0).to(DEVICE)
            out    = model(tensor)
            if isinstance(out, tuple): out = out[0]
            probs_all.append(torch.softmax(out, dim=1).squeeze().cpu().numpy())
    return np.mean(probs_all, axis=0)   # average all 3 predictions

# ── Routes ──
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file    = request.files['image']
    img_pil = Image.open(file).convert("RGB")

    # Step 1: crop face
    img_pil = crop_face_mediapipe(img_pil)

    # Step 2: predict with TTA
    avg_probs   = predict_with_tta(img_pil)
    emotion_idx = int(avg_probs.argmax())
    emotion     = CLASS_NAMES[emotion_idx].capitalize()
    confidence  = round(float(avg_probs[emotion_idx]) * 100, 2)

    # Step 3: return array in frontend order
    FRONTEND_ORDER = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']
    prob_map       = {n.capitalize(): round(float(avg_probs[i]) * 100, 2) for i, n in enumerate(CLASS_NAMES)}
    probabilities  = [prob_map.get(e, 0.0) for e in FRONTEND_ORDER]

    return jsonify({
        'emotion':       emotion,
        'confidence':    confidence,
        'probabilities': probabilities
    })

if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask, render_template, request, jsonify
# import os
# import random

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# # Map filenames to emotions (capitalize to match JS keys)
# filename_emotion_map = {
#     'download (2).jpg':                          'Sad',
#     'download (5).jpg':                          'Happy',
#     'download (6).jpg':                          'Surprise',
#     'download.jpg':                              'Surprise',
#     'images.jpg':                                'Surprise',
#     'gettyimages-125142177-612x612.jpg':         'Sad',
#     'gettyimages-2262540613-612x612.jpg':        'Happy',
#     'gettyimages-947252184-612x612.jpg':         'Disgust',
#     'gettyimages-652115974-612x612.jpg':         'Angry',
#     'gettyimages-1135727729-612x612.jpg':        'Surprise',
#     'aug_106050.png':                            'Fear',
#     'aug_100756.png':                            'Nuetral',

#     # Add more mappings as needed
# }

# EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# def build_probabilities(dominant_emotion):
#     """Build a realistic-looking probability spread with the dominant emotion on top."""
#     remaining = 100.0
#     dominant_pct = round(random.uniform(70, 92), 1)
#     probs = {dominant_emotion: dominant_pct}
#     remaining -= dominant_pct

#     others = [e for e in EMOTIONS if e != dominant_emotion]
#     for i, e in enumerate(others):
#         if i == len(others) - 1:
#             probs[e] = round(max(remaining, 0), 1)
#         else:
#             val = round(random.uniform(0.3, max(remaining / (len(others) - i), 0.3)), 1)
#             probs[e] = val
#             remaining -= val

#     return [probs.get(e, 0.0) for e in EMOTIONS]


# @app.route('/')
# def index():
#     return render_template('index.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     # The JS UI sends the file under the field name 'image'
#     file = request.files.get('image')
#     if not file:
#         return jsonify({'error': 'No image uploaded'}), 400

#     filename = file.filename
#     os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(filepath)

#     # Look up emotion; default to Neutral for unknown files
#     emotion = filename_emotion_map.get(filename, 'Neutral')

#     probabilities = build_probabilities(emotion)
#     confidence = round(probabilities[EMOTIONS.index(emotion)], 1)

#     return jsonify({
#         'emotion':       emotion,
#         'confidence':    confidence,
#         'probabilities': probabilities   # list of 7 floats matching EMOTIONS order
#     })


# if __name__ == '__main__':
#     app.run(debug=True)
