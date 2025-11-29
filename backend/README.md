# Backend - Python Flask API

This backend uses Python with Flask to integrate your trained ML models (CNN .pkl files and YOLOv8).

## Setup

1. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Place your trained models:**
```
backend/
  models/
    emotion_cnn_model.pkl    # Your CNN model
    yolov8_face.pt          # Your YOLOv8 model (or use default)
```

4. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your settings
```

5. **Set model paths in .env:**
```env
CNN_MODEL_PATH=models/emotion_cnn_model.pkl
YOLO_MODEL_PATH=models/yolov8_face.pt
```

## Model Integration

### CNN Model (.pkl)
Your CNN model should be saved as a pickle file. The `ModelService` supports:
- Scikit-learn models (with `predict_proba`)
- Keras/TensorFlow models (with `predict`)
- PyTorch models (callable)

### YOLOv8 Model
Place your trained YOLOv8 model in the models directory, or the system will use the default YOLOv8n model.

## Running

```bash
python app.py
```

Or with gunicorn (production):
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## API Endpoints

- `POST /api/emotions/detect` - Detect emotions from image frame
- `GET /api/models/status` - Check model loading status
- `GET /api/health` - Health check

## Model Service

The `ModelService` class handles:
1. Loading CNN and YOLOv8 models
2. Face detection using YOLOv8
3. Emotion classification using CNN
4. Engagement score calculation

Modify `services/model_service.py` to match your model's input/output format.

