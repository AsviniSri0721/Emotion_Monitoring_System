import os
import pickle
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import torch
import logging
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

class ModelService:
    """
    Service to load and run trained ML models:
    - CNN models (from .pkl files) for emotion classification
    - YOLOv8 (Ultralytics) for face detection
    """
    
    def __init__(self):
        self.models_loaded = False
        self.cnn_model = None
        self.yolo_model = None
        self.model_config = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
    
    def load_models(self):
        """Load all trained models"""
        try:
            # Load YOLOv8 model for face detection
            yolo_model_path = os.getenv('YOLO_MODEL_PATH', 'models/yolov8_face.pt')
            if os.path.exists(yolo_model_path):
                self.yolo_model = YOLO(yolo_model_path)
                logger.info(f"YOLOv8 model loaded from {yolo_model_path}")
            else:
                # Use default YOLOv8n if custom model not found
                logger.warning(f"YOLOv8 model not found at {yolo_model_path}, using default")
                self.yolo_model = YOLO('yolov8n.pt')
            
            # Load CNN emotion classification model
            cnn_model_path = os.getenv('CNN_MODEL_PATH', 'models/emotion_cnn_model.pkl')
            if os.path.exists(cnn_model_path):
                with open(cnn_model_path, 'rb') as f:
                    self.cnn_model = pickle.load(f)
                logger.info(f"CNN model loaded from {cnn_model_path}")
            else:
                logger.warning(f"CNN model not found at {cnn_model_path}")
            
            # Load additional models if specified
            self._load_additional_models()
            
            self.models_loaded = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def _load_additional_models(self):
        """Load additional CNN models if specified in environment"""
        additional_models = os.getenv('ADDITIONAL_MODEL_PATHS', '').split(',')
        self.additional_models = {}
        
        for model_path in additional_models:
            if model_path.strip() and os.path.exists(model_path.strip()):
                try:
                    with open(model_path.strip(), 'rb') as f:
                        model_name = os.path.basename(model_path.strip()).replace('.pkl', '')
                        self.additional_models[model_name] = pickle.load(f)
                        logger.info(f"Additional model loaded: {model_name}")
                except Exception as e:
                    logger.error(f"Error loading additional model {model_path}: {str(e)}")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in image using YOLOv8
        Returns list of face bounding boxes and confidence scores
        """
        if self.yolo_model is None:
            raise ValueError("YOLOv8 model not loaded")
        
        try:
            # Run YOLOv8 inference
            results = self.yolo_model(image, conf=0.25, verbose=False)
            
            faces = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    faces.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence
                    })
            
            return faces
            
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return []
    
    def preprocess_face(self, image: np.ndarray, bbox: List[int], target_size: Tuple[int, int] = (48, 48)) -> np.ndarray:
        """
        Extract and preprocess face region for emotion classification
        """
        x1, y1, x2, y2 = bbox
        
        # Extract face region
        face_roi = image[y1:y2, x1:x2]
        
        # Convert to grayscale if needed
        if len(face_roi.shape) == 3:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size
        face_roi = cv2.resize(face_roi, target_size)
        
        # Normalize pixel values
        face_roi = face_roi.astype(np.float32) / 255.0
        
        # Reshape for model input (add batch and channel dimensions)
        # Adjust based on your model's expected input shape
        face_roi = face_roi.reshape(1, target_size[0], target_size[1], 1)
        
        return face_roi
    
    def predict_emotion(self, face_image: np.ndarray) -> Dict[str, float]:
        """
        Predict emotion from preprocessed face image using CNN model
        Returns dictionary of emotion probabilities
        """
        if self.cnn_model is None:
            raise ValueError("CNN model not loaded")
        
        try:
            # Predict using the loaded model
            # Adjust this based on your model's API (sklearn, keras, pytorch, etc.)
            if hasattr(self.cnn_model, 'predict_proba'):
                # Scikit-learn model
                probabilities = self.cnn_model.predict_proba(face_image.reshape(1, -1))[0]
                emotion_classes = self.cnn_model.classes_
            elif hasattr(self.cnn_model, 'predict'):
                # Keras/TensorFlow model
                probabilities = self.cnn_model.predict(face_image, verbose=0)[0]
                # Define emotion classes based on your training
                emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            elif hasattr(self.cnn_model, '__call__'):
                # PyTorch model
                with torch.no_grad():
                    input_tensor = torch.from_numpy(face_image).float()
                    if self.device == 'cuda':
                        input_tensor = input_tensor.to(self.device)
                    output = self.cnn_model(input_tensor)
                    probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                    emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            else:
                raise ValueError("Unknown model type")
            
            # Create emotion dictionary
            emotion_dict = {}
            for i, emotion in enumerate(emotion_classes):
                emotion_dict[emotion] = float(probabilities[i])
            
            return emotion_dict
            
        except Exception as e:
            logger.error(f"Error in emotion prediction: {str(e)}")
            raise
    
    def detect_emotions_in_image(self, image: np.ndarray) -> List[Dict]:
        """
        Complete pipeline: detect faces and predict emotions
        Returns list of detections with emotions
        """
        # Detect faces
        faces = self.detect_faces(image)
        
        results = []
        for face in faces:
            bbox = face['bbox']
            
            # Preprocess face
            face_image = self.preprocess_face(image, bbox)
            
            # Predict emotion
            emotions = self.predict_emotion(face_image)
            
            # Get dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            
            # Map to engagement states
            engagement_score = self._calculate_engagement(emotions, dominant_emotion[0])
            
            results.append({
                'bbox': bbox,
                'emotions': emotions,
                'dominant_emotion': dominant_emotion[0],
                'confidence': dominant_emotion[1],
                'engagement_score': engagement_score,
                'face_confidence': face['confidence']
            })
        
        return results
    
    def _calculate_engagement(self, emotions: Dict[str, float], dominant: str) -> float:
        """
        Calculate engagement score based on emotions
        Higher score = more engaged
        """
        # Positive emotions increase engagement
        positive_emotions = ['happy', 'surprise']
        negative_emotions = ['sad', 'angry', 'fear', 'disgust']
        neutral_emotions = ['neutral']
        
        if dominant in positive_emotions:
            base_score = 0.7 + (emotions.get(dominant, 0) * 0.3)
        elif dominant in negative_emotions:
            base_score = 0.3 - (emotions.get(dominant, 0) * 0.2)
        else:  # neutral
            base_score = 0.5
        
        # Adjust based on confidence
        confidence = emotions.get(dominant, 0)
        engagement = base_score * confidence + 0.3 * (1 - confidence)
        
        return max(0.0, min(1.0, engagement))
    
    def map_emotion_to_state(self, emotion: str) -> str:
        """
        Map detected emotion to engagement state
        """
        emotion_mapping = {
            'happy': 'focused',
            'surprise': 'focused',
            'neutral': 'neutral',
            'sad': 'confused',
            'fear': 'confused',
            'angry': 'frustrated',
            'disgust': 'bored'
        }
        return emotion_mapping.get(emotion, 'neutral')

