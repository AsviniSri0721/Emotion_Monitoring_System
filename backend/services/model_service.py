import os
import pickle
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO  # type: ignore[reportPrivateImportUsage]
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
            # Try .pt (PyTorch) first, then .pkl (pickle)
            cnn_model_path = os.getenv('CNN_MODEL_PATH', None)
            
            # If not specified, look for common model file names
            if not cnn_model_path:
                # Get the directory where this script is located
                script_dir = os.path.dirname(os.path.abspath(__file__))
                backend_dir = os.path.dirname(script_dir)  # Go up one level from services/
                project_root = os.path.dirname(backend_dir)  # Go up one more level
                cwd = os.getcwd()  # Current working directory
                
                possible_paths = [
                    os.path.join(backend_dir, 'models', 'efficientnetb0_emotion_best.pt'),  # backend/models/
                    os.path.join(backend_dir, 'backend', 'models', 'efficientnetb0_emotion_best.pt'),  # backend/backend/models/
                    os.path.join(project_root, 'backend', 'models', 'efficientnetb0_emotion_best.pt'),  # From project root
                    os.path.join(cwd, 'backend', 'models', 'efficientnetb0_emotion_best.pt'),  # From CWD
                    os.path.join(cwd, 'models', 'efficientnetb0_emotion_best.pt'),  # From CWD/models
                    os.path.join(backend_dir, 'models', 'emotion_cnn_model.pt'),
                    os.path.join(backend_dir, 'models', 'emotion_cnn_model.pkl'),
                    'models/efficientnetb0_emotion_best.pt',  # Relative to current working directory
                    'backend/models/efficientnetb0_emotion_best.pt',  # Relative to current working directory
                    'backend/backend/models/efficientnetb0_emotion_best.pt',  # Relative to current working directory
                    'models/emotion_cnn_model.pt',
                    'models/emotion_cnn_model.pkl'
                ]
                logger.info(f"Searching for CNN model. Script dir: {script_dir}, Backend dir: {backend_dir}, CWD: {cwd}")
                for path in possible_paths:
                    abs_path = os.path.abspath(path)
                    exists = os.path.exists(abs_path)
                    logger.info(f"Checking for model at: {abs_path} (exists: {exists})")
                    if exists:
                        cnn_model_path = abs_path
                        logger.info(f"Found CNN model at: {cnn_model_path}")
                        break
                else:
                    logger.warning(f"CNN model not found. Searched in: {[os.path.abspath(p) for p in possible_paths[:5]]}")
                    logger.warning(f"All searched paths: {[os.path.abspath(p) for p in possible_paths]}")
            
            cnn_loaded = False
            logger.info(f"Attempting to load CNN model. Path: {cnn_model_path}, Exists: {cnn_model_path and os.path.exists(cnn_model_path) if cnn_model_path else False}")
            if cnn_model_path and os.path.exists(cnn_model_path):
                if cnn_model_path.endswith('.pt'):
                    # Load PyTorch model
                    try:
                        loaded_obj = torch.load(cnn_model_path, map_location=self.device)
                        
                        # Check if it's a state dict (dictionary/OrderedDict) or a model object
                        from collections import OrderedDict
                        is_state_dict = isinstance(loaded_obj, (dict, OrderedDict))
                        
                        if is_state_dict:
                            # It's a state dict - we need to reconstruct the model architecture
                            logger.info("Model file is a state dict. Reconstructing EfficientNetB0 architecture...")
                            logger.info(f"State dict keys (first 3): {list(loaded_obj.keys())[:3] if hasattr(loaded_obj, 'keys') else 'N/A'}")
                            
                            # Try to reconstruct EfficientNetB0 model from state dict
                            try:
                                try:
                                    from torchvision.models import efficientnet_b0
                                    import torch.nn as nn
                                except ImportError:
                                    logger.error("torchvision not available. Cannot reconstruct EfficientNetB0 model.")
                                    raise ValueError("Model file is a state dict. torchvision required but not installed.")
                                
                                # Create model architecture
                                model_arch = efficientnet_b0(pretrained=False)
                                
                                # Modify the classifier for emotion classification (7 emotions)
                                num_emotions = 7  # happy, sad, angry, fear, surprise, neutral, disgust
                                # EfficientNetB0 classifier uses 1280 input features
                                # We'll use 1280 directly (standard for EfficientNetB0)
                                classifier_in_features = 1280
                                model_arch.classifier = nn.Sequential(
                                    nn.Dropout(p=0.2, inplace=True),
                                    nn.Linear(classifier_in_features, num_emotions)
                                )
                                
                                # Try to load state dict (might fail if architecture doesn't match exactly)
                                try:
                                    # Convert OrderedDict to regular dict if needed
                                    state_dict = dict(loaded_obj) if isinstance(loaded_obj, OrderedDict) else loaded_obj
                                    
                                    # Try strict loading first
                                    try:
                                        model_arch.load_state_dict(state_dict, strict=True)
                                        logger.info("Successfully loaded state dict (strict mode)")
                                    except RuntimeError:
                                        # If strict fails, try non-strict (ignore missing/extra keys)
                                        logger.warning("Strict loading failed, trying non-strict mode...")
                                        missing, unexpected = model_arch.load_state_dict(state_dict, strict=False)
                                        if missing:
                                            logger.warning(f"Missing keys (ignored): {missing[:5]}...")
                                        if unexpected:
                                            logger.warning(f"Unexpected keys (ignored): {unexpected[:5]}...")
                                        logger.info("Successfully loaded state dict (non-strict mode)")
                                    
                                    self.cnn_model = model_arch
                                    logger.info("Successfully reconstructed EfficientNetB0 model from state dict")
                                    
                                except Exception as load_error:
                                    logger.error(f"Failed to load state dict into EfficientNetB0: {load_error}")
                                    logger.error("This might be a custom EfficientNet implementation.")
                                    logger.error("State dict keys suggest custom architecture - cannot reconstruct automatically.")
                                    raise ValueError(f"Cannot load state dict into standard EfficientNetB0: {load_error}")
                                    
                            except Exception as recon_error:
                                logger.error(f"Failed to reconstruct model: {recon_error}")
                                import traceback
                                logger.error(f"Traceback: {traceback.format_exc()}")
                                raise
                        else:
                            # It's a model object
                            self.cnn_model = loaded_obj
                        
                        # Try to set eval mode if it's a PyTorch module
                        if isinstance(self.cnn_model, torch.nn.Module):
                            self.cnn_model.eval()  # Set to evaluation mode
                            logger.info(f"PyTorch CNN model loaded from {cnn_model_path}")
                            logger.info(f"Model type: {type(self.cnn_model)}")
                            cnn_loaded = True
                        elif callable(self.cnn_model):
                            logger.info(f"Callable model loaded from {cnn_model_path}")
                            logger.info(f"Model type: {type(self.cnn_model)}")
                            cnn_loaded = True
                        else:
                            logger.warning(f"Loaded object is not a PyTorch module or callable: {type(self.cnn_model)}")
                            # Still try to use it
                            cnn_loaded = True
                    except Exception as e:
                        logger.error(f"Error loading PyTorch model: {str(e)}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        raise
                elif cnn_model_path.endswith('.pkl'):
                    # Load pickle model (sklearn, etc.)
                    with open(cnn_model_path, 'rb') as f:
                        self.cnn_model = pickle.load(f)
                    logger.info(f"Pickle CNN model loaded from {cnn_model_path}")
                    cnn_loaded = True
                else:
                    logger.warning(f"Unknown model file format: {cnn_model_path}")
            else:
                logger.warning(f"CNN model not found. Searched paths: {possible_paths if 'possible_paths' in locals() else 'CNN_MODEL_PATH env var'}")
            
            # Only set models_loaded if CNN was actually loaded
            if not cnn_loaded:
                logger.error("CNN model was not loaded. Emotion detection will not work.")
                raise ValueError("CNN model is required but was not loaded successfully.")
            
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
    
    def preprocess_face(self, image: np.ndarray, bbox: List[int], target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Extract and preprocess face region for emotion classification
        For EfficientNet models, use 224x224 RGB images
        For other models, use 48x48 grayscale
        """
        x1, y1, x2, y2 = bbox
        
        # Extract face region
        face_roi = image[y1:y2, x1:x2]
        
        # Determine target size based on model type
        if target_size is None:
            # Check if model is EfficientNet or similar (expects 224x224 RGB)
            if isinstance(self.cnn_model, torch.nn.Module):
                # Check model name/type to determine input size
                model_name = str(type(self.cnn_model)).lower()
                if 'efficientnet' in model_name or 'resnet' in model_name:
                    target_size = (224, 224)
                    use_rgb = True
                else:
                    target_size = (48, 48)
                    use_rgb = False
            else:
                target_size = (48, 48)
                use_rgb = False
        else:
            use_rgb = target_size[0] >= 224  # Assume RGB for larger sizes
        
        # Convert color space if needed
        if use_rgb:
            # EfficientNet expects RGB (not BGR)
            if len(face_roi.shape) == 3:
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            elif len(face_roi.shape) == 2:
                # Convert grayscale to RGB
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
        else:
            # Convert to grayscale for smaller models
            if len(face_roi.shape) == 3:
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size
        face_roi = cv2.resize(face_roi, target_size)
        
        # Normalize pixel values
        face_roi = face_roi.astype(np.float32) / 255.0
        
        # Reshape for model input
        if use_rgb:
            # RGB: (H, W, 3) -> (1, 3, H, W) for PyTorch
            face_roi = face_roi.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)
            face_roi = face_roi.reshape(1, 3, target_size[0], target_size[1])
        else:
            # Grayscale: (H, W) -> (1, 1, H, W) for PyTorch or (1, H, W, 1) for TensorFlow
            if isinstance(self.cnn_model, torch.nn.Module):
                face_roi = face_roi.reshape(1, 1, target_size[0], target_size[1])
            else:
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
            # Check model type in order: PyTorch -> Scikit-learn -> Keras/TensorFlow
            
            # PyTorch model check (must be first, before checking __call__)
            if isinstance(self.cnn_model, torch.nn.Module):
                with torch.no_grad():
                    # Ensure input is in correct format (N, C, H, W)
                    if len(face_image.shape) == 4:
                        input_tensor = torch.from_numpy(face_image).float()
                    elif len(face_image.shape) == 3:
                        input_tensor = torch.from_numpy(face_image).float().unsqueeze(0)
                    else:
                        raise ValueError(f"Unexpected input shape: {face_image.shape}")
                    
                    if self.device == 'cuda':
                        input_tensor = input_tensor.to(self.device)
                        self.cnn_model = self.cnn_model.to(self.device)
                    
                    output = self.cnn_model(input_tensor)
                    
                    # Handle different output formats
                    if isinstance(output, torch.Tensor):
                        if output.dim() > 1:
                            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                        else:
                            probabilities = torch.softmax(output, dim=0).cpu().numpy()
                    else:
                        probabilities = output[0] if isinstance(output, (list, tuple)) else output
                    
                    # Default emotion classes (adjust based on your model)
                    emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            
            # Scikit-learn model check
            elif hasattr(self.cnn_model, 'predict_proba'):
                probabilities = self.cnn_model.predict_proba(face_image.reshape(1, -1))[0]
                emotion_classes = self.cnn_model.classes_
            
            # Keras/TensorFlow model check
            elif hasattr(self.cnn_model, 'predict') and callable(getattr(self.cnn_model, 'predict', None)):
                probabilities = self.cnn_model.predict(face_image, verbose=0)[0]
                # Define emotion classes based on your training
                emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            
            else:
                raise ValueError(f"Unknown model type: {type(self.cnn_model)}. Expected PyTorch nn.Module, scikit-learn, or Keras/TensorFlow model.")
            
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

