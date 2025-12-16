import os
import pickle
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO  # type: ignore[reportPrivateImportUsage]
import torch
import logging
import warnings
from typing import Dict, Optional, Tuple, List, Union

# Suppress torchvision deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

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
        self.debug_dir = None  # Debug directory for saving face crops
        self.emotion_labels = None  # Emotion labels from model
        self.id2label = None  # ID to label mapping
        self.label2id = None  # Label to ID mapping
        self.transform = None  # TorchVision transforms for preprocessing
        logger.info(f"Using device: {self.device}")
    
    def load_models(self):
        """Load all trained models"""
        try:
            # Load YOLOv8 model for face detection (PyTorch .pt only)
            yolo_model_path = os.getenv('YOLO_MODEL_PATH', None)
            yolo_loaded = False
            
            # Check if env var path exists, if not, set to None to trigger search
            if yolo_model_path:
                # Resolve to absolute path for checking
                abs_yolo_path = os.path.abspath(yolo_model_path)
                if os.path.exists(abs_yolo_path):
                    # Use absolute path when file exists
                    yolo_model_path = abs_yolo_path
                    logger.info(f"Using YOLO_MODEL_PATH from env var: {yolo_model_path}")
                else:
                    logger.warning(f"YOLO_MODEL_PATH env var points to non-existent file: {yolo_model_path} (resolved: {abs_yolo_path}). Will search for model.")
                    yolo_model_path = None
            
            # Search for YOLO .pt models if not specified or path doesn't exist
            if not yolo_model_path:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                backend_dir = os.path.dirname(script_dir)
                cwd = os.getcwd()
                project_root = os.path.dirname(backend_dir) if os.path.basename(backend_dir) == 'backend' else os.path.dirname(cwd)
                
                # PRIORITY CHECK: Direct check for known model location (.pt files only)
                priority_paths = [
                    os.path.join(project_root, 'backend', 'backend', 'models', 'yolov8n.pt'),
                    os.path.join(cwd, 'backend', 'backend', 'models', 'yolov8n.pt'),
                    os.path.join(backend_dir, 'backend', 'models', 'yolov8n.pt'),
                    os.path.join(project_root, 'backend', 'backend', 'models', 'yolov8_face.pt'),
                    os.path.join(cwd, 'backend', 'backend', 'models', 'yolov8_face.pt'),
                    os.path.join(backend_dir, 'backend', 'models', 'yolov8_face.pt'),
                ]
                
                print("\n" + "="*70)
                print("YOLO MODEL SEARCH - Starting Priority Check (.pt files only)")
                print("="*70)
                logger.info("="*70)
                logger.info("YOLO MODEL SEARCH - Starting Priority Check (.pt files only)")
                logger.info("="*70)
                
                for priority_path in priority_paths:
                    abs_priority = os.path.abspath(priority_path)
                    exists = os.path.exists(abs_priority)
                    print(f"Priority check: {abs_priority}")
                    print(f"  → Exists: {exists}")
                    logger.info(f"Priority check: {abs_priority} (exists: {exists})")
                    if exists:
                        yolo_model_path = abs_priority
                        file_size_mb = os.path.getsize(abs_priority) / (1024 * 1024)
                        print(f"✓ FOUND YOLO MODEL (Priority): {yolo_model_path}")
                        print(f"  → File size: {file_size_mb:.2f} MB")
                        logger.info(f"✓ FOUND YOLO MODEL (Priority): {yolo_model_path}")
                        logger.info(f"  File size: {file_size_mb:.2f} MB")
                        break
                
                # If priority check didn't find it, do full search
                if not yolo_model_path:
                    print("\nPriority check failed. Starting full search...")
                    logger.info("Priority check failed. Starting full search...")
                    
                    yolo_paths = [
                        # PyTorch face detection models (.pt files only)
                        os.path.join(cwd, 'backend', 'models', 'yolov8n.pt'),
                        os.path.join(backend_dir, 'backend', 'models', 'yolov8n.pt'),
                        os.path.join(os.path.dirname(cwd), 'backend', 'backend', 'models', 'yolov8n.pt'),
                        os.path.join(cwd, 'models', 'yolov8n.pt'),
                        'backend/models/yolov8n.pt',
                        'backend/backend/models/yolov8n.pt',
                        './backend/models/yolov8n.pt',
                        '../backend/models/yolov8n.pt',
                        'models/yolov8n.pt',
                        os.path.join(cwd, 'backend', 'models', 'yolov8_face.pt'),
                        os.path.join(backend_dir, 'models', 'yolov8_face.pt'),
                        'models/yolov8_face.pt',
                    ]
                    
                    print(f"Searching for YOLO model (.pt files only). CWD: {cwd}, Backend dir: {backend_dir}, Project root: {project_root}")
                    logger.info(f"Searching for YOLO model (.pt files only). CWD: {cwd}, Backend dir: {backend_dir}, Project root: {project_root}")
                    for i, path in enumerate(yolo_paths):
                        abs_path = os.path.abspath(path)
                        exists = os.path.exists(abs_path)
                        if i < 5 or exists:  # Log first 5 and any that exist
                            print(f"Checking YOLO path {i+1}/{len(yolo_paths)}: {abs_path} (exists: {exists})")
                            logger.info(f"Checking YOLO path {i+1}/{len(yolo_paths)}: {abs_path} (exists: {exists})")
                        if exists:
                            yolo_model_path = abs_path
                            file_size_mb = os.path.getsize(abs_path) / (1024 * 1024)
                            print(f"✓ Found YOLO model at: {yolo_model_path} ({file_size_mb:.2f} MB)")
                            logger.info(f"✓ Found YOLO model at: {yolo_model_path} ({file_size_mb:.2f} MB)")
                            break
                    else:
                        print(f"✗ YOLO model not found after checking {len(yolo_paths)} paths")
                        logger.warning(f"YOLO model not found after checking {len(yolo_paths)} paths")
                        logger.warning(f"First 5 paths: {[os.path.abspath(p) for p in yolo_paths[:5]]}")
            
            # Try to load YOLO PyTorch model (.pt files only)
            if yolo_model_path and os.path.exists(yolo_model_path):
                if not yolo_model_path.endswith('.pt'):
                    logger.warning(f"YOLO model path does not end with .pt: {yolo_model_path}. Skipping.")
                    yolo_model_path = None
                else:
                    file_size_mb = os.path.getsize(yolo_model_path) / (1024 * 1024)
                    print(f"\nAttempting to load YOLO PyTorch model from: {yolo_model_path}")
                    print(f"  → File size: {file_size_mb:.2f} MB")
                    logger.info(f"Attempting to load YOLO PyTorch model from: {yolo_model_path} ({file_size_mb:.2f} MB)")
                    try:
                        print(f"  → Loading PyTorch model...")
                        logger.info(f"Loading PyTorch YOLO model...")
                        self.yolo_model = YOLO(yolo_model_path)
                        print(f"✓ YOLOv8 PyTorch model loaded successfully!")
                        print(f"  → Model path: {yolo_model_path}")
                        logger.info(f"YOLOv8 PyTorch model loaded from {yolo_model_path}")
                        logger.info("YOLO face detector loaded (PyTorch .pt)")
                        yolo_loaded = True
                    except Exception as e:
                        print(f"✗ ERROR loading PyTorch YOLO model: {str(e)}")
                        logger.error(f"Error loading PyTorch YOLO model: {str(e)}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        logger.warning("Falling back to default YOLOv8n")
            
            if not yolo_loaded:
                # Use default YOLOv8n if custom model not found
                print("\n⚠ WARNING: Custom YOLO model not found, using default YOLOv8n")
                logger.warning(f"Custom YOLO model not found, using default YOLOv8n")
                try:
                    self.yolo_model = YOLO('yolov8n.pt')
                    print("  → Default YOLOv8n model loaded")
                    logger.info("Default YOLOv8n model loaded")
                    logger.info("YOLO face detector loaded (PyTorch .pt)")
                except Exception as e:
                    print(f"✗ ERROR loading default YOLOv8n: {str(e)}")
                    logger.error(f"Error loading default YOLOv8n: {str(e)}")
            
            # Final summary
            print("\n" + "="*70)
            print("YOLO MODEL LOADING SUMMARY")
            print("="*70)
            if self.yolo_model:
                model_name = "yolov8n.pt" if yolo_model_path and 'face' in yolo_model_path else "yolov8_face.pt" if yolo_model_path and 'face' in yolo_model_path else "yolov8n.pt (default)"
                print(f"✓ Status: LOADED (PyTorch)")
                print(f"  → Model: {model_name}")
                print(f"  → Path: {yolo_model_path if yolo_model_path else 'yolov8n.pt (downloaded)'}")
                print(f"  → Device: {self.device}")
            else:
                print(f"✗ Status: FAILED")
                print(f"  → No YOLO model loaded")
            print("="*70 + "\n")
            logger.info("="*70)
            logger.info("YOLO MODEL LOADING SUMMARY")
            if self.yolo_model:
                logger.info(f"Status: LOADED (PyTorch) - {yolo_model_path if yolo_model_path else 'yolov8n.pt (default)'}")
                logger.info("YOLO face detector loaded (PyTorch .pt)")
            else:
                logger.error("Status: FAILED - No YOLO model loaded")
            logger.info("="*70)
            
            # ----------------- LOAD EMOTION MODEL (TorchVision ResNet50) -----------------
            from torchvision import models
            import torch.nn as nn
            
            cnn_model_path = os.getenv("CNN_MODEL_PATH", "models/resnet50_emotion_model_CPU.pkl")
            self.model_type = "resnet50"
            self.use_onnx = False
            
            # Search for .pkl file if path doesn't exist
            if not os.path.exists(cnn_model_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                backend_dir = os.path.dirname(script_dir)
                cwd = os.getcwd()
                
                search_paths = [
                    os.path.join(backend_dir, "models", "resnet50_emotion_model_CPU.pkl"),
                    os.path.join(backend_dir, "backend", "models", "resnet50_emotion_model_CPU.pkl"),
                    os.path.join(cwd, "backend", "models", "resnet50_emotion_model_CPU.pkl"),
                    os.path.join(cwd, "backend", "backend", "models", "resnet50_emotion_model_CPU.pkl"),
                    os.path.join(cwd, "models", "resnet50_emotion_model_CPU.pkl"),
                    "models/resnet50_emotion_model_CPU.pkl",
                    "backend/models/resnet50_emotion_model_CPU.pkl",
                    "backend/backend/models/resnet50_emotion_model_CPU.pkl",
                    # Also try without _CPU suffix
                    os.path.join(backend_dir, "models", "resnet50_emotion_model.pkl"),
                    os.path.join(backend_dir, "backend", "models", "resnet50_emotion_model.pkl"),
                    os.path.join(cwd, "backend", "models", "resnet50_emotion_model.pkl"),
                    os.path.join(cwd, "backend", "backend", "models", "resnet50_emotion_model.pkl"),
                    os.path.join(cwd, "models", "resnet50_emotion_model.pkl"),
                    "models/resnet50_emotion_model.pkl",
                    "backend/models/resnet50_emotion_model.pkl",
                    "backend/backend/models/resnet50_emotion_model.pkl",
                ]
                
                for path in search_paths:
                    abs_path = os.path.abspath(path)
                    if os.path.exists(abs_path):
                        cnn_model_path = abs_path
                        logger.info(f"Found CNN model at: {cnn_model_path}")
                        break
                else:
                    logger.error(f"CNN model pkl not found. Searched: {search_paths}")
                    raise FileNotFoundError(f"CNN model pkl not found at {cnn_model_path}")
            
            if not os.path.exists(cnn_model_path):
                raise FileNotFoundError(f"CNN model pkl not found at {cnn_model_path}")
            
            # Load pickle file (use torch.load to handle CUDA->CPU mapping)
            # If the model was saved on CUDA but we're on CPU, map_location='cpu' will handle it
            try:
                payload = torch.load(cnn_model_path, map_location='cpu', weights_only=False)
            except Exception as e:
                # Fallback to pickle if torch.load fails (for non-PyTorch pickle files)
                logger.warning(f"torch.load failed, trying pickle.load: {str(e)}")
                with open(cnn_model_path, "rb") as f:
                    payload = pickle.load(f)
            
            state_dict = payload["state_dict"]
            labels = payload["labels"]
            id2label = payload["id2label"]
            label2id = payload["label2id"]
            
            self.emotion_labels = labels
            self.id2label = id2label
            self.label2id = label2id
            
            # Create TorchVision ResNet50 model
            num_classes = len(labels)
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            self.cnn_model = model
            cnn_loaded = True
            
            # Initialize TorchVision transforms for preprocessing
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            ])
            
            print("\n===== CNN MODEL LOADED (TorchVision ResNet50) =====")
            print("Labels:", labels)
            print("Device:", self.device)
            print("=============================================\n")
            logger.info("===== CNN MODEL LOADED (TorchVision ResNet50) =====")
            logger.info(f"Loaded ResNet50 emotion model with {len(labels)} classes")
            logger.info(f"Labels: {labels}")
            logger.info(f"Device: {self.device}")
            logger.info("=============================================")
            
            # Only set models_loaded if CNN was actually loaded
            if not cnn_loaded:
                logger.error("CNN model was not loaded. Emotion detection will not work.")
                raise ValueError("CNN model is required but was not loaded successfully.")
            
            # CNN Model Summary
            if cnn_model_path:
                file_size_mb = os.path.getsize(cnn_model_path) / (1024 * 1024) if os.path.exists(cnn_model_path) else 0
                print("\n" + "="*70)
                print("CNN MODEL LOADING SUMMARY")
                print("="*70)
                print(f"✓ Status: LOADED")
                print(f"  → Model: {self.model_type}")
                print(f"  → Path: {cnn_model_path}")
                if file_size_mb > 0:
                    print(f"  → File size: {file_size_mb:.2f} MB")
                print(f"  → Device: {self.device}")
                print("="*70 + "\n")
                logger.info("="*70)
                logger.info(f"CNN MODEL LOADING SUMMARY - {self.model_type} at {cnn_model_path}")
                logger.info("="*70)
            
            # Load additional models if specified
            self._load_additional_models()
            
            # Create debug directory
            self._setup_debug_dir()
            
            self.models_loaded = True
            print("\n" + "="*70)
            print("✓ ALL MODELS LOADED SUCCESSFULLY")
            print("="*70 + "\n")
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
                    model_name = os.path.basename(model_path.strip()).replace('.pkl', '')
                    # Use torch.load to handle CUDA->CPU mapping
                    try:
                        self.additional_models[model_name] = torch.load(model_path.strip(), map_location='cpu', weights_only=False)
                    except Exception as e:
                        # Fallback to pickle if torch.load fails
                        logger.warning(f"torch.load failed for {model_name}, trying pickle.load: {str(e)}")
                        with open(model_path.strip(), 'rb') as f:
                            self.additional_models[model_name] = pickle.load(f)
                    logger.info(f"Additional model loaded: {model_name}")
                except Exception as e:
                    logger.error(f"Error loading additional model {model_path}: {str(e)}")
    
    
    def _setup_debug_dir(self):
        """Create debug directory for saving face crop previews"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            backend_dir = os.path.dirname(script_dir)
            self.debug_dir = os.path.join(backend_dir, 'debug')
            os.makedirs(self.debug_dir, exist_ok=True)
            logger.info(f"Debug directory created/verified: {self.debug_dir}")
        except Exception as e:
            logger.warning(f"Could not create debug directory: {str(e)}")
            self.debug_dir = None
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in image using YOLOv8 PyTorch (.pt)
        Returns list of face bounding boxes and confidence scores
        """
        logger.info(f"[FaceDetection] Starting face detection. Image shape: {image.shape}, dtype: {image.dtype}")
        logger.info(f"[FaceDetection] Image value range: min={image.min()}, max={image.max()}")
        logger.info(f"[FaceDetection] Using PyTorch YOLO model")
        
        if self.yolo_model is None:
            logger.error("[FaceDetection] YOLOv8 PyTorch model not loaded")
            raise ValueError("YOLOv8 model not loaded")
        
        faces = self._detect_faces_pytorch(image)
        
        logger.info(f"[FaceDetection] Detected {len(faces)} face(s)")
        if len(faces) > 0:
            for i, face in enumerate(faces):
                logger.info(f"[FaceDetection] Face {i+1}: bbox={face.get('bbox', 'N/A')}, confidence={face.get('confidence', 'N/A')}")
        else:
            logger.warning("[FaceDetection] NO FACES DETECTED - This may indicate:")
            logger.warning("[FaceDetection] 1. YOLO model is not working correctly")
            logger.warning("[FaceDetection] 2. Image quality is too poor")
            logger.warning("[FaceDetection] 3. Face is not visible or too small")
            logger.warning("[FaceDetection] 4. Confidence threshold is too high")
        return faces
    
    def _detect_faces_pytorch(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using PyTorch YOLO model"""
        try:
            # Type guard: ensure yolo_model is not None
            if self.yolo_model is None:
                raise ValueError("YOLOv8 PyTorch model is not initialized")
            
            yolo_model = self.yolo_model  # Type narrowing
            
            # Run YOLOv8 inference
            results = yolo_model(image, conf=0.3, verbose=False)  # type: ignore
            
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
            logger.error(f"Error in PyTorch face detection: {str(e)}")
            return []
    
    def preprocess_face(self, image: np.ndarray, bbox: List[int], target_size: Optional[Tuple[int, int]] = None) -> Tuple[Optional[Union[np.ndarray, torch.Tensor]], Optional[str]]:
        """
        Extract and preprocess face region for emotion classification
        For ResNet50 models, use 224x224 RGB images
        Returns: (preprocessed_face, error_message) where error_message is None if valid
        """
        x1, y1, x2, y2 = bbox
        
        # Validate and clamp bounding box (initial)
        x1_orig = max(0, int(x1))
        y1_orig = max(0, int(y1))
        x2_orig = min(image.shape[1], int(x2))
        y2_orig = min(image.shape[0], int(y2))
        
        # Ensure valid bounding box
        if x2_orig <= x1_orig or y2_orig <= y1_orig:
            logger.error(f"[FacePreprocessing] Invalid bounding box: ({x1_orig}, {y1_orig}, {x2_orig}, {y2_orig})")
            return None, "face_invalid"
        
        # Calculate face dimensions
        face_width = x2_orig - x1_orig
        face_height = y2_orig - y1_orig
        
        # Add padding to bounding box - use stable 30% padding
        pad = int(0.3 * min(face_width, face_height))
        
        pad_w = pad
        pad_h = pad
        
        logger.info(f"[FacePreprocessing] Bbox: [{x1_orig}, {y1_orig}, {x2_orig}, {y2_orig}], size: {face_width}x{face_height}, padding: {pad_w}x{pad_h}")
        
        # Apply padding and clamp to image bounds
        x1 = max(0, x1_orig - pad_w)
        y1 = max(0, y1_orig - pad_h)
        x2 = min(image.shape[1], x2_orig + pad_w)
        y2 = min(image.shape[0], y2_orig + pad_h)
        
        # Extract face region with padding
        face_roi = image[y1:y2, x1:x2]
        
        # Check if face_roi is empty
        if face_roi.size == 0:
            logger.error(f"[FacePreprocessing] Empty face region extracted from bbox: ({x1}, {y1}, {x2}, {y2})")
            return None, "face_invalid"
        
        # Validate face crop quality - add variance and brightness checks
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        variance = np.var(gray)
        mean = np.mean(gray)
        
        if variance < 10:
            print(f"[EmotionDetection] ⚠️ Very low face variance ({variance:.4f}) — image may be too blurry/dark")
            logger.warning(f"[EmotionDetection] ⚠️ Very low face variance ({variance:.4f}) — image may be too blurry/dark")
        
        if mean < 30:
            print(f"[EmotionDetection] ⚠️ Face too dark (mean={mean:.1f}) — lighting issue")
            logger.warning(f"[EmotionDetection] ⚠️ Face too dark (mean={mean:.1f}) — lighting issue")
        
        face_std = float(face_roi.std())
        face_brightness = float(face_roi.mean())
        
        logger.info(f"[FacePreprocessing] Face crop - size: {face_roi.shape}, std: {face_std:.2f}, brightness: {face_brightness:.2f}, variance: {variance:.2f}")
        
        # Reject invalid crops
        if face_std < 5 or face_brightness < 20:
            logger.warning(f"[FacePreprocessing] Face crop rejected - std: {face_std:.2f} (< 5) or brightness: {face_brightness:.2f} (< 20)")
            return None, "face_invalid"
        
        # Save debug preview
        if self.debug_dir:
            try:
                import uuid
                debug_filename = f"face_crop_{uuid.uuid4().hex[:8]}.jpg"
                debug_path = os.path.join(self.debug_dir, debug_filename)
                cv2.imwrite(debug_path, face_roi)
                logger.info(f"[FacePreprocessing] Debug preview saved: {debug_path}")
            except Exception as e:
                logger.debug(f"[FacePreprocessing] Could not save debug preview: {str(e)}")
        
        # Use TorchVision transforms for ResNet50 preprocessing
        if self.transform is not None:
            # Convert BGR to RGB
            rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            # Apply transforms: Resize, ToTensor, Normalize
            # ToTensor() converts PIL Image to tensor
            transformed = self.transform(pil_img)  # Returns torch.Tensor [C, H, W]
            # Ensure it's a tensor and add batch dimension
            if isinstance(transformed, torch.Tensor):
                tensor = transformed.unsqueeze(0).float()  # Add batch dimension: [1, C, H, W]
            else:
                # Fallback: convert to tensor if needed
                tensor = torch.from_numpy(np.array(transformed)).unsqueeze(0).float()
            logger.info(f"[FacePreprocessing] Using TorchVision transforms - output shape: {tensor.shape}")
            return tensor, None
        
        # Fallback: Manual preprocessing if transform not available
        # ResNet50 expects 224x224 RGB images
        target_size = (224, 224) if target_size is None else target_size
        
        # Convert BGR to RGB
        if len(face_roi.shape) == 3:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        elif len(face_roi.shape) == 2:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
        
        # Resize to target size
        face_roi = cv2.resize(face_roi, target_size)
        
        # Normalize pixel values using ImageNet normalization
        face_roi = face_roi.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        face_roi = (face_roi - mean) / std
        
        # Reshape for PyTorch: (H, W, 3) -> (1, 3, H, W)
        face_roi = face_roi.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        face_roi = face_roi.reshape(1, 3, target_size[0], target_size[1])
        
        return face_roi, None
    
    def predict_emotion(self, face_image: Union[np.ndarray, torch.Tensor]) -> Union[Dict[str, float], str]:
        """
        Predict emotion from preprocessed face image using CNN model
        Returns dictionary of emotion probabilities or "model_not_confident" (as string) if validation fails
        """
        if self.cnn_model is None:
            logger.error("[EmotionPrediction] CNN model not loaded")
            return "model_not_confident"
        
        try:
            # Use emotion_labels from model as the only source of truth
            if not hasattr(self, 'emotion_labels') or not self.emotion_labels:
                logger.error("[EmotionPrediction] emotion_labels not loaded from model")
                return "model_not_confident"
            
            emotion_classes = self.emotion_labels
            
            # Predict using the loaded model
            # Check model type in order: ONNX -> PyTorch -> Scikit-learn -> Keras/TensorFlow
            
            if self.use_onnx:
                # ONNX model inference (for emotion model - not YOLO)
                import onnxruntime as ort
                
                # Type guard: ensure cnn_model is an InferenceSession
                if not isinstance(self.cnn_model, ort.InferenceSession):
                    raise ValueError("CNN model is not an ONNX InferenceSession")
                
                onnx_session = self.cnn_model  # Type narrowing
                
                # Get input and output names
                input_name = onnx_session.get_inputs()[0].name
                output_name = onnx_session.get_outputs()[0].name
                
                # Ensure input is numpy array and float32
                if isinstance(face_image, torch.Tensor):
                    face_image_np = face_image.cpu().numpy()
                else:
                    face_image_np = face_image
                
                if face_image_np.dtype != np.float32:
                    face_image_np = face_image_np.astype(np.float32)
                
                # Run inference
                logger.debug(f"[EmotionPrediction] Running ONNX inference - input shape: {face_image.shape}, dtype: {face_image.dtype}")
                outputs = onnx_session.run([output_name], {input_name: face_image})
                output = np.asarray(outputs[0])  # Ensure numpy array
                logger.debug(f"[EmotionPrediction] ONNX inference complete - output shape: {output.shape}, dtype: {output.dtype}, min: {output.min():.4f}, max: {output.max():.4f}, mean: {output.mean():.4f}")
                
                # Check if output is already probabilities (sums to ~1.0 and all values in [0,1])
                output_flat = output.flatten() if output.ndim > 1 else output
                output_sum = float(output_flat.sum())
                output_in_range = bool(np.all((output_flat >= 0) & (output_flat <= 1)))
                
                if abs(output_sum - 1.0) < 0.1 and output_in_range:
                    # Output is already probabilities (softmax already applied)
                    logger.debug(f"[EmotionPrediction] ONNX output appears to already have softmax applied (sum={output_sum:.4f})")
                    if output.ndim > 1:
                        probabilities = output[0]
                    else:
                        probabilities = output
                else:
                    # Apply softmax if needed (ONNX models may or may not include softmax)
                    logger.debug(f"[EmotionPrediction] Applying softmax to ONNX output (sum={output_sum:.4f}, in_range={output_in_range})")
                    if output.ndim > 1:
                        # Apply softmax
                        exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
                        probabilities = (exp_output / exp_output.sum(axis=1, keepdims=True))[0]
                    else:
                        exp_output = np.exp(output - np.max(output))
                        probabilities = exp_output / exp_output.sum()
                
                prob_std = float(probabilities.std())
                prob_range = float(probabilities.max() - probabilities.min())
                logger.debug(f"[EmotionPrediction] Final probabilities - sum: {probabilities.sum():.4f}, min: {probabilities.min():.4f}, max: {probabilities.max():.4f}, std: {prob_std:.4f}, range: {prob_range:.4f}")
                
                # Validate emotion logits after softmax
                if prob_std < 0.01 or prob_range < 0.02:
                    logger.warning(f"[EmotionPrediction] Model not confident - std: {prob_std:.4f} (< 0.01) or range: {prob_range:.4f} (< 0.02)")
                    return "model_not_confident"
                
                # Create emotion dictionary from ONNX output
                # Ensure we have the same number of classes as probabilities
                if len(emotion_classes) != len(probabilities):
                    logger.error(f"[EmotionPrediction] Mismatch: {len(emotion_classes)} emotion classes but {len(probabilities)} probabilities")
                    return "model_not_confident"
                
                emotions = {emo: float(probabilities[i]) for i, emo in enumerate(emotion_classes)}
                
                # Ensure all 8 emotions are present (set missing ones to 0.0)
                for emotion in emotion_classes:
                    if emotion not in emotions:
                        emotions[emotion] = 0.0
                
                return emotions
            
            # PyTorch model check (TorchVision ResNet50)
            elif isinstance(self.cnn_model, torch.nn.Module):
                with torch.no_grad():
                    # Handle input conversion - support both numpy arrays and torch tensors
                    if isinstance(face_image, np.ndarray):
                        face_tensor = torch.from_numpy(face_image).float()
                    elif isinstance(face_image, torch.Tensor):
                        face_tensor = face_image.float()
                    else:
                        logger.error(f"[EmotionPrediction] Unexpected face_image type: {type(face_image)}")
                        return "model_not_confident"
                    
                    # Ensure batch dimension (should be 4D: [batch, channels, height, width])
                    if face_tensor.ndim == 3:
                        face_tensor = face_tensor.unsqueeze(0)
                    elif face_tensor.ndim != 4:
                        logger.error(f"[EmotionPrediction] Unexpected tensor shape: {face_tensor.shape}, expected 3D or 4D")
                        return "model_not_confident"
                    
                    # Move to device and run inference
                    face_tensor = face_tensor.to(self.device)
                    logits = self.cnn_model(face_tensor)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                
                # Use emotion_labels from model loading (already set above)
                # Ensure we have the same number of classes as probabilities
                if len(emotion_classes) != len(probs):
                    logger.error(f"[EmotionPrediction] Mismatch: {len(emotion_classes)} emotion classes but {len(probs)} probabilities")
                    return "model_not_confident"
                
                emotions = {emo: float(probs[i]) for i, emo in enumerate(emotion_classes)}
                
                # Ensure all 8 emotions are present (set missing ones to 0.0)
                for emotion in emotion_classes:
                    if emotion not in emotions:
                        emotions[emotion] = 0.0
                
                # Validate probabilities
                prob_std = float(np.std(list(emotions.values())))
                prob_range = float(max(emotions.values()) - min(emotions.values()))
                
                if prob_std < 0.01:
                    print("[EmotionDetection] ⚠️ Probabilities nearly uniform — preprocessing/model mismatch")
                    logger.warning(f"[EmotionPrediction] Probabilities nearly uniform (std: {prob_std:.4f}) — preprocessing/model mismatch")
                
                return emotions
            
            # Scikit-learn model check
            elif hasattr(self.cnn_model, 'predict_proba') and callable(getattr(self.cnn_model, 'predict_proba', None)):
                probabilities = self.cnn_model.predict_proba(face_image.reshape(1, -1))[0]  # type: ignore
                # Use emotion_labels from model, not model.classes_
                # emotion_classes already set above from self.emotion_labels
                
                # Validate emotion logits
                prob_std = float(probabilities.std())
                prob_range = float(probabilities.max() - probabilities.min())
                if prob_std < 0.01 or prob_range < 0.02:
                    logger.warning(f"[EmotionPrediction] Model not confident - std: {prob_std:.4f} (< 0.01) or range: {prob_range:.4f} (< 0.02)")
                    return "model_not_confident"
            
            # Keras/TensorFlow model check
            elif hasattr(self.cnn_model, 'predict') and callable(getattr(self.cnn_model, 'predict', None)):
                probabilities = self.cnn_model.predict(face_image, verbose=0)[0]  # type: ignore
                
                # Validate emotion logits
                prob_std = float(probabilities.std())
                prob_range = float(probabilities.max() - probabilities.min())
                if prob_std < 0.01 or prob_range < 0.02:
                    logger.warning(f"[EmotionPrediction] Model not confident - std: {prob_std:.4f} (< 0.01) or range: {prob_range:.4f} (< 0.02)")
                    return "model_not_confident"
            
            else:
                logger.error(f"[EmotionPrediction] Unknown model type: {type(self.cnn_model)}")
                return "model_not_confident"
            
            # Create emotion dictionary (for non-PyTorch models)
            emotion_dict = {}
            for i, emotion in enumerate(emotion_classes):
                if i < len(probabilities):
                    emotion_dict[emotion] = float(probabilities[i])
            
            # Ensure all 8 emotions are present (set missing ones to 0.0)
            for emotion in emotion_classes:
                if emotion not in emotion_dict:
                    emotion_dict[emotion] = 0.0
            
            return emotion_dict
            
        except Exception as e:
            logger.error(f"Error in emotion prediction: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Safe fallback
            return "model_not_confident"
    
    def predict_emotion_vector(self, face_tensor: np.ndarray) -> Dict[str, float]:
        """
        Predict emotion probabilities from preprocessed face tensor
        Returns dictionary of emotion probabilities
        """
        emotions = self.predict_emotion(face_tensor)
        if isinstance(emotions, str):
            # Return safe fallback distribution if model not confident
            # This prevents artificially low or misleading scores
            fallback_emotions = {
                'neutral': 0.7,
                'focus': 0.2,
                'boredom': 0.1
            }
            
            # Ensure all 8 emotions are present (set missing ones to 0.0)
            if hasattr(self, 'emotion_labels') and self.emotion_labels:
                for emotion in self.emotion_labels:
                    if emotion not in fallback_emotions:
                        fallback_emotions[emotion] = 0.0
            else:
                # If emotion_labels not available, use all 8 emotions
                all_emotions = ['boredom', 'confusion', 'focus', 'frustration', 'happy', 'neutral', 'sleepy', 'surprise']
                for emotion in all_emotions:
                    if emotion not in fallback_emotions:
                        fallback_emotions[emotion] = 0.0
            
            return fallback_emotions
        return emotions
    
    def _get_safe_fallback_emotions(self) -> Dict[str, float]:
        """
        Get safe fallback emotion distribution when model is not confident.
        This prevents artificially low or misleading concentration scores.
        """
        fallback_emotions = {
            'neutral': 0.7,
            'focus': 0.2,
            'boredom': 0.1
        }
        
        # Ensure all 8 emotions are present (set missing ones to 0.0)
        if hasattr(self, 'emotion_labels') and self.emotion_labels:
            for emotion in self.emotion_labels:
                if emotion not in fallback_emotions:
                    fallback_emotions[emotion] = 0.0
        else:
            # If emotion_labels not available, use all 8 emotions
            all_emotions = ['boredom', 'confusion', 'focus', 'frustration', 'happy', 'neutral', 'sleepy', 'surprise']
            for emotion in all_emotions:
                if emotion not in fallback_emotions:
                    fallback_emotions[emotion] = 0.0
        
        return fallback_emotions
    
    def compute_concentration(self, emotion_probs: Dict[str, float]) -> float:
        """
        Compute concentration score from emotion probabilities
        Uses the same logic as calculate_concentration_score
        """
        return self.calculate_concentration_score(emotion_probs)
    
    def detect_emotions_in_image(self, image: np.ndarray) -> List[Dict]:
        """
        Complete pipeline: detect faces and predict emotions
        Returns list of detections with emotions, engagement scores, and concentration scores
        """
        # Add image hash to ensure we're processing new images
        import hashlib
        image_hash = hashlib.md5(image.tobytes()).hexdigest()[:8]
        logger.info(f"[EmotionDetection] Starting emotion detection. Image shape: {image.shape}, hash: {image_hash}, mean: {image.mean():.2f}, std: {image.std():.2f}")
        
        # Detect faces
        faces = self.detect_faces(image)
        logger.info(f"[EmotionDetection] Face detection completed. Found {len(faces)} face(s)")
        
        if len(faces) == 0:
            logger.warning("[EmotionDetection] No faces detected in image. Returning empty list.")
        
        results = []
        for face in faces:
            bbox = face['bbox']
            
            # Preprocess face - now returns (face_image, error_message)
            face_image, preprocess_error = self.preprocess_face(image, bbox)
            
            # Handle preprocessing errors
            if preprocess_error is not None or face_image is None:
                logger.warning(f"[EmotionDetection] Face preprocessing failed: {preprocess_error}")
                # Safe fallback with proper 8-emotion distribution
                fallback_emotions = self._get_safe_fallback_emotions()
                dominant_emotion = max(fallback_emotions.items(), key=lambda x: x[1])
                concentration_score = self.calculate_concentration_score(fallback_emotions)
                emotion_state = self.map_emotion_to_state(dominant_emotion[0])
                
                results.append({
                    'bbox': bbox,
                    'emotions': fallback_emotions,
                    'dominant_emotion': dominant_emotion[0],
                    'emotion_state': emotion_state,
                    'confidence': dominant_emotion[1],
                    'engagement_score': concentration_score / 100.0,  # Convert to 0-1 range for compatibility
                    'concentration_score': concentration_score,
                    'face_confidence': face['confidence'],
                    'error': preprocess_error
                })
                continue
            
            # Log face crop info for debugging
            if isinstance(face_image, np.ndarray):
                logger.info(f"[EmotionDetection] Face crop - shape: {face_image.shape}, dtype: {face_image.dtype}, mean: {face_image.mean():.4f}, std: {face_image.std():.4f}, min: {face_image.min():.4f}, max: {face_image.max():.4f}")
                # Check if face image is valid (not all zeros or all same value)
                if face_image.std() < 1.0:
                    logger.warning(f"[EmotionDetection] ⚠️ Face crop has very low std ({face_image.std():.4f}) - may be invalid or poorly cropped")
                if np.all(face_image == face_image.flat[0]):
                    logger.error(f"[EmotionDetection] ❌ Face crop is uniform (all values = {face_image.flat[0]:.4f}) - preprocessing may have failed")
            elif hasattr(face_image, 'shape'):
                logger.info(f"[EmotionDetection] Face crop - shape: {face_image.shape}, dtype: {face_image.dtype}")
            
            # Predict emotion - can return "model_not_confident"
            emotions = self.predict_emotion(face_image)
            
            # Handle model confidence errors
            if emotions == "model_not_confident" or not isinstance(emotions, dict):
                logger.warning("[EmotionDetection] Model not confident, using safe fallback")
                # Safe fallback with proper 8-emotion distribution
                fallback_emotions = self._get_safe_fallback_emotions()
                dominant_emotion = max(fallback_emotions.items(), key=lambda x: x[1])
                concentration_score = self.calculate_concentration_score(fallback_emotions)
                emotion_state = self.map_emotion_to_state(dominant_emotion[0])
                
                results.append({
                    'bbox': bbox,
                    'emotions': fallback_emotions,
                    'dominant_emotion': dominant_emotion[0],
                    'emotion_state': emotion_state,
                    'confidence': dominant_emotion[1],
                    'engagement_score': concentration_score / 100.0,  # Convert to 0-1 range for compatibility
                    'concentration_score': concentration_score,
                    'face_confidence': face['confidence'],
                    'error': 'model_not_confident'
                })
                continue
            
            # Type guard: ensure emotions is a dict
            if not isinstance(emotions, dict):
                logger.error(f"[EmotionDetection] Unexpected emotions type: {type(emotions)}")
                continue
            
            # Log all emotion probabilities to verify model is running inference
            emotion_probs_str = ', '.join([f"{emotion}: {prob:.4f}" for emotion, prob in sorted(emotions.items(), key=lambda x: x[1], reverse=True)])
            logger.info(f"[EmotionDetection] Emotion probabilities (all): {emotion_probs_str}")
            
            # Verify probabilities sum to ~1.0 (model inference working)
            prob_sum = sum(emotions.values())
            if abs(prob_sum - 1.0) > 0.1:
                logger.warning(f"[EmotionDetection] ⚠️ Emotion probabilities sum to {prob_sum:.4f} (expected ~1.0) - model may not be running correctly")
            else:
                logger.debug(f"[EmotionDetection] ✓ Emotion probabilities sum to {prob_sum:.4f} (model inference verified)")
            
            # Get dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            logger.info(f"[EmotionDetection] Dominant emotion: {dominant_emotion[0]} (confidence: {dominant_emotion[1]:.4f})")
            
            # Log if probabilities are too uniform (potential issue)
            prob_values = list(emotions.values())
            prob_std = np.std(prob_values) if len(prob_values) > 1 else 0.0
            prob_range = max(prob_values) - min(prob_values)
            
            # Check if probabilities are suspiciously uniform (all within 1% of each other)
            if prob_range < 0.01 or prob_std < 0.005:
                logger.warning(f"[EmotionDetection] ⚠️ Emotion probabilities are too uniform (std: {prob_std:.4f}, range: {prob_range:.4f}) - model may not be working correctly or input preprocessing may be wrong")
                logger.warning(f"[EmotionDetection] ⚠️ This suggests the model is not distinguishing between emotions properly")
            
            # Check if all probabilities are identical
            if len(set([round(p, 3) for p in prob_values])) == 1:
                logger.error(f"[EmotionDetection] ❌ All emotion probabilities are identical ({prob_values[0]:.4f}) - model may be returning cached/hardcoded values or not running inference")
            
            # Calculate scores
            engagement_score = self._calculate_engagement(emotions, dominant_emotion[0])
            concentration_score = self.calculate_concentration_score(emotions)
            logger.info(f"[EmotionDetection] Scores - engagement: {engagement_score:.2f}, concentration: {concentration_score:.2f}")
            logger.info(f"[EmotionDetection] Bbox coordinates: {bbox}")
            
            # Map to engagement states
            emotion_state = self.map_emotion_to_state(dominant_emotion[0])
            
            results.append({
                'bbox': bbox,
                'emotions': emotions,
                'dominant_emotion': dominant_emotion[0],
                'emotion_state': emotion_state,
                'confidence': dominant_emotion[1],
                'engagement_score': engagement_score,
                'concentration_score': concentration_score,
                'face_confidence': face['confidence']
            })
        
        return results
    
    def calculate_concentration_score(self, emotions: Dict[str, float]) -> float:
        """
        Calculate concentration score (0–100) from 8-emotion distribution.
        """
        
        attentive = (
            1.0 * emotions.get("focus", 0.0) +
            0.9 * emotions.get("happy", 0.0) +
            0.7 * emotions.get("surprise", 0.0) +
            0.4 * emotions.get("neutral", 0.0)
        )
        
        inattentive = (
            1.0 * emotions.get("sleepy", 0.0) +
            0.9 * emotions.get("boredom", 0.0) +
            0.85 * emotions.get("frustration", 0.0) +
            0.85 * emotions.get("confusion", 0.0)
        )
        
        total = attentive + inattentive
        if total == 0:
            return 0.0
        
        return round((attentive / total) * 100, 2)
    
    def _calculate_engagement(self, emotions: Dict[str, float], dominant: str) -> float:
        """
        Calculate engagement score based on emotions (0.0 to 1.0)
        Higher score = more engaged
        This is a legacy method, concentration_score should be used instead
        """
        # Map 8-emotion system to engagement categories
        positive_emotions = ['happy', 'surprise', 'focus']
        negative_emotions = ['frustration', 'confusion', 'boredom', 'sleepy']
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
        Map detected emotion to engagement state (8-emotion system)
        """
        emotion_mapping = {
            'focus': 'focused',
            'happy': 'focused',
            'surprise': 'focused',
            'neutral': 'neutral',
            'confusion': 'confused',
            'frustration': 'frustrated',
            'boredom': 'bored',
            'sleepy': 'bored'
        }
        return emotion_mapping.get(emotion, 'neutral')

