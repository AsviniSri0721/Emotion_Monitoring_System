import os
import pickle
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO  # type: ignore[reportPrivateImportUsage]
import torch
import logging
import warnings
from typing import Dict, Optional, Tuple, List

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
        self.yolo_onnx_session = None  # For ONNX YOLO models
        self.use_yolo_onnx = False
        self.model_config = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
    
    def load_models(self):
        """Load all trained models"""
        try:
            # Load YOLOv8 model for face detection
            # Priority: ONNX face model > PyTorch face model > default YOLOv8n
            yolo_model_path = os.getenv('YOLO_MODEL_PATH', None)
            yolo_loaded = False
            
            # Check if env var path exists, if not, set to None to trigger search
            if yolo_model_path:
                # Resolve to absolute path for checking
                abs_yolo_path = os.path.abspath(yolo_model_path)
                if not os.path.exists(abs_yolo_path):
                    logger.warning(f"YOLO_MODEL_PATH env var points to non-existent file: {yolo_model_path} (resolved: {abs_yolo_path}). Will search for model.")
                    yolo_model_path = None
                else:
                    yolo_model_path = abs_yolo_path  # Use absolute path
                    logger.info(f"Using YOLO_MODEL_PATH from env var: {yolo_model_path}")
            
            # Search for YOLO models if not specified or path doesn't exist
            if not yolo_model_path:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                backend_dir = os.path.dirname(script_dir)
                cwd = os.getcwd()
                project_root = os.path.dirname(backend_dir) if os.path.basename(backend_dir) == 'backend' else os.path.dirname(cwd)
                
                # PRIORITY CHECK: Direct check for known model location
                priority_paths = [
                    os.path.join(project_root, 'backend', 'backend', 'models', 'yolov8n-face.onnx'),
                    os.path.join(cwd, 'backend', 'backend', 'models', 'yolov8n-face.onnx'),
                    os.path.join(backend_dir, 'backend', 'models', 'yolov8n-face.onnx'),
                ]
                
                print("\n" + "="*70)
                print("YOLO MODEL SEARCH - Starting Priority Check")
                print("="*70)
                logger.info("="*70)
                logger.info("YOLO MODEL SEARCH - Starting Priority Check")
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
                        # ONNX face detection model (priority)
                        # When running from backend/, model is at backend/backend/models/
                        os.path.join(cwd, 'backend', 'models', 'yolov8n-face.onnx'),  # backend/backend/models/ when cwd=backend/
                        os.path.join(backend_dir, 'backend', 'models', 'yolov8n-face.onnx'),  # backend/backend/models/
                        os.path.join(os.path.dirname(cwd), 'backend', 'backend', 'models', 'yolov8n-face.onnx'),  # From project root
                        os.path.join(cwd, 'models', 'yolov8n-face.onnx'),  # backend/models/ when cwd=backend/
                        'backend/models/yolov8n-face.onnx',
                        'backend/backend/models/yolov8n-face.onnx',
                        './backend/models/yolov8n-face.onnx',  # Explicit relative
                        '../backend/models/yolov8n-face.onnx',  # From project root
                        'models/yolov8n-face.onnx',
                        # PyTorch models
                        os.path.join(cwd, 'models', 'yolov8_face.pt'),
                        os.path.join(backend_dir, 'models', 'yolov8_face.pt'),
                        'models/yolov8_face.pt',
                    ]
                    
                    print(f"Searching for YOLO model. CWD: {cwd}, Backend dir: {backend_dir}, Project root: {project_root}")
                    logger.info(f"Searching for YOLO model. CWD: {cwd}, Backend dir: {backend_dir}, Project root: {project_root}")
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
            
            # Try to load YOLO model
            if yolo_model_path and os.path.exists(yolo_model_path):
                file_size_mb = os.path.getsize(yolo_model_path) / (1024 * 1024)
                print(f"\nAttempting to load YOLO model from: {yolo_model_path}")
                print(f"  → File size: {file_size_mb:.2f} MB")
                logger.info(f"Attempting to load YOLO model from: {yolo_model_path} ({file_size_mb:.2f} MB)")
                if yolo_model_path.endswith('.onnx'):
                    # Load ONNX model
                    try:
                        import onnxruntime as ort
                        providers = ['CPUExecutionProvider']
                        if self.device == 'cuda':
                            try:
                                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                            except:
                                logger.warning("CUDA provider not available for YOLO, using CPU")
                        
                        print(f"  → Loading ONNX model with providers: {providers}")
                        logger.info(f"Loading ONNX model with providers: {providers}")
                        self.yolo_onnx_session = ort.InferenceSession(yolo_model_path, providers=providers)
                        self.use_yolo_onnx = True
                        print(f"✓ YOLOv8 ONNX model loaded successfully!")
                        print(f"  → Model path: {yolo_model_path}")
                        print(f"  → Device: {self.device}")
                        logger.info(f"✓ YOLOv8 ONNX model loaded successfully from {yolo_model_path}")
                        logger.info(f"  Device: {self.device}, Providers: {providers}")
                        yolo_loaded = True
                    except ImportError:
                        print("✗ ERROR: onnxruntime not available")
                        logger.error("onnxruntime not available. Install with: pip install onnxruntime")
                        logger.error("Please run: pip install onnxruntime")
                        logger.warning("Falling back to default YOLOv8n")
                    except Exception as e:
                        print(f"✗ ERROR loading ONNX YOLO model: {str(e)}")
                        logger.error(f"Error loading ONNX YOLO model: {str(e)}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        logger.warning("Falling back to default YOLOv8n")
                else:
                    # Load PyTorch model
                    try:
                        print(f"  → Loading PyTorch model...")
                        logger.info(f"Loading PyTorch YOLO model...")
                        self.yolo_model = YOLO(yolo_model_path)
                        print(f"✓ YOLOv8 PyTorch model loaded successfully!")
                        print(f"  → Model path: {yolo_model_path}")
                        logger.info(f"YOLOv8 PyTorch model loaded from {yolo_model_path}")
                        yolo_loaded = True
                    except Exception as e:
                        print(f"✗ ERROR loading PyTorch YOLO model: {str(e)}")
                        logger.error(f"Error loading PyTorch YOLO model: {str(e)}")
                        logger.warning("Falling back to default YOLOv8n")
            
            if not yolo_loaded:
                # Use default YOLOv8n if custom model not found
                print("\n⚠ WARNING: Custom YOLO model not found, using default YOLOv8n")
                logger.warning(f"Custom YOLO model not found, using default YOLOv8n")
                try:
                    self.yolo_model = YOLO('yolov8n.pt')
                    print("  → Default YOLOv8n model loaded")
                    logger.info("Default YOLOv8n model loaded")
                except Exception as e:
                    print(f"✗ ERROR loading default YOLOv8n: {str(e)}")
                    logger.error(f"Error loading default YOLOv8n: {str(e)}")
            
            # Final summary
            print("\n" + "="*70)
            print("YOLO MODEL LOADING SUMMARY")
            print("="*70)
            if self.use_yolo_onnx and self.yolo_onnx_session:
                print(f"✓ Status: LOADED (ONNX)")
                print(f"  → Model: yolov8n-face.onnx (Face Detection)")
                print(f"  → Path: {yolo_model_path if yolo_model_path else 'N/A'}")
                print(f"  → Device: {self.device}")
            elif self.yolo_model:
                model_name = "yolov8n-face.onnx" if yolo_model_path and 'face' in yolo_model_path else "yolov8_face.pt" if yolo_model_path and 'face' in yolo_model_path else "yolov8n.pt (default)"
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
            if self.use_yolo_onnx and self.yolo_onnx_session:
                logger.info(f"Status: LOADED (ONNX) - yolov8n-face.onnx at {yolo_model_path}")
            elif self.yolo_model:
                logger.info(f"Status: LOADED (PyTorch) - {yolo_model_path if yolo_model_path else 'yolov8n.pt (default)'}")
            else:
                logger.error("Status: FAILED - No YOLO model loaded")
            logger.info("="*70)
            
            # Load CNN emotion classification model
            # Priority: MobileNetV2 > EfficientNetB0 > other models
            # Try .pt (PyTorch) first, then .onnx, then .pkl (pickle)
            cnn_model_path = os.getenv('CNN_MODEL_PATH', None)
            self.model_type = None  # Track model type for preprocessing
            self.use_onnx = False
            
            # Check if env var path exists, if not, set to None to trigger search
            if cnn_model_path and not os.path.exists(cnn_model_path):
                logger.warning(f"CNN_MODEL_PATH env var points to non-existent file: {cnn_model_path}. Will search for model.")
                cnn_model_path = None
            
            # If not specified or path doesn't exist, look for common model file names (prioritize MobileNetV2)
            # Also check if we're in backend/ directory and model is in backend/backend/models/
            if not cnn_model_path:
                # Quick check: if running from backend/, check backend/backend/models/ directly
                quick_check = os.path.join(os.getcwd(), 'backend', 'models', 'mobilenetv2_best.pt')
                if os.path.exists(quick_check):
                    cnn_model_path = quick_check
                    logger.info(f"Found model via quick check: {cnn_model_path}")
            
            if not cnn_model_path:
                # Get the directory where this script is located
                script_dir = os.path.dirname(os.path.abspath(__file__))
                backend_dir = os.path.dirname(script_dir)  # Go up one level from services/
                project_root = os.path.dirname(backend_dir)  # Go up one more level
                cwd = os.getcwd()  # Current working directory
                
                possible_paths = [
                    # MobileNetV2 (priority) - check current directory first
                    os.path.join(cwd, 'backend', 'models', 'mobilenetv2_best.pt'),  # When running from backend/, model in backend/backend/models/
                    os.path.join(backend_dir, 'backend', 'models', 'mobilenetv2_best.pt'),  # Explicit backend/backend/models/
                    os.path.join(cwd, 'models', 'mobilenetv2_best.pt'),  # When running from backend/, model in backend/models/
                    os.path.join(backend_dir, 'models', 'mobilenetv2_best.pt'),
                    os.path.join(project_root, 'backend', 'models', 'mobilenetv2_best.pt'),
                    os.path.join(project_root, 'backend', 'backend', 'models', 'mobilenetv2_best.pt'),
                    os.path.join(cwd, 'models', 'mobilenetv2.onnx'),
                    os.path.join(backend_dir, 'models', 'mobilenetv2.onnx'),
                    os.path.join(backend_dir, 'backend', 'models', 'mobilenetv2.onnx'),
                    # EfficientNetB0 (fallback)
                    os.path.join(backend_dir, 'models', 'efficientnetb0_emotion_best.pt'),
                    os.path.join(backend_dir, 'backend', 'models', 'efficientnetb0_emotion_best.pt'),
                    os.path.join(project_root, 'backend', 'models', 'efficientnetb0_emotion_best.pt'),
                    os.path.join(cwd, 'backend', 'models', 'efficientnetb0_emotion_best.pt'),
                    os.path.join(cwd, 'models', 'efficientnetb0_emotion_best.pt'),
                    # Generic model names
                    os.path.join(backend_dir, 'models', 'emotion_cnn_model.pt'),
                    os.path.join(backend_dir, 'models', 'emotion_cnn_model.pkl'),
                    os.path.join(backend_dir, 'models', 'mobilenetv2_emotion_model.pkl'),
                    # Relative paths (from current working directory)
                    'models/mobilenetv2_best.pt',  # When running from backend/
                    './models/mobilenetv2_best.pt',  # Explicit relative
                    '../models/mobilenetv2_best.pt',  # From project root
                    'backend/models/mobilenetv2_best.pt',
                    'backend/backend/models/mobilenetv2_best.pt',
                    'models/efficientnetb0_emotion_best.pt',
                    'backend/models/efficientnetb0_emotion_best.pt',
                    'backend/backend/models/efficientnetb0_emotion_best.pt',
                    'models/emotion_cnn_model.pt',
                    'models/emotion_cnn_model.pkl'
                ]
                logger.info(f"Searching for CNN model. Script dir: {script_dir}, Backend dir: {backend_dir}, CWD: {cwd}")
                logger.info(f"Total paths to check: {len(possible_paths)}")
                for i, path in enumerate(possible_paths):
                    abs_path = os.path.abspath(path)
                    exists = os.path.exists(abs_path)
                    if i < 10 or exists:  # Log first 10 and any that exist
                        logger.info(f"Checking path {i+1}/{len(possible_paths)}: {abs_path} (exists: {exists})")
                    if exists:
                        cnn_model_path = abs_path
                        logger.info(f"✓ Found CNN model at: {cnn_model_path}")
                        break
                else:
                    logger.warning(f"CNN model not found after checking {len(possible_paths)} paths")
                    logger.warning(f"First 10 paths checked: {[os.path.abspath(p) for p in possible_paths[:10]]}")
                    logger.warning(f"Last 5 paths checked: {[os.path.abspath(p) for p in possible_paths[-5:]]}")
            
            cnn_loaded = False
            logger.info(f"Attempting to load CNN model. Path: {cnn_model_path}, Exists: {cnn_model_path and os.path.exists(cnn_model_path) if cnn_model_path else False}")
            if cnn_model_path and os.path.exists(cnn_model_path):
                if cnn_model_path.endswith('.onnx'):
                    # Load ONNX model
                    try:
                        try:
                            import onnxruntime as ort
                        except ImportError:
                            logger.error("onnxruntime not available. Install with: pip install onnxruntime")
                            raise ValueError("ONNX model requires onnxruntime package")
                        
                        # Create ONNX runtime session
                        providers = ['CPUExecutionProvider']  # Use CPU by default
                        if self.device == 'cuda':
                            try:
                                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                            except:
                                logger.warning("CUDA provider not available, using CPU")
                        
                        self.cnn_model = ort.InferenceSession(cnn_model_path, providers=providers)
                        self.use_onnx = True
                        self.model_type = 'mobilenetv2' if 'mobilenet' in cnn_model_path.lower() else 'unknown'
                        logger.info(f"ONNX model loaded from {cnn_model_path}")
                        logger.info(f"Model type: {self.model_type}")
                        cnn_loaded = True
                    except Exception as e:
                        logger.error(f"Error loading ONNX model: {str(e)}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        raise
                elif cnn_model_path.endswith('.pt'):
                    # Load PyTorch model
                    try:
                        loaded_obj = torch.load(cnn_model_path, map_location=self.device)
                        
                        # Check if it's a state dict (dictionary/OrderedDict) or a model object
                        from collections import OrderedDict
                        is_state_dict = isinstance(loaded_obj, (dict, OrderedDict))
                        
                        # Detect model type from filename
                        is_mobilenet = 'mobilenet' in cnn_model_path.lower()
                        is_efficientnet = 'efficientnet' in cnn_model_path.lower()
                        
                        if is_state_dict:
                            # It's a state dict - we need to reconstruct the model architecture
                            logger.info(f"Model file is a state dict. Reconstructing architecture...")
                            logger.info(f"State dict keys (first 3): {list(loaded_obj.keys())[:3] if hasattr(loaded_obj, 'keys') else 'N/A'}")
                            
                            try:
                                from torchvision.models import mobilenet_v2, efficientnet_b0
                                import torch.nn as nn
                            except ImportError:
                                logger.error("torchvision not available. Cannot reconstruct model architecture.")
                                raise ValueError("Model file is a state dict. torchvision required but not installed.")
                            
                            num_emotions = 7  # happy, sad, angry, fear, surprise, neutral, disgust
                            
                            if is_mobilenet:
                                # Reconstruct MobileNetV2
                                logger.info("Reconstructing MobileNetV2 architecture...")
                                model_arch = mobilenet_v2(pretrained=False)
                                # MobileNetV2 classifier uses 1280 input features
                                classifier_in_features = 1280
                                model_arch.classifier = nn.Sequential(
                                    nn.Dropout(p=0.2, inplace=True),
                                    nn.Linear(classifier_in_features, num_emotions)
                                )
                                self.model_type = 'mobilenetv2'
                            elif is_efficientnet:
                                # Reconstruct EfficientNetB0
                                logger.info("Reconstructing EfficientNetB0 architecture...")
                                model_arch = efficientnet_b0(pretrained=False)
                                classifier_in_features = 1280
                                model_arch.classifier = nn.Sequential(
                                    nn.Dropout(p=0.2, inplace=True),
                                    nn.Linear(classifier_in_features, num_emotions)
                                )
                                self.model_type = 'efficientnetb0'
                            else:
                                # Try to infer from state dict keys
                                keys = list(loaded_obj.keys()) if hasattr(loaded_obj, 'keys') else []
                                if any('features' in k and 'conv' in k for k in keys[:10]):
                                    logger.info("Inferring MobileNetV2 from state dict keys...")
                                    model_arch = mobilenet_v2(pretrained=False)
                                    classifier_in_features = 1280
                                    model_arch.classifier = nn.Sequential(
                                        nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(classifier_in_features, num_emotions)
                                    )
                                    self.model_type = 'mobilenetv2'
                                else:
                                    logger.warning("Cannot determine model architecture, trying EfficientNetB0...")
                                    model_arch = efficientnet_b0(pretrained=False)
                                    classifier_in_features = 1280
                                    model_arch.classifier = nn.Sequential(
                                        nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(classifier_in_features, num_emotions)
                                    )
                                    self.model_type = 'efficientnetb0'
                            
                            # Try to load state dict
                            try:
                                state_dict = dict(loaded_obj) if isinstance(loaded_obj, OrderedDict) else loaded_obj
                                try:
                                    model_arch.load_state_dict(state_dict, strict=True)
                                    logger.info("Successfully loaded state dict (strict mode)")
                                except RuntimeError:
                                    # Non-strict loading is expected for custom trained models
                                    logger.debug("Strict loading failed, trying non-strict mode (this is normal for custom models)...")
                                    missing, unexpected = model_arch.load_state_dict(state_dict, strict=False)
                                    if missing:
                                        logger.debug(f"Missing keys (ignored): {missing[:5]}...")
                                    if unexpected:
                                        logger.debug(f"Unexpected keys (ignored): {unexpected[:5]}...")
                                    logger.info("Successfully loaded state dict (non-strict mode)")
                                
                                self.cnn_model = model_arch
                                logger.info(f"Successfully reconstructed {self.model_type} model from state dict")
                                
                            except Exception as load_error:
                                logger.error(f"Failed to load state dict: {load_error}")
                                raise ValueError(f"Cannot load state dict: {load_error}")
                                
                        else:
                            # It's a model object
                            self.cnn_model = loaded_obj
                            # Try to detect model type
                            if 'mobilenet' in str(type(self.cnn_model)).lower():
                                self.model_type = 'mobilenetv2'
                            elif 'efficientnet' in str(type(self.cnn_model)).lower():
                                self.model_type = 'efficientnetb0'
                            else:
                                self.model_type = 'unknown'
                        
                        # Try to set eval mode if it's a PyTorch module
                        if isinstance(self.cnn_model, torch.nn.Module):
                            self.cnn_model.eval()  # Set to evaluation mode
                            logger.info(f"PyTorch CNN model loaded from {cnn_model_path}")
                            logger.info(f"Model type: {self.model_type}")
                            cnn_loaded = True
                        elif callable(self.cnn_model):
                            logger.info(f"Callable model loaded from {cnn_model_path}")
                            logger.info(f"Model type: {self.model_type}")
                            cnn_loaded = True
                        else:
                            logger.warning(f"Loaded object is not a PyTorch module or callable: {type(self.cnn_model)}")
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
                    self.model_type = 'sklearn' if hasattr(self.cnn_model, 'predict_proba') else 'unknown'
                    logger.info(f"Pickle CNN model loaded from {cnn_model_path}")
                    logger.info(f"Model type: {self.model_type}")
                    cnn_loaded = True
                else:
                    logger.warning(f"Unknown model file format: {cnn_model_path}")
            else:
                # This should not be reached if path search ran correctly
                logger.warning(f"CNN model path not set. CNN_MODEL_PATH env var: {os.getenv('CNN_MODEL_PATH', 'NOT SET')}")
            
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
                    with open(model_path.strip(), 'rb') as f:
                        model_name = os.path.basename(model_path.strip()).replace('.pkl', '')
                        self.additional_models[model_name] = pickle.load(f)
                        logger.info(f"Additional model loaded: {model_name}")
                except Exception as e:
                    logger.error(f"Error loading additional model {model_path}: {str(e)}")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in image using YOLOv8 (ONNX or PyTorch)
        Returns list of face bounding boxes and confidence scores
        """
        logger.info(f"[FaceDetection] Starting face detection. Image shape: {image.shape}, dtype: {image.dtype}")
        
        if self.use_yolo_onnx:
            if self.yolo_onnx_session is None:
                logger.error("[FaceDetection] YOLOv8 ONNX model not loaded")
                raise ValueError("YOLOv8 ONNX model not loaded")
            logger.info("[FaceDetection] Using ONNX YOLO model")
            faces = self._detect_faces_onnx(image)
        else:
            if self.yolo_model is None:
                logger.error("[FaceDetection] YOLOv8 PyTorch model not loaded")
                raise ValueError("YOLOv8 model not loaded")
            logger.info("[FaceDetection] Using PyTorch YOLO model")
            faces = self._detect_faces_pytorch(image)
        
        logger.info(f"[FaceDetection] Detected {len(faces)} face(s)")
        return faces
    
    def _detect_faces_onnx(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using ONNX YOLO model"""
        try:
            import onnxruntime as ort
            from onnxruntime import InferenceSession
            
            # Type guard: ensure session is not None
            if self.yolo_onnx_session is None:
                raise ValueError("YOLOv8 ONNX session is not initialized")
            
            # Type narrowing: assign to local variable with explicit type
            session: InferenceSession = self.yolo_onnx_session
            
            logger.debug(f"[ONNX FaceDetection] Input image shape: {image.shape}, dtype: {image.dtype}")
            
            # Get input and output names
            input_name = session.get_inputs()[0].name
            output_names = [output.name for output in session.get_outputs()]
            logger.debug(f"[ONNX FaceDetection] Input name: {input_name}, Output names: {output_names}")
            
            # Preprocess image for YOLO (resize to 640x640, normalize)
            original_shape = image.shape[:2]  # (H, W)
            input_size = 640
            resized = cv2.resize(image, (input_size, input_size))
            logger.debug(f"[ONNX FaceDetection] Resized to: {resized.shape}, Original: {original_shape}")
            
            # Convert BGR to RGB and normalize
            if len(resized.shape) == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            resized = resized.astype(np.float32) / 255.0
            
            # Transpose to (1, 3, H, W) format
            input_tensor = resized.transpose(2, 0, 1)[np.newaxis, ...]
            logger.debug(f"[ONNX FaceDetection] Input tensor shape: {input_tensor.shape}, range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
            
            # Run inference
            outputs = session.run(output_names, {input_name: input_tensor})
            
            # Ensure outputs are numpy arrays
            outputs = [np.asarray(out) if not isinstance(out, np.ndarray) else out for out in outputs]
            
            logger.debug(f"[ONNX FaceDetection] Number of outputs: {len(outputs)}")
            for i, out in enumerate(outputs):
                # Ensure output is a numpy array
                out_array: np.ndarray = np.asarray(out)
                logger.debug(f"[ONNX FaceDetection] Output {i} shape: {out_array.shape}, dtype: {out_array.dtype}")
                if out_array.size > 0:
                    logger.debug(f"[ONNX FaceDetection] Output {i} sample values (first 10): {out_array.flatten()[:10]}")
            
            # Parse outputs - YOLOv8 ONNX typically outputs [batch, num_detections, 6]
            # Format: [x_center, y_center, width, height, confidence, class]
            # Or sometimes: [x1, y1, x2, y2, confidence, class]
            # Ensure output is a numpy array for type checking
            output_array: np.ndarray = np.asarray(outputs[0])  # Get first output and ensure it's a numpy array
            
            logger.info(f"[ONNX FaceDetection] Raw output shape: {output_array.shape}")
            
            # Handle different output formats
            # YOLOv8 can output in different shapes:
            # - [1, num_detections, 6] - standard format
            # - [1, 6, num_detections] - transposed
            # - [num_detections, 6] - no batch dimension
            # - [1, 8400, 4+num_classes] - raw YOLO output (needs NMS)
            
            faces = []
            detections_before_filter = 0
            detections_after_confidence = 0
            
            # Try to handle different output shapes
            if len(output_array.shape) == 3:
                # output shape: [batch, num_detections, features]
                detections = output_array[0]  # Get first batch
                detections_before_filter = len(detections)
                logger.info(f"[ONNX FaceDetection] Total detections before filtering: {detections_before_filter}")
                
                # Scale factors to convert from 640x640 to original size
                scale_x = original_shape[1] / input_size
                scale_y = original_shape[0] / input_size
                logger.debug(f"[ONNX FaceDetection] Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
                
                for idx, det in enumerate(detections):
                    if len(det) < 5:
                        logger.debug(f"[ONNX FaceDetection] Detection {idx} has < 5 elements, skipping")
                        continue
                    
                    # Get confidence (usually at index 4)
                    conf = float(det[4])
                    logger.debug(f"[ONNX FaceDetection] Detection {idx}: confidence={conf:.3f}, values={det[:6]}")
                    
                    # Filter by confidence threshold (lowered from 0.25 to 0.15 for better detection)
                    confidence_threshold = 0.15
                    if conf < confidence_threshold:
                        logger.debug(f"[ONNX FaceDetection] Detection {idx} filtered out (confidence {conf:.3f} < {confidence_threshold})")
                        continue
                    
                    detections_after_confidence += 1
                    
                    # Check format - try both center+size and x1y1x2y2
                    if len(det) >= 6:
                        # Try center+size format first (most common for YOLO)
                        x_center, y_center, width, height = det[0], det[1], det[2], det[3]
                        
                        # Convert center+size to x1, y1, x2, y2
                        x1 = (x_center - width / 2) * scale_x
                        y1 = (y_center - height / 2) * scale_y
                        x2 = (x_center + width / 2) * scale_x
                        y2 = (y_center + height / 2) * scale_y
                    else:
                        # Fallback: assume x1, y1, x2, y2 format
                        x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                        x1 *= scale_x
                        y1 *= scale_y
                        x2 *= scale_x
                        y2 *= scale_y
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(original_shape[1], int(x2))
                    y2 = min(original_shape[0], int(y2))
                    
                    # Only add if bbox is valid
                    if x2 > x1 and y2 > y1:
                        faces.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf
                        })
                        logger.info(f"[ONNX FaceDetection] Added face: bbox=[{x1}, {y1}, {x2}, {y2}], conf={conf:.3f}")
                    else:
                        logger.warning(f"[ONNX FaceDetection] Invalid bbox for detection {idx}: [{x1}, {y1}, {x2}, {y2}]")
            elif len(output_array.shape) == 2:
                # Handle 2D output: [num_detections, features]
                logger.info(f"[ONNX FaceDetection] Processing 2D output format: {output_array.shape}")
                detections = output_array
                detections_before_filter = len(detections)
                logger.info(f"[ONNX FaceDetection] Total detections before filtering: {detections_before_filter}")
                
                scale_x = original_shape[1] / input_size
                scale_y = original_shape[0] / input_size
                
                for idx, det in enumerate(detections):
                    if len(det) < 5:
                        continue
                    
                    conf = float(det[4])
                    confidence_threshold = 0.15
                    
                    if conf < confidence_threshold:
                        continue
                    
                    detections_after_confidence += 1
                    
                    if len(det) >= 6:
                        x_center, y_center, width, height = det[0], det[1], det[2], det[3]
                        x1 = (x_center - width / 2) * scale_x
                        y1 = (y_center - height / 2) * scale_y
                        x2 = (x_center + width / 2) * scale_x
                        y2 = (y_center + height / 2) * scale_y
                    else:
                        x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                        x1 *= scale_x
                        y1 *= scale_y
                        x2 *= scale_x
                        y2 *= scale_y
                    
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(original_shape[1], int(x2))
                    y2 = min(original_shape[0], int(y2))
                    
                    if x2 > x1 and y2 > y1:
                        faces.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf
                        })
                        logger.info(f"[ONNX FaceDetection] Added face (2D format): bbox=[{x1}, {y1}, {x2}, {y2}], conf={conf:.3f}")
            else:
                logger.warning(f"[ONNX FaceDetection] Unexpected output shape: {output_array.shape}, expected 2D or 3D array")
                logger.warning(f"[ONNX FaceDetection] Attempting to process as flattened array...")
                # Try to reshape if possible
                if output_array.size > 0:
                    # Try common YOLO output sizes
                    possible_shapes = [
                        (1, output_array.size // 6, 6),  # [batch, detections, 6]
                        (output_array.size // 6, 6),    # [detections, 6]
                    ]
                    for shape in possible_shapes:
                        if output_array.size == np.prod(shape):
                            try:
                                reshaped = output_array.reshape(shape)
                                logger.info(f"[ONNX FaceDetection] Successfully reshaped to: {reshaped.shape}")
                                # Process reshaped output (recursive call would be complex, so log and continue)
                                break
                            except:
                                pass
            
            logger.info(f"[ONNX FaceDetection] Final results: {detections_before_filter} total, {detections_after_confidence} after confidence filter, {len(faces)} valid faces")
            
            return faces
            
        except Exception as e:
            logger.error(f"Error in ONNX face detection: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _detect_faces_pytorch(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using PyTorch YOLO model"""
        try:
            # Type guard: ensure yolo_model is not None
            if self.yolo_model is None:
                raise ValueError("YOLOv8 PyTorch model is not initialized")
            
            yolo_model = self.yolo_model  # Type narrowing
            
            # Run YOLOv8 inference
            results = yolo_model(image, conf=0.25, verbose=False)  # type: ignore
            
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
    
    def preprocess_face(self, image: np.ndarray, bbox: List[int], target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Extract and preprocess face region for emotion classification
        For MobileNetV2/EfficientNet models, use 224x224 RGB images
        For other models, use 48x48 grayscale
        """
        x1, y1, x2, y2 = bbox
        
        # Extract face region
        face_roi = image[y1:y2, x1:x2]
        
        # Determine target size based on model type
        if target_size is None:
            # Check model type - MobileNetV2 and EfficientNet use 224x224 RGB
            if self.model_type in ['mobilenetv2', 'efficientnetb0'] or self.use_onnx:
                target_size = (224, 224)
                use_rgb = True
            elif isinstance(self.cnn_model, torch.nn.Module):
                # Check model name/type to determine input size
                model_name = str(type(self.cnn_model)).lower()
                if 'efficientnet' in model_name or 'mobilenet' in model_name or 'resnet' in model_name:
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
            # MobileNetV2/EfficientNet expect RGB (not BGR)
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
        
        # Normalize pixel values (0-1 range)
        face_roi = face_roi.astype(np.float32) / 255.0
        
        # Reshape for model input
        if use_rgb:
            if self.use_onnx:
                # ONNX expects (1, 3, H, W) format
                face_roi = face_roi.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)
                face_roi = np.expand_dims(face_roi, axis=0)  # (3, H, W) -> (1, 3, H, W)
            else:
                # PyTorch: RGB: (H, W, 3) -> (1, 3, H, W)
                face_roi = face_roi.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)
                face_roi = face_roi.reshape(1, 3, target_size[0], target_size[1])
        else:
            # Grayscale: (H, W) -> (1, 1, H, W) for PyTorch or (1, H, W, 1) for TensorFlow
            if isinstance(self.cnn_model, torch.nn.Module) or self.use_onnx:
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
            # Default emotion classes (7 emotions)
            emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            
            # Predict using the loaded model
            # Check model type in order: ONNX -> PyTorch -> Scikit-learn -> Keras/TensorFlow
            
            if self.use_onnx:
                # ONNX model inference
                import onnxruntime as ort
                
                # Type guard: ensure cnn_model is an InferenceSession
                if not isinstance(self.cnn_model, ort.InferenceSession):
                    raise ValueError("CNN model is not an ONNX InferenceSession")
                
                onnx_session = self.cnn_model  # Type narrowing
                
                # Get input and output names
                input_name = onnx_session.get_inputs()[0].name
                output_name = onnx_session.get_outputs()[0].name
                
                # Ensure input is float32 and correct shape
                if face_image.dtype != np.float32:
                    face_image = face_image.astype(np.float32)
                
                # Run inference
                outputs = onnx_session.run([output_name], {input_name: face_image})
                output = np.asarray(outputs[0])  # Ensure numpy array
                
                # Apply softmax if needed (ONNX models may or may not include softmax)
                if output.ndim > 1:
                    # Apply softmax
                    exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
                    probabilities = (exp_output / exp_output.sum(axis=1, keepdims=True))[0]
                else:
                    exp_output = np.exp(output - np.max(output))
                    probabilities = exp_output / exp_output.sum()
            
            # PyTorch model check
            elif isinstance(self.cnn_model, torch.nn.Module):
                pytorch_model = self.cnn_model  # Type narrowing
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
                        pytorch_model = pytorch_model.to(self.device)
                    
                    output = pytorch_model(input_tensor)
                    self.cnn_model = pytorch_model  # Update in case it was moved to device
                    
                    # Handle different output formats
                    if isinstance(output, torch.Tensor):
                        if output.dim() > 1:
                            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                        else:
                            probabilities = torch.softmax(output, dim=0).cpu().numpy()
                    else:
                        probabilities = output[0] if isinstance(output, (list, tuple)) else output
            
            # Scikit-learn model check
            elif hasattr(self.cnn_model, 'predict_proba') and callable(getattr(self.cnn_model, 'predict_proba', None)):
                probabilities = self.cnn_model.predict_proba(face_image.reshape(1, -1))[0]  # type: ignore
                emotion_classes = self.cnn_model.classes_ if hasattr(self.cnn_model, 'classes_') else emotion_classes  # type: ignore
            
            # Keras/TensorFlow model check
            elif hasattr(self.cnn_model, 'predict') and callable(getattr(self.cnn_model, 'predict', None)):
                probabilities = self.cnn_model.predict(face_image, verbose=0)[0]  # type: ignore
            
            else:
                raise ValueError(f"Unknown model type: {type(self.cnn_model)}. Expected ONNX, PyTorch nn.Module, scikit-learn, or Keras/TensorFlow model.")
            
            # Create emotion dictionary
            emotion_dict = {}
            for i, emotion in enumerate(emotion_classes):
                if i < len(probabilities):
                    emotion_dict[emotion] = float(probabilities[i])
            
            return emotion_dict
            
        except Exception as e:
            logger.error(f"Error in emotion prediction: {str(e)}")
            raise
    
    def detect_emotions_in_image(self, image: np.ndarray) -> List[Dict]:
        """
        Complete pipeline: detect faces and predict emotions
        Returns list of detections with emotions, engagement scores, and concentration scores
        """
        logger.info(f"[EmotionDetection] Starting emotion detection. Image shape: {image.shape}")
        
        # Detect faces
        faces = self.detect_faces(image)
        logger.info(f"[EmotionDetection] Face detection completed. Found {len(faces)} face(s)")
        
        if len(faces) == 0:
            logger.warning("[EmotionDetection] No faces detected in image. Returning empty list.")
        
        results = []
        for face in faces:
            bbox = face['bbox']
            
            # Preprocess face
            face_image = self.preprocess_face(image, bbox)
            
            # Predict emotion
            emotions = self.predict_emotion(face_image)
            
            # Get dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            
            # Calculate scores
            engagement_score = self._calculate_engagement(emotions, dominant_emotion[0])
            concentration_score = self.calculate_concentration_score(emotions)
            
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
        Calculate concentration score based on emotion probabilities
        attentive = neutral, happy, surprise, focus
        inattentive = sleepy, boredom, frustration, confusion
        
        score = (sum(attentive_probs) / sum(attentive + inattentive)) * 100
        """
        # Define emotion categories
        attentive_emotions = ['neutral', 'happy', 'surprise']
        inattentive_emotions = ['sleepy', 'boredom', 'frustration', 'confusion']
        
        # Map detected emotions to categories (handle variations in emotion names)
        emotion_mapping = {
            'angry': 'frustration',
            'disgust': 'boredom',
            'fear': 'confusion',
            'sad': 'confusion',
            'happy': 'happy',
            'surprise': 'surprise',
            'neutral': 'neutral',
            'focused': 'happy',  # Treat focused as happy
            'bored': 'boredom',
            'confused': 'confusion',
            'frustrated': 'frustration',
            'sleepy': 'sleepy'
        }
        
        # Calculate probabilities for attentive and inattentive
        attentive_sum = 0.0
        inattentive_sum = 0.0
        
        for emotion, prob in emotions.items():
            mapped_emotion = emotion_mapping.get(emotion.lower(), emotion.lower())
            
            if mapped_emotion in attentive_emotions:
                attentive_sum += prob
            elif mapped_emotion in inattentive_emotions:
                inattentive_sum += prob
        
        # Calculate concentration score
        total = attentive_sum + inattentive_sum
        if total > 0:
            concentration = (attentive_sum / total) * 100.0
        else:
            # If no emotions match categories, default to neutral (50%)
            concentration = 50.0
        
        return max(0.0, min(100.0, concentration))
    
    def _calculate_engagement(self, emotions: Dict[str, float], dominant: str) -> float:
        """
        Calculate engagement score based on emotions (0.0 to 1.0)
        Higher score = more engaged
        This is a legacy method, concentration_score should be used instead
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

