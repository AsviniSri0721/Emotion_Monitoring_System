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
        self.yolo_onnx_session = None  # For ONNX YOLO models
        self.use_yolo_onnx = False
        self.model_config = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.image_processor = None  # HuggingFace AutoImageProcessor
        self.debug_dir = None  # Debug directory for saving face crops
        self.emotion_labels = None  # Emotion labels from model
        self.hf_checkpoint = None  # HuggingFace checkpoint name
        self.id2label = None  # ID to label mapping
        self.label2id = None  # Label to ID mapping
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
                if os.path.exists(abs_yolo_path):
                    # Use absolute path when file exists
                    yolo_model_path = abs_yolo_path
                    logger.info(f"Using YOLO_MODEL_PATH from env var: {yolo_model_path}")
                else:
                    logger.warning(f"YOLO_MODEL_PATH env var points to non-existent file: {yolo_model_path} (resolved: {abs_yolo_path}). Will search for model.")
                    yolo_model_path = None
            
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
            
            # ----------------- LOAD EMOTION MODEL (HuggingFace MobileNetV2) -----------------
            from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification
            
            cnn_model_path = os.getenv("CNN_MODEL_PATH", "models/mobilenetv2_emotion_model_CPU.pkl")
            self.model_type = "mobilenetv2"
            self.use_onnx = False
            
            # Search for .pkl file if path doesn't exist
            if not os.path.exists(cnn_model_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                backend_dir = os.path.dirname(script_dir)
                cwd = os.getcwd()
                
                search_paths = [
                    os.path.join(backend_dir, "models", "mobilenetv2_emotion_model_CPU.pkl"),
                    os.path.join(backend_dir, "backend", "models", "mobilenetv2_emotion_model_CPU.pkl"),
                    os.path.join(cwd, "backend", "models", "mobilenetv2_emotion_model_CPU.pkl"),
                    os.path.join(cwd, "backend", "backend", "models", "mobilenetv2_emotion_model_CPU.pkl"),
                    os.path.join(cwd, "models", "mobilenetv2_emotion_model_CPU.pkl"),
                    "models/mobilenetv2_emotion_model_CPU.pkl",
                    "backend/models/mobilenetv2_emotion_model_CPU.pkl",
                    "backend/backend/models/mobilenetv2_emotion_model_CPU.pkl",
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
            checkpoint = payload.get("checkpoint", "google/mobilenet_v2_1.0_224")
            
            self.hf_checkpoint = checkpoint
            self.emotion_labels = labels
            self.id2label = id2label
            self.label2id = label2id
            
            # Load model config and create model
            cfg = AutoConfig.from_pretrained(
                checkpoint,
                num_labels=len(labels),
                id2label=id2label,
                label2id=label2id,
            )
            
            model = AutoModelForImageClassification.from_config(cfg)
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            
            self.cnn_model = model
            cnn_loaded = True
            
            print("\n===== CNN MODEL LOADED (HF MobileNetV2) =====")
            print("Checkpoint:", checkpoint)
            print("Labels:", labels)
            print("Device:", self.device)
            print("=============================================\n")
            logger.info("===== CNN MODEL LOADED (HF MobileNetV2) =====")
            logger.info(f"Checkpoint: {checkpoint}")
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
            
            # Load HuggingFace image processor FIRST (needed for preprocessing)
            self._load_hf_processor()
            
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
    
    def _load_hf_processor(self):
        """Load HuggingFace AutoImageProcessor for preprocessing"""
        from transformers import AutoImageProcessor
        import os
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(script_dir)
        candidate_paths = [
            os.path.join(backend_dir, "models", "mobilenetv2_processor"),
            os.path.join(script_dir, "..", "models", "mobilenetv2_processor"),
            os.path.join(os.getcwd(), "backend/models/mobilenetv2_processor"),
            os.path.join(backend_dir, "backend", "models", "mobilenetv2_processor"),
            os.path.join(os.getcwd(), "backend", "backend", "models", "mobilenetv2_processor"),
            "backend/models/mobilenetv2_processor",
            "backend/backend/models/mobilenetv2_processor",
        ]
        
        for p in candidate_paths:
            cfg_path = os.path.join(p, "preprocessor_config.json")
            if os.path.exists(cfg_path):
                try:
                    self.image_processor = AutoImageProcessor.from_pretrained(p)
                    print("✓ Loaded HF processor from:", p)
                    logger.info(f"✓ Loaded HF processor from: {p}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load processor from {p}: {str(e)}, trying HF checkpoint")
                    break
        
        print("⚠️ mobilenetv2_processor not found or invalid, using HF checkpoint processor")
        logger.warning("mobilenetv2_processor not found or invalid, using HF checkpoint processor")
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(self.hf_checkpoint)
        except Exception as e:
            logger.error(f"Failed to load processor from HF checkpoint {self.hf_checkpoint}: {str(e)}")
            # Use a default processor if available
            try:
                self.image_processor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
                logger.info("Using default processor: google/mobilenet_v2_1.0_224")
            except Exception as e2:
                logger.error(f"Failed to load default processor: {str(e2)}")
                self.image_processor = None
    
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
        Detect faces in image using YOLOv8 (ONNX or PyTorch)
        Returns list of face bounding boxes and confidence scores
        """
        logger.info(f"[FaceDetection] Starting face detection. Image shape: {image.shape}, dtype: {image.dtype}")
        logger.info(f"[FaceDetection] Image value range: min={image.min()}, max={image.max()}")
        logger.info(f"[FaceDetection] Using {'ONNX' if self.use_yolo_onnx else 'PyTorch'} YOLO model")
        
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
            
            logger.info(f"[ONNX FaceDetection] Number of outputs: {len(outputs)}")
            for i, out in enumerate(outputs):
                # Ensure output is a numpy array
                out_array: np.ndarray = np.asarray(out)
                logger.info(f"[ONNX FaceDetection] Output {i} shape: {out_array.shape}, dtype: {out_array.dtype}, min={out_array.min():.3f}, max={out_array.max():.3f}, mean={out_array.mean():.3f}")
                if out_array.size > 0 and out_array.size < 1000:
                    logger.info(f"[ONNX FaceDetection] Output {i} all values: {out_array.flatten()}")
                elif out_array.size > 0:
                    logger.info(f"[ONNX FaceDetection] Output {i} sample values (first 20): {out_array.flatten()[:20]}")
            
            # Try to find the correct output - YOLOv8 might have multiple outputs
            # Look for output with shape that suggests detections
            output_array = None
            for i, out in enumerate(outputs):
                out_arr = np.asarray(out)
                # Prefer outputs that look like detection format
                if len(out_arr.shape) == 3 and out_arr.shape[0] == 1 and out_arr.shape[2] >= 6:
                    # Shape like [1, num_detections, 6+] - this is likely the detection output
                    output_array = out_arr
                    logger.info(f"[ONNX FaceDetection] Using output {i} as detection output: {output_array.shape}")
                    break
                elif len(out_arr.shape) == 2 and out_arr.shape[1] >= 6:
                    # Shape like [num_detections, 6+] - also detection format
                    output_array = out_arr
                    logger.info(f"[ONNX FaceDetection] Using output {i} as detection output: {output_array.shape}")
                    break
            
            # If no detection-like output found, use first output
            if output_array is None:
                output_array = np.asarray(outputs[0])
                logger.warning(f"[ONNX FaceDetection] No detection-like output found, using first output: {output_array.shape}")
            
            logger.info(f"[ONNX FaceDetection] Processing output shape: {output_array.shape}, ndim={output_array.ndim}")
            
            # Handle different output formats
            # YOLOv8 can output in different shapes:
            # - [1, num_detections, 6] - standard format
            # - [1, 6, num_detections] - transposed
            # - [num_detections, 6] - no batch dimension
            # - [1, 8400, 4+num_classes] - raw YOLO output (needs NMS)
            # - [1, 80, 80, 80] - 4D feature map (needs decoding)
            
            faces = []
            detections_before_filter = 0
            detections_after_confidence = 0
            
            # Try to handle different output shapes - CHECK 4D FIRST since that's what we're getting
            if len(output_array.shape) == 4:
                # Handle 4D output FIRST - this is what yolov8n-face.onnx outputs
                logger.warning(f"[ONNX FaceDetection] Detected 4D feature map output: {output_array.shape}")
                logger.warning(f"[ONNX FaceDetection] ONNX model outputs feature map instead of detections - falling back to PyTorch YOLO")
                
                # Fallback: Load PyTorch YOLO model instead
                try:
                    yolo_pt_paths = [
                        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend', 'models', 'yolov8n-face.pt'),
                        os.path.join(os.getcwd(), 'backend', 'models', 'yolov8n-face.pt'),
                        os.path.join(os.getcwd(), 'backend', 'backend', 'models', 'yolov8n-face.pt'),
                        'yolov8n-face.pt',
                    ]
                    
                    yolo_pt_path = None
                    for path in yolo_pt_paths:
                        abs_path = os.path.abspath(path)
                        if os.path.exists(abs_path):
                            yolo_pt_path = abs_path
                            break
                    
                    if yolo_pt_path:
                        logger.info(f"[ONNX FaceDetection] Loading PyTorch YOLO fallback from: {yolo_pt_path}")
                        self.yolo_model = YOLO(yolo_pt_path)
                        self.use_yolo_onnx = False
                        # Use PyTorch detection instead
                        return self._detect_faces_pytorch(image)
                    else:
                        logger.warning("[ONNX FaceDetection] PyTorch YOLO fallback not found, trying default YOLOv8n")
                        self.yolo_model = YOLO('yolov8n.pt')
                        self.use_yolo_onnx = False
                        return self._detect_faces_pytorch(image)
                except Exception as fallback_error:
                    logger.error(f"[ONNX FaceDetection] Fallback to PyTorch failed: {str(fallback_error)}")
                    # Continue with manual decoding as last resort
                    logger.info(f"[ONNX FaceDetection] Processing 4D feature map output: {output_array.shape}")
                    logger.info(f"[ONNX FaceDetection] This is a YOLOv8 feature map - need to decode and apply NMS")
                
                # Remove batch dimension
                feature_map = output_array[0]  # Shape: (80, 80, 80)
                grid_h, grid_w, channels = feature_map.shape
                
                logger.info(f"[ONNX FaceDetection] Feature map: grid_h={grid_h}, grid_w={grid_w}, channels={channels}")
                
                # YOLOv8 format: channels = 4 (bbox) + 1 (objectness) + num_classes
                # For face detection, typically: 4 (bbox) + 1 (conf) + 1 (face class) = 6
                # But 80 channels suggests: 4 + 1 + 75 classes OR different format
                
                # Try reshaping to (grid_h * grid_w, channels) and process each cell
                num_cells = grid_h * grid_w
                feature_map_flat = feature_map.reshape(num_cells, channels)
                
                logger.info(f"[ONNX FaceDetection] Flattened to: {feature_map_flat.shape}")
                
                # Try to extract bboxes - assume first 4 channels are bbox coords
                # and channel 4 is confidence
                scale_x = original_shape[1] / input_size
                scale_y = original_shape[0] / input_size
                
                # Grid cell size
                cell_w = input_size / grid_w
                cell_h = input_size / grid_h
                
                detections_list = []
                
                for cell_idx in range(num_cells):
                    cell_data = feature_map_flat[cell_idx]
                    
                    # Get cell position
                    cell_y = cell_idx // grid_w
                    cell_x = cell_idx % grid_w
                    
                    # Try different channel interpretations
                    # YOLOv8 format: coordinates are typically sigmoid-activated and relative to cell
                    # Format: [x_offset, y_offset, w, h, objectness, class_scores...]
                    if channels >= 5:
                        # Apply sigmoid to get values in 0-1 range (if needed)
                        # Check if values are already in reasonable range
                        x_offset_raw = float(cell_data[0])
                        y_offset_raw = float(cell_data[1])
                        w_raw = float(cell_data[2])
                        h_raw = float(cell_data[3])
                        conf_raw = float(cell_data[4])
                        
                        # Apply sigmoid if values are outside 0-1 range
                        def sigmoid(x):
                            return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
                        
                        # Check if sigmoid is needed (values > 1 or < 0 suggest raw logits)
                        if abs(x_offset_raw) > 1 or abs(y_offset_raw) > 1:
                            x_offset = sigmoid(x_offset_raw)
                            y_offset = sigmoid(y_offset_raw)
                            w = sigmoid(w_raw) * input_size  # Scale width
                            h = sigmoid(h_raw) * input_size  # Scale height
                            conf = sigmoid(conf_raw)
                        else:
                            # Values already in 0-1 range
                            x_offset = x_offset_raw
                            y_offset = y_offset_raw
                            # Width/height interpretation - be VERY conservative
                            # YOLOv8 typically outputs relative sizes (0-1 range)
                            # If values are in 0-1 range, they're relative to image
                            # If values are > 1, they might be absolute pixels
                            if 0 <= w_raw <= 1 and 0 <= h_raw <= 1:
                                # Relative to image size - this is the correct format
                                w = w_raw * input_size
                                h = h_raw * input_size
                                # Early rejection: if relative size is > 0.5, it's probably wrong
                                if w_raw > 0.5 or h_raw > 0.5:
                                    continue
                            elif w_raw > 1 or h_raw > 1:
                                # Absolute pixels - REJECT if close to image size
                                if w_raw > input_size * 0.6 or h_raw > input_size * 0.6:
                                    continue
                                w = min(w_raw, input_size * 0.5)  # Cap at 50% of image
                                h = min(h_raw, input_size * 0.5)
                            else:
                                # Negative or invalid values
                                continue
                            conf = max(0.0, min(1.0, conf_raw))  # Clamp confidence
                        
                        # Convert cell-relative to absolute coordinates (in 640x640 space)
                        x_center = (cell_x + x_offset) * cell_w
                        y_center = (cell_y + y_offset) * cell_h
                        
                        # CRITICAL: Ensure width/height are reasonable BEFORE converting
                        # Face should be 5-50% of image (not 80% - that's too large)
                        min_face_size = input_size * 0.05  # 5% minimum
                        max_face_size = input_size * 0.5   # 50% maximum (reduced from 80%)
                        
                        # If w/h are already in pixel space and too large, they're wrong
                        if w > max_face_size or h > max_face_size:
                            continue
                        
                        w = np.clip(w, min_face_size, max_face_size)
                        h = np.clip(h, min_face_size, max_face_size)
                        
                        # Convert to x1, y1, x2, y2 in original image coordinates
                        x1 = (x_center - w / 2) * scale_x
                        y1 = (y_center - h / 2) * scale_y
                        x2 = (x_center + w / 2) * scale_x
                        y2 = (y_center + h / 2) * scale_y
                        
                        # Additional validation: ensure coordinates are within image bounds
                        x1 = max(0, min(x1, original_shape[1] - 10))  # Leave at least 10px margin
                        y1 = max(0, min(y1, original_shape[0] - 10))
                        x2 = max(x1 + 10, min(x2, original_shape[1]))  # Ensure minimum size
                        y2 = max(y1 + 10, min(y2, original_shape[0]))
                        
                        # Validate bbox is reasonable (not covering entire image)
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        img_width = original_shape[1]
                        img_height = original_shape[0]
                        
                        # Stricter validation: face should be less than 50% of image (reduced from 80%)
                        width_ratio = bbox_width / img_width
                        height_ratio = bbox_height / img_height
                        if width_ratio > 0.5 or height_ratio > 0.5:
                            continue
                        
                        # Face should be at least 5% of image
                        if width_ratio < 0.05 or height_ratio < 0.05:
                            continue
                        
                        # Filter by confidence
                        confidence_threshold = 0.3
                        if conf > confidence_threshold:
                            detections_list.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': conf,
                                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2  # For NMS
                            })
                
                logger.info(f"[ONNX FaceDetection] Extracted {len(detections_list)} detections from feature map")
                
                # Apply Non-Maximum Suppression (NMS) to remove overlapping boxes
                if len(detections_list) > 0:
                    # Sort by confidence
                    detections_list.sort(key=lambda x: x['confidence'], reverse=True)
                    
                    # Simple NMS implementation
                    nms_threshold = 0.5
                    keep = []
                    while detections_list:
                        best = detections_list.pop(0)
                        keep.append(best)
                        
                        # Remove overlapping detections
                        detections_list = [
                            det for det in detections_list
                            if self._iou(best, det) < nms_threshold
                        ]
                    
                    logger.info(f"[ONNX FaceDetection] After NMS: {len(keep)} faces")
                    
                    # Convert to final format
                    for det in keep:
                        x1 = max(0, int(det['x1']))
                        y1 = max(0, int(det['y1']))
                        x2 = min(original_shape[1], int(det['x2']))
                        y2 = min(original_shape[0], int(det['y2']))
                        
                        if x2 > x1 and y2 > y1:
                            faces.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': det['confidence']
                            })
                            logger.info(f"[ONNX FaceDetection] Added face (4D format): bbox=[{x1}, {y1}, {x2}, {y2}], conf={det['confidence']:.3f}")
            elif len(output_array.shape) == 3:
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
                    
                    # Filter by confidence threshold (increased to 0.3 for better quality detections)
                    confidence_threshold = 0.3
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
                    confidence_threshold = 0.3
                    
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
                logger.warning(f"[ONNX FaceDetection] Unexpected output shape: {output_array.shape}, expected 2D, 3D, or 4D array")
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
            
            logger.info(f"[ONNX FaceDetection] Final results: {len(faces)} valid faces")
            
            return faces
            
        except Exception as e:
            logger.error(f"Error in ONNX face detection: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _iou(self, box1: Dict, box2: Dict) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1['x1'], box1['y1'], box1['x2'], box1['y2']
        x1_2, y1_2, x2_2, y2_2 = box2['x1'], box2['y1'], box2['x2'], box2['y2']
        
        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
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
    
    def preprocess_face(self, image: np.ndarray, bbox: List[int], target_size: Optional[Tuple[int, int]] = None) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Extract and preprocess face region for emotion classification
        For MobileNetV2/EfficientNet models, use 224x224 RGB images
        For other models, use 48x48 grayscale
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
        
        # Use HuggingFace processor if available
        if self.image_processor is not None:
            rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            inputs = self.image_processor(images=pil_img, return_tensors="np")
            face_tensor = inputs["pixel_values"]  # NCHW float32 normalized
            logger.info(f"[FacePreprocessing] Using HuggingFace processor - output shape: {face_tensor.shape}")
            return face_tensor, None
        
        # Manual preprocessing fallback (if processor not available)
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
        
        # Normalize pixel values
        face_roi = face_roi.astype(np.float32)
        
        if use_rgb:
            # For MobileNetV2/EfficientNet: Use ImageNet normalization
            # ImageNet mean and std
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            
            # Normalize: (pixel / 255.0 - mean) / std
            face_roi = face_roi / 255.0
            face_roi = (face_roi - mean) / std
        else:
            # For grayscale models: simple 0-1 normalization
            face_roi = face_roi / 255.0
        
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
        
        return face_roi, None
    
    def predict_emotion(self, face_image: np.ndarray) -> Union[Dict[str, float], str]:
        """
        Predict emotion from preprocessed face image using CNN model
        Returns dictionary of emotion probabilities or "model_not_confident" (as string) if validation fails
        """
        if self.cnn_model is None:
            logger.error("[EmotionPrediction] CNN model not loaded")
            return "model_not_confident"
        
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
            
            # PyTorch model check (including HuggingFace models)
            elif isinstance(self.cnn_model, torch.nn.Module):
                with torch.no_grad():
                    if isinstance(face_image, np.ndarray):
                        x = torch.from_numpy(face_image).float()
                    else:
                        x = face_image.float()
                    
                    if x.ndim == 3:
                        x = x.unsqueeze(0)
                    
                    x = x.to(self.device)
                    outputs = self.cnn_model(x)
                    logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                
                # Use emotion_labels from model loading
                if hasattr(self, 'emotion_labels') and self.emotion_labels:
                    emotion_classes = self.emotion_labels
                else:
                    emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
                
                emotions = {emo: float(probs[i]) for i, emo in enumerate(emotion_classes)}
                
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
                emotion_classes = self.cnn_model.classes_ if hasattr(self.cnn_model, 'classes_') else emotion_classes  # type: ignore
                
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
            # Return default uniform probabilities if model not confident
            if hasattr(self, 'emotion_labels') and self.emotion_labels:
                labels = self.emotion_labels
            else:
                labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            return {label: 1.0 / len(labels) for label in labels}
        return emotions
    
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
                # Safe fallback
                results.append({
                    'bbox': bbox,
                    'emotions': {'neutral': 1.0},
                    'dominant_emotion': 'neutral',
                    'emotion_state': 'neutral',
                    'confidence': 0.0,
                    'engagement_score': 0.0,
                    'concentration_score': 50.0,
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
                # Safe fallback
                results.append({
                    'bbox': bbox,
                    'emotions': {'neutral': 1.0},
                    'dominant_emotion': 'neutral',
                    'emotion_state': 'neutral',
                    'confidence': 0.0,
                    'engagement_score': 0.0,
                    'concentration_score': 50.0,
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

