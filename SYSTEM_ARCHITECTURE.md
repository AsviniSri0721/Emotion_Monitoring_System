# Emotion Monitoring System - Complete Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Folder Structure](#folder-structure)
3. [Backend Architecture](#backend-architecture)
4. [Model Loading Pipeline](#model-loading-pipeline)
5. [HuggingFace Integration](#huggingface-integration)
6. [YOLO Integration](#yolo-integration)
7. [Inference Flow](#inference-flow)
8. [PKL/PT Model Usage](#pklpt-model-usage)
9. [API Endpoints](#api-endpoints)
10. [Frontend Architecture](#frontend-architecture)
11. [UI Workflow](#ui-workflow)
12. [Database Schema](#database-schema)
13. [Code Explanations](#code-explanations)

---

## System Overview

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ LiveSession  │  │ VideoPlayer  │  │  Dashboard   │         │
│  │  Component   │  │  Component   │  │  Components  │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                 │                 │                  │
│         └─────────────────┼─────────────────┘                  │
│                           │                                     │
│                    ┌──────▼──────┐                             │
│                    │  API Client │                             │
│                    │  (axios)    │                             │
│                    └──────┬──────┘                             │
└───────────────────────────┼─────────────────────────────────────┘
                            │ HTTP/REST
┌───────────────────────────▼─────────────────────────────────────┐
│                    Backend (Flask)                              │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                    Flask Application                      │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐         │ │
│  │  │   Auth     │  │  Emotions  │  │   Videos  │         │ │
│  │  │  Routes    │  │   Routes    │  │   Routes  │         │ │
│  │  └─────┬──────┘  └──────┬──────┘  └──────┬──────┘        │ │
│  │        │                │                 │                │ │
│  │        └────────────────┼─────────────────┘                │ │
│  │                          │                                 │ │
│  │                   ┌──────▼──────┐                         │ │
│  │                   │ ModelService│                         │ │
│  │                   └──────┬──────┘                         │ │
│  └───────────────────────────┼─────────────────────────────────┘ │
│                              │                                   │
│  ┌───────────────────────────▼─────────────────────────────────┐ │
│  │                    ML Models                                 │ │
│  │  ┌──────────────┐              ┌──────────────┐            │ │
│  │  │ YOLOv8-face  │              │ MobileNetV2  │            │ │
│  │  │  (ONNX/PyT)  │              │  (HuggingFace)│            │ │
│  │  │ Face Detect  │              │  Emotion CLF │            │ │
│  │  └──────────────┘              └──────────────┘            │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│  ┌───────────────────────────▼─────────────────────────────────┐ │
│  │                  MySQL Database                              │ │
│  │  users | videos | live_sessions | emotion_data | reports   │ │
│  └──────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Backend:**
- Python 3.x
- Flask (Web framework)
- Flask-JWT-Extended (Authentication)
- PyTorch (Deep learning)
- HuggingFace Transformers (Model loading)
- Ultralytics YOLO (Face detection)
- OpenCV (Image processing)
- NumPy (Numerical operations)
- MySQL/MariaDB (Database)
- ONNX Runtime (Optional ONNX inference)

**Frontend:**
- React 18+
- TypeScript
- Axios (HTTP client)
- React Router (Routing)

**Infrastructure:**
- XAMPP (MySQL server)
- Node.js (Frontend build)

### System Components

1. **Authentication System**: JWT-based authentication with role-based access (teacher/student)
2. **Model Service**: Centralized ML model loading and inference service
3. **Face Detection**: YOLOv8-based face detection (ONNX or PyTorch)
4. **Emotion Classification**: HuggingFace MobileNetV2-based emotion recognition
5. **Live Sessions**: Real-time emotion monitoring during live classes
6. **Video Processing**: Recorded video analysis and emotion tracking
7. **Reporting System**: Engagement and concentration analytics
8. **Intervention System**: Automated interventions for low engagement

---

## Folder Structure

```
Emotion_Monitoring_System/
│
├── backend/                          # Backend Flask application
│   ├── app.py                        # Main Flask application entry point
│   ├── config.py                     # Configuration settings
│   ├── config_logging.py             # Logging configuration
│   ├── requirements.txt              # Python dependencies
│   │
│   ├── backend/                      # Nested backend directory
│   │   ├── models/                   # ML model files
│   │   │   ├── mobilenetv2_emotion_model_CPU.pkl    # Main emotion model (PKL)
│   │   │   ├── mobilenetv2_emotion_model.pkl        # Alternative emotion model
│   │   │   ├── mobilenetv2_best.pt                   # PyTorch emotion model
│   │   │   ├── mobilenetv2.onnx                      # ONNX emotion model
│   │   │   ├── yolov8n-face.onnx                     # YOLO face detection (ONNX)
│   │   │   └── mobilenetv2_processor/                # HuggingFace processor config
│   │   │       └── preprocessor_config.json
│   │   └── uploads/                  # Uploaded video files
│   │
│   ├── routes/                       # Flask route blueprints
│   │   ├── __init__.py
│   │   ├── auth.py                   # Authentication endpoints
│   │   ├── emotions.py               # Emotion detection endpoints
│   │   ├── videos.py                 # Video upload/management
│   │   ├── sessions.py               # Session management
│   │   ├── reports.py                # Report generation
│   │   └── interventions.py          # Intervention triggers
│   │
│   ├── services/                     # Business logic services
│   │   ├── __init__.py
│   │   ├── database.py               # Database connection and queries
│   │   └── model_service.py          # ML model loading and inference
│   │
│   ├── live_sessions/                # Live session module
│   │   ├── __init__.py
│   │   ├── models.py                 # Live session data models
│   │   ├── routes.py                 # Live session routes
│   │   ├── service.py                # Live session business logic
│   │   └── controller.py             # Live session controllers
│   │
│   ├── utils/                        # Utility functions
│   │   ├── __init__.py
│   │   └── jwt_helpers.py            # JWT token utilities
│   │
│   ├── logs/                         # Application logs
│   ├── debug/                        # Debug face crop images
│   ├── uploads/                      # Uploaded files
│   ├── database_schema_mysql.sql     # Database schema
│   └── yolov8n.pt                    # Default YOLO model (fallback)
│
└── client/                           # Frontend React application
    ├── package.json                  # Node.js dependencies
    ├── tsconfig.json                 # TypeScript configuration
    │
    └── src/
        ├── index.tsx                 # React entry point
        ├── App.tsx                   # Main app component
        │
        ├── pages/                    # Page components
        │   ├── Login.tsx             # Login page
        │   ├── Register.tsx          # Registration page
        │   ├── Dashboard.tsx         # Main dashboard
        │   ├── TeacherDashboard.tsx  # Teacher dashboard
        │   ├── StudentDashboard.tsx  # Student dashboard
        │   ├── LiveSession.tsx      # Live session page
        │   ├── VideoPlayer.tsx       # Video playback page
        │   └── ReportPage.tsx        # Report viewing page
        │
        ├── components/                # Reusable components
        │   └── EngagementMeter.tsx  # Engagement visualization
        │
        ├── hooks/                    # Custom React hooks
        │   ├── useEmotionStream.ts           # Emotion streaming hook
        │   └── useLiveSessionEmotionStream.ts # Live session streaming
        │
        ├── services/                 # API services
        │   ├── api.ts               # Axios instance and interceptors
        │   └── emotionDetection.ts  # Emotion detection API calls
        │
        ├── api/                      # API client modules
        │   └── liveSessions.ts      # Live session API client
        │
        ├── contexts/                  # React contexts
        │   └── AuthContext.tsx      # Authentication context
        │
        └── utils/                    # Utility functions
            └── logger.ts             # Frontend logging utility
```

### Key Directory Explanations

- **`backend/backend/models/`**: Contains all ML model files (PKL, PT, ONNX formats)
- **`backend/routes/`**: Flask blueprints organized by feature domain
- **`backend/services/`**: Core business logic separated from HTTP handling
- **`backend/live_sessions/`**: Self-contained module for live session functionality
- **`client/src/hooks/`**: Custom React hooks for emotion streaming logic
- **`client/src/api/`**: Type-safe API client modules

---

## Backend Architecture

### Flask Application Structure

The main Flask application is initialized in `backend/app.py`:

```47:94:backend/app.py
app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET', 'your-secret-key-change-this')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = False
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_DIR', './uploads')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Initialize extensions
jwt = JWTManager(app)
CORS(app, 
     resources={r"/api/*": {
         "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
         "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
         "expose_headers": ["Content-Type", "Authorization"],
         "supports_credentials": True,
         "max_age": 3600
     }},
     supports_credentials=True,
     automatic_options=True)

# Setup comprehensive logging
logger = setup_logging(app)
api_logger = logging.getLogger('api')
auth_logger = logging.getLogger('auth')

# Initialize database
init_db()

# Initialize ML models (optional - only if ModelService is available)
if MODEL_SERVICE_AVAILABLE and ModelService:
    try:
        model_service = ModelService()
        model_service.load_models()
        setattr(app, 'model_service', model_service)
        logger.info("ML models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to load ML models: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.warning("Running without ML models - emotion detection will not work")
        setattr(app, 'model_service', None)
else:
    logger.warning("ModelService not available - running without ML models")
    if not MODEL_SERVICE_AVAILABLE:
        logger.error("ModelService import failed - check if all dependencies are installed")
    logger.info("Backend will work for authentication and basic features")
    setattr(app, 'model_service', None)
```

### Blueprint Organization

Routes are organized into blueprints by domain:

```130:137:backend/app.py
# Register blueprints
app.register_blueprint(auth.bp, url_prefix='/api/auth')
app.register_blueprint(videos.bp, url_prefix='/api/videos')
app.register_blueprint(sessions.bp, url_prefix='/api/sessions')
app.register_blueprint(emotions.bp, url_prefix='/api/emotions')
app.register_blueprint(reports.bp, url_prefix='/api/reports')
app.register_blueprint(interventions.bp, url_prefix='/api/interventions')
app.register_blueprint(live_sessions_routes.bp, url_prefix='/api/live-sessions')
```

### Service Layer

The `ModelService` class in `backend/services/model_service.py` is the core service for ML operations:

```17:38:backend/services/model_service.py
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
```

### Database Layer

Database operations are handled through `backend/services/database.py` with connection pooling and query execution utilities.

---

## Model Loading Pipeline

### Initialization Flow

The model loading process starts when the Flask app initializes:

```40:356:backend/services/model_service.py
def load_models(self):
    """Load all trained models"""
    try:
        # Load YOLOv8 model for face detection
        # Priority: ONNX face model > PyTorch face model > default YOLOv8n
        yolo_model_path = os.getenv('YOLO_MODEL_PATH', None)
        yolo_loaded = False
        
        # ... YOLO model search and loading logic ...
        
        # ----------------- LOAD EMOTION MODEL (HuggingFace MobileNetV2) -----------------
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification
        
        cnn_model_path = os.getenv("CNN_MODEL_PATH", "models/mobilenetv2_emotion_model_CPU.pkl")
        self.model_type = "mobilenetv2"
        self.use_onnx = False
        
        # Search for .pkl file if path doesn't exist
        if not os.path.exists(cnn_model_path):
            # ... search paths ...
        
        # Load pickle file (use torch.load to handle CUDA->CPU mapping)
        payload = torch.load(cnn_model_path, map_location='cpu', weights_only=False)
        
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
        
        # Load HuggingFace image processor FIRST (needed for preprocessing)
        self._load_hf_processor()
        
        # Create debug directory
        self._setup_debug_dir()
        
        self.models_loaded = True
```

### YOLO Model Loading

YOLO models are loaded with priority: ONNX face model > PyTorch face model > default YOLOv8n:

```138:230:backend/services/model_service.py
# Try to load YOLO model
if yolo_model_path and os.path.exists(yolo_model_path):
    file_size_mb = os.path.getsize(yolo_model_path) / (1024 * 1024)
    if yolo_model_path.endswith('.onnx'):
        # Load ONNX model
        import onnxruntime as ort
        providers = ['CPUExecutionProvider']
        if self.device == 'cuda':
            try:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            except:
                logger.warning("CUDA provider not available for YOLO, using CPU")
        
        self.yolo_onnx_session = ort.InferenceSession(yolo_model_path, providers=providers)
        self.use_yolo_onnx = True
        yolo_loaded = True
    else:
        # Load PyTorch model
        self.yolo_model = YOLO(yolo_model_path)
        yolo_loaded = True
```

### Device Detection

The system automatically detects and uses GPU if available:

```31:31:backend/services/model_service.py
self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

---

## HuggingFace Integration

### AutoImageProcessor Loading

The HuggingFace image processor is loaded to handle image preprocessing:

```379:420:backend/services/model_service.py
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
    
    # Fallback to checkpoint-based processor
    try:
        self.image_processor = AutoImageProcessor.from_pretrained(self.hf_checkpoint)
    except Exception as e:
        logger.error(f"Failed to load processor from HF checkpoint {self.hf_checkpoint}: {str(e)}")
        self.image_processor = None
```

### Model Configuration

The emotion model uses HuggingFace's AutoModelForImageClassification:

```290:303:backend/services/model_service.py
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
```

### Checkpoint Handling

The system uses `google/mobilenet_v2_1.0_224` as the base checkpoint:

```283:283:backend/services/model_service.py
checkpoint = payload.get("checkpoint", "google/mobilenet_v2_1.0_224")
```

---

## YOLO Integration

### Face Detection Workflow

YOLO is used for face detection in images:

```434:466:backend/services/model_service.py
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
    return faces
```

### ONNX Face Detection

ONNX models use ONNX Runtime for inference:

```468:900:backend/services/model_service.py
def _detect_faces_onnx(self, image: np.ndarray) -> List[Dict]:
    """Detect faces using ONNX YOLO model"""
    try:
        import onnxruntime as ort
        
        # Preprocess image for YOLO (resize to 640x640, normalize)
        original_shape = image.shape[:2]
        input_size = 640
        resized = cv2.resize(image, (input_size, input_size))
        
        # Convert BGR to RGB and normalize
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        resized = resized.astype(np.float32) / 255.0
        
        # Transpose to (1, 3, H, W) format
        input_tensor = resized.transpose(2, 0, 1)[np.newaxis, ...]
        
        # Run inference
        outputs = session.run(output_names, {input_name: input_tensor})
        
        # Process outputs and extract bounding boxes
        # ... (complex output processing logic)
```

### PyTorch Face Detection

PyTorch models use Ultralytics YOLO:

```928:957:backend/services/model_service.py
def _detect_faces_pytorch(self, image: np.ndarray) -> List[Dict]:
    """Detect faces using PyTorch YOLO model"""
    try:
        if self.yolo_model is None:
            raise ValueError("YOLOv8 PyTorch model is not initialized")
        
        yolo_model = self.yolo_model
        
        # Run YOLOv8 inference
        results = yolo_model(image, conf=0.3, verbose=False)
        
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
```

---

## Inference Flow

### Complete Pipeline

The inference pipeline follows these steps:

1. **Image Input** → Base64 decode or file upload
2. **Face Detection** → YOLO detects faces
3. **Face Preprocessing** → Extract and normalize face region
4. **Emotion Prediction** → MobileNetV2 predicts emotions
5. **Post-processing** → Calculate scores and map to states
6. **Database Storage** → Save results

### Face Preprocessing

Faces are extracted and preprocessed for emotion classification:

```959:1119:backend/services/model_service.py
def preprocess_face(self, image: np.ndarray, bbox: List[int], target_size: Optional[Tuple[int, int]] = None) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Extract and preprocess face region for emotion classification
    For MobileNetV2/EfficientNet models, use 224x224 RGB images
    """
    x1, y1, x2, y2 = bbox
    
    # Validate and clamp bounding box
    x1_orig = max(0, int(x1))
    y1_orig = max(0, int(y1))
    x2_orig = min(image.shape[1], int(x2))
    y2_orig = min(image.shape[0], int(y2))
    
    # Add padding to bounding box
    pad = int(0.3 * min(face_width, face_height))
    
    # Extract face region with padding
    face_roi = image[y1:y2, x1:x2]
    
    # Use HuggingFace processor if available
    if self.image_processor is not None:
        rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        inputs = self.image_processor(images=pil_img, return_tensors="np")
        face_tensor = inputs["pixel_values"]  # NCHW float32 normalized
        return face_tensor, None
    
    # Manual preprocessing fallback
    # ... (resize, normalize, convert to tensor)
```

### Emotion Prediction

Emotions are predicted using the loaded MobileNetV2 model:

```1121:1267:backend/services/model_service.py
def predict_emotion(self, face_image: np.ndarray) -> Union[Dict[str, float], str]:
    """
    Predict emotion from preprocessed face image using CNN model
    """
    if self.cnn_model is None:
        logger.error("[EmotionPrediction] CNN model not loaded")
        return "model_not_confident"
    
    try:
        # PyTorch model check (including HuggingFace models)
        if isinstance(self.cnn_model, torch.nn.Module):
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
            return emotions
```

### Concentration Score Calculation

Concentration scores are calculated from emotion probabilities:

```1419:1467:backend/services/model_service.py
def calculate_concentration_score(self, emotions: Dict[str, float]) -> float:
    """
    Calculate concentration score based on emotion probabilities
    attentive = neutral, happy, surprise, focus
    inattentive = sleepy, boredom, frustration, confusion
    """
    attentive_emotions = ['neutral', 'happy', 'surprise']
    inattentive_emotions = ['sleepy', 'boredom', 'frustration', 'confusion']
    
    # Map detected emotions to categories
    emotion_mapping = {
        'angry': 'frustration',
        'disgust': 'boredom',
        'fear': 'confusion',
        'sad': 'confusion',
        'happy': 'happy',
        'surprise': 'surprise',
        'neutral': 'neutral',
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
        concentration = 50.0
    
    return max(0.0, min(100.0, concentration))
```

---

## PKL/PT Model Usage

### PKL File Structure

PKL files contain serialized PyTorch model state with metadata:

```269:315:backend/services/model_service.py
# Load pickle file (use torch.load to handle CUDA->CPU mapping)
payload = torch.load(cnn_model_path, map_location='cpu', weights_only=False)

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
```

### PKL File Contents

The PKL file contains:
- **`state_dict`**: PyTorch model weights
- **`labels`**: List of emotion class names
- **`id2label`**: Mapping from class ID to label
- **`label2id`**: Mapping from label to class ID
- **`checkpoint`**: HuggingFace checkpoint name

### Model File Locations

- **Primary**: `backend/backend/models/mobilenetv2_emotion_model_CPU.pkl`
- **Alternative**: `backend/backend/models/mobilenetv2_emotion_model.pkl`
- **PyTorch**: `backend/backend/models/mobilenetv2_best.pt`
- **ONNX**: `backend/backend/models/mobilenetv2.onnx`

### Loading Process

1. Search for PKL file in multiple locations
2. Load with `torch.load()` using `map_location='cpu'` for CPU compatibility
3. Extract state_dict and metadata
4. Reconstruct HuggingFace model from checkpoint
5. Load state_dict into model
6. Set model to evaluation mode

---

## API Endpoints

### Authentication Endpoints

**POST `/api/auth/register`**
- Register new user (teacher or student)
- Request: `{ email, password, firstName, lastName, role }`
- Response: `{ message, user: { id, email, role } }`

**POST `/api/auth/login`**
- Authenticate user and get JWT token
- Request: `{ email, password }`
- Response: `{ token, user: { id, email, role, firstName, lastName } }`

**GET `/api/auth/me`**
- Get current user info (requires JWT)
- Response: `{ id, email, role, firstName, lastName }`

### Emotion Detection Endpoints

**POST `/api/emotions/detect`**
- Detect emotions in a single image frame
- Request: `{ image: base64, sessionType, sessionId, timestamp }`
- Response: `{ emotion, confidence, concentration, probs, bbox }`

```18:114:backend/routes/emotions.py
@bp.route('/detect', methods=['POST'])
@jwt_required()
def detect_emotions():
    """
    Receive image frame from frontend and return emotion predictions
    Expected: base64 encoded image or image file
    """
    try:
        current_user = get_current_user()
        user_id = current_user['id']
        
        # Get model service from app context
        model_service = current_app.model_service
        if not model_service:
            return jsonify({'error': 'ML models not loaded'}), 503
        
        # Get image data
        data = request.get_json()
        
        if 'image' in data:
            # Base64 encoded image
            image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect faces
        faces = model_service.detect_faces(image)
        
        if not faces:
            return jsonify({
                'emotions': [],
                'message': 'No faces detected'
            })
        
        # Use first detected face
        face = faces[0]
        box = face['bbox']
        
        # Preprocess face
        face_tensor, preprocess_error = model_service.preprocess_face(image, box)
        
        # Predict emotions
        emotion_probs = model_service.predict_emotion_vector(face_tensor)
        emotion_label = max(emotion_probs, key=emotion_probs.get)
        concentration = model_service.compute_concentration(emotion_probs)
        
        return jsonify({
            'emotion': emotion_state,
            'confidence': float(confidence),
            'concentration': float(concentration),
            'probs': emotion_probs,
            'bbox': box
        })
```

**POST `/api/emotions/stream`**
- Stream emotion detection for real-time monitoring
- Request: `{ image: base64, sessionType, sessionId, timestamp }`
- Response: `{ emotion, confidence, concentration_score, probs, bbox, timestamp }`
- Automatically stores results in database

**GET `/api/emotions/<session_type>/<session_id>`**
- Get emotion timeline for a session
- Response: `{ emotionData: [{ id, emotion, confidence, timestamp, engagement_score }] }`

### Live Session Endpoints

**POST `/api/live-sessions/create`**
- Create a new live session (teacher only)
- Request: `{ title, meetUrl, startTime }`
- Response: `{ session: { id }, message }`

**GET `/api/live-sessions/available`**
- Get all available live sessions
- Response: `{ sessions: [{ id, title, description, meet_url, scheduled_at, status }] }`

**POST `/api/live-sessions/<session_id>/stream`**
- Stream emotion detection for live session
- Request: `{ image: base64, timestamp }`
- Response: `{ emotion, confidence, concentration_score, engagement_score, probs, bbox, timestamp }`

```61:86:backend/live_sessions/controller.py
def stream_emotion(session_id: str):
    """POST /live-sessions/:id/stream - Stream emotion detection for live session"""
    try:
        current_user = get_current_user()
        student_id = current_user['id']
        
        # Get image data
        data = request.get_json() if request.is_json else {}
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Process emotion detection (service will check for model service)
        result = LiveSessionService.stream_emotion(session_id, student_id, data)
        
        return jsonify(result), 200
```

**GET `/api/live-sessions/<session_id>/report`**
- Get analytics report for a live session
- Response: `{ report: { session, total_students, total_logs, overall_avg_engagement, students: [...] } }`

### Video Endpoints

**POST `/api/videos/upload`**
- Upload a video file for analysis
- Request: Multipart form data with video file
- Response: `{ video: { id, title, file_path, duration } }`

**GET `/api/videos`**
- Get list of videos (filtered by user role)
- Response: `{ videos: [{ id, title, description, duration, created_at }] }`

**GET `/api/videos/<video_id>`**
- Get video details
- Response: `{ video: { id, title, description, file_path, duration } }`

### Report Endpoints

**GET `/api/reports/dashboard/all`**
- Get dashboard data with engagement statistics
- Response: `{ totalSessions, totalStudents, avgEngagement, recentSessions }`

**POST `/api/reports/generate/<session_type>/<session_id>`**
- Generate engagement report for a session
- Response: `{ report: { id, overall_engagement, average_emotion, engagement_drops, emotional_timeline } }`

---

## Frontend Architecture

### React Application Structure

The frontend is a React TypeScript application organized into:

- **Pages**: Route-level components (`LiveSession.tsx`, `VideoPlayer.tsx`, etc.)
- **Components**: Reusable UI components (`EngagementMeter.tsx`)
- **Hooks**: Custom React hooks for business logic
- **Services**: API client and service layers
- **Contexts**: React context providers (`AuthContext.tsx`)

### Live Session Component

The live session page handles real-time emotion monitoring:

```10:273:client/src/pages/LiveSession.tsx
const LiveSession: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { user } = useAuth();
  const webcamRef = useRef<HTMLVideoElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const [session, setSession] = useState<any>(null);
  const [emotionDetectionEnabled, setEmotionDetectionEnabled] = useState(false);

  // Use live session emotion stream hook (NO interventions)
  const {
    emotionResult,
    isDetecting,
    error: emotionError,
    startDetection,
    stopDetection,
  } = useLiveSessionEmotionStream({
    videoElement: webcamRef.current,
    sessionId: id || '',
    interval: 5000, // 5 seconds for live sessions
    enabled: emotionDetectionEnabled && !!id,
  });

  // ... component logic ...
```

### Emotion Streaming Hook

The `useLiveSessionEmotionStream` hook handles frame capture and API calls:

```21:222:client/src/hooks/useLiveSessionEmotionStream.ts
export const useLiveSessionEmotionStream = ({
  videoElement,
  sessionId,
  interval = 5000,
  enabled = true,
}: UseLiveSessionEmotionStreamOptions) => {
  const [emotionResult, setEmotionResult] = useState<LiveEmotionStreamResult | null>(null);
  const [isDetecting, setIsDetecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Capture frame and send to backend
  const captureAndDetect = useCallback(async () => {
    if (!videoElement || !sessionId || !isDetectingRef.current) {
      return;
    }

    // Create canvas and capture frame
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    // Get image data
    const imageData = canvas.toDataURL('image/jpeg', 0.8);

    // Calculate timestamp
    const currentTimestamp = Math.floor((Date.now() - startTimeRef.current) / 1000);

    // Send to live session stream endpoint
    const response: LiveSessionStreamResponse = await liveSessionsApi.stream(sessionId, {
      image: imageData,
      timestamp: currentTimestamp,
    });

    // Update state with result
    setEmotionResult({
      emotion: response.emotion,
      confidence: response.confidence,
      concentrationScore: response.concentration_score,
      timestamp: response.timestamp,
      bbox: response.bbox,
      allEmotions: (response as any).probs,
    });
  }, [videoElement, sessionId]);
```

### API Service Layer

The API service uses Axios with interceptors for authentication:

```1:93:client/src/services/api.ts
import axios, { AxiosError } from 'axios';
import logger from '../utils/logger';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:5000/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add token dynamically
api.interceptors.request.use(
  (config) => {
    // Add token from localStorage dynamically for each request
    let token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    // Don't set Content-Type for FormData
    if (config.data instanceof FormData) {
      delete config.headers['Content-Type'];
    }
    return config;
  },
  (error) => {
    logger.error('API Request Error', error);
    return Promise.reject(error);
  }
);
```

---

## UI Workflow

### Live Session Flow

1. **Join Session**: Student navigates to live session page
2. **Start Monitoring**: Click "Start Monitoring" button
3. **Webcam Access**: Browser requests camera permission
4. **Frame Capture**: Hook captures frames every 5 seconds
5. **API Call**: Frame sent to `/api/live-sessions/<id>/stream`
6. **Backend Processing**: Face detection → Emotion prediction → Database storage
7. **Response**: Emotion data returned to frontend
8. **Display**: Results shown in EngagementMeter component with bounding box overlay

### Video Playback Flow

1. **Upload Video**: Teacher uploads video file
2. **Video Processing**: Backend stores video file
3. **Join Session**: Student joins recorded video session
4. **Video Playback**: Video plays in VideoPlayer component
5. **Frame Analysis**: Frames extracted and analyzed during playback
6. **Emotion Tracking**: Emotions tracked over time
7. **Report Generation**: Engagement report generated on completion
8. **View Report**: Teacher/student views analytics report

### Real-time Emotion Display

The UI displays:
- **EngagementMeter**: Visual gauge showing concentration score
- **Bounding Box**: Green rectangle around detected face
- **Emotion Label**: Current emotion with confidence percentage
- **All Emotions**: Grid showing all emotion probabilities
- **Timestamp**: Time elapsed in session

---

## Database Schema

### Tables

**users**
- Stores user accounts (teachers and students)
- Fields: `id`, `email`, `password_hash`, `first_name`, `last_name`, `role`, `created_at`

**videos**
- Stores uploaded video files
- Fields: `id`, `teacher_id`, `title`, `description`, `file_path`, `duration`, `created_at`
- Foreign key: `teacher_id` → `users.id`

**live_sessions**
- Stores live session information
- Fields: `id`, `teacher_id`, `title`, `description`, `meet_url`, `scheduled_at`, `started_at`, `ended_at`, `status`
- Foreign key: `teacher_id` → `users.id`

**emotion_data**
- Stores emotion detection results
- Fields: `id`, `session_type`, `session_id`, `student_id`, `emotion`, `confidence`, `timestamp`, `engagement_score`
- Foreign key: `student_id` → `users.id`

**engagement_reports**
- Stores generated engagement reports
- Fields: `id`, `session_type`, `session_id`, `student_id`, `overall_engagement`, `average_emotion`, `engagement_drops`, `emotional_timeline`
- Foreign key: `student_id` → `users.id`

**live_session_logs**
- Stores live session emotion logs (separate from emotion_data)
- Fields: `id`, `session_id`, `student_id`, `emotion`, `confidence`, `engagement_score`, `concentration_score`, `timestamp`

**interventions**
- Stores triggered interventions
- Fields: `id`, `session_id`, `student_id`, `intervention_type`, `triggered_at`, `triggered_emotion`, `completed_at`
- Foreign key: `student_id` → `users.id`

### Relationships

```
users (1) ──< (many) videos
users (1) ──< (many) live_sessions
users (1) ──< (many) emotion_data
users (1) ──< (many) engagement_reports
users (1) ──< (many) interventions
live_sessions (1) ──< (many) live_session_logs
```

### Indexes

Indexes are created on:
- `users.email` (unique)
- `users.role`
- `emotion_data(session_type, session_id)`
- `emotion_data.student_id`
- `emotion_data.timestamp`
- `engagement_reports(session_type, session_id)`

---

## Code Explanations

### Model Service Initialization

The ModelService is initialized when the Flask app starts:

```76:93:backend/app.py
# Initialize ML models (optional - only if ModelService is available)
if MODEL_SERVICE_AVAILABLE and ModelService:
    try:
        model_service = ModelService()
        model_service.load_models()
        setattr(app, 'model_service', model_service)
        logger.info("ML models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to load ML models: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.warning("Running without ML models - emotion detection will not work")
        setattr(app, 'model_service', None)
else:
    logger.warning("ModelService not available - running without ML models")
    if not MODEL_SERVICE_AVAILABLE:
        logger.error("ModelService import failed - check if all dependencies are installed")
    logger.info("Backend will work for authentication and basic features")
    setattr(app, 'model_service', None)
```

### Emotion Detection Endpoint Flow

The `/api/emotions/stream` endpoint processes images:

```120:255:backend/routes/emotions.py
@bp.route('/stream', methods=['POST'])
@jwt_required()
def stream_emotions():
    """
    Stream endpoint for real-time emotion detection
    """
    try:
        current_user = get_current_user()
        user_id = current_user['id']
        
        # Get model service from app context
        model_service = current_app.model_service
        if not model_service:
            return jsonify({'error': 'ML models not loaded'}), 503
        
        # Get image data
        data = request.get_json() if request.is_json else {}
        
        # Decode base64 image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Detect faces
        faces = model_service.detect_faces(image)
        
        if not faces:
            return jsonify({
                'emotion': 'neutral',
                'confidence': 0.0,
                'concentration_score': 50.0,
                'timestamp': timestamp,
                'message': 'No faces detected',
                'bbox': None
            })
        
        # Use first detected face
        face = faces[0]
        box = face['bbox']
        
        # Preprocess face
        face_tensor, preprocess_error = model_service.preprocess_face(image, box)
        
        # Predict emotions
        emotion_probs = model_service.predict_emotion_vector(face_tensor)
        emotion_label = max(emotion_probs, key=emotion_probs.get)
        concentration = model_service.compute_concentration(emotion_probs)
        
        # Store in database
        execute_query(
            """INSERT INTO emotion_data 
               (id, session_type, session_id, student_id, emotion, confidence, timestamp, engagement_score)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (emotion_id, session_type, session_id, user_id, emotion_state, confidence, timestamp, concentration / 100.0)
        )
        
        return jsonify({
            'emotion': emotion_state,
            'confidence': float(confidence),
            'concentration': float(concentration),
            'probs': emotion_probs,
            'bbox': box,
            'timestamp': timestamp
        })
```

### Live Session Service Flow

Live sessions use a separate service that stores logs differently:

```37:128:backend/live_sessions/service.py
@staticmethod
def stream_emotion(session_id: str, student_id: str, image_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process emotion detection for live session stream
    Reuses ModelService but WITHOUT intervention logic
    """
    # Get model service from app context
    model_service = current_app.model_service
    if not model_service:
        raise Exception('ML models not loaded')
    
    # Parse image
    image_str = image_data['image']
    image_data_str = image_str.split(',')[1] if ',' in image_str else image_str
    image_bytes = base64.b64decode(image_data_str)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Detect faces
    faces = model_service.detect_faces(image)
    
    # Use first detected face
    face = faces[0]
    box = face['bbox']
    
    # Preprocess face
    face_tensor, preprocess_error = model_service.preprocess_face(image, box)
    
    # Predict emotions
    emotion_probs = model_service.predict_emotion_vector(face_tensor)
    emotion_label = max(emotion_probs, key=emotion_probs.get)
    concentration = model_service.compute_concentration(emotion_probs)
    
    # Save to live_session_logs table (NOT emotion_data)
    LiveSessionLog.create(
        session_id=session_id,
        student_id=student_id,
        emotion=emotion_state,
        confidence=float(confidence),
        engagement_score=engagement_score,
        concentration_score=float(concentration),
        timestamp=timestamp
    )
    
    return {
        'emotion': emotion_state,
        'confidence': float(confidence),
        'concentration_score': float(concentration),
        'probs': emotion_probs,
        'bbox': box,
        'timestamp': timestamp
    }
```

### Error Handling

The system includes comprehensive error handling:

- **Model Loading Errors**: Logged and app continues without models
- **Face Detection Errors**: Returns default neutral emotion
- **Preprocessing Errors**: Returns error message in response
- **Database Errors**: Logged but don't block response
- **API Errors**: Proper HTTP status codes returned

---

## Environment Variables

### Backend Configuration

- `JWT_SECRET`: Secret key for JWT token signing
- `DB_HOST`: MySQL host (default: localhost)
- `DB_NAME`: Database name (default: emotion_monitoring)
- `DB_USER`: Database username
- `DB_PASSWORD`: Database password
- `CNN_MODEL_PATH`: Path to emotion model PKL file
- `YOLO_MODEL_PATH`: Path to YOLO face detection model
- `UPLOAD_DIR`: Directory for uploaded files
- `FLASK_ENV`: Flask environment (development/production)

### Frontend Configuration

- `REACT_APP_API_URL`: Backend API URL (default: http://localhost:5000/api)

---

## Model Files Summary

### Emotion Models
- **Primary**: `backend/backend/models/mobilenetv2_emotion_model_CPU.pkl` (HuggingFace MobileNetV2)
- **Alternative**: `backend/backend/models/mobilenetv2_emotion_model.pkl`
- **PyTorch**: `backend/backend/models/mobilenetv2_best.pt`
- **ONNX**: `backend/backend/models/mobilenetv2.onnx`

### Face Detection Models
- **Primary**: `backend/backend/models/yolov8n-face.onnx` (ONNX format)
- **Fallback**: `backend/yolov8n.pt` (Default YOLOv8n)

### Processor Configuration
- **Location**: `backend/backend/models/mobilenetv2_processor/preprocessor_config.json`
- **Checkpoint**: `google/mobilenet_v2_1.0_224`

---

## Conclusion

This documentation provides a complete overview of the Emotion Monitoring System architecture, covering:

- System components and technology stack
- Complete folder structure
- Backend Flask architecture
- Model loading and inference pipelines
- HuggingFace and YOLO integrations
- API endpoints and workflows
- Frontend React architecture
- Database schema
- Code explanations with references

For additional details, refer to the specific code files referenced throughout this document.




