"""
Full pipeline test script for emotion detection
Tests: face detection, preprocessing, logits, and final emotion prediction
"""
import os
import sys
import cv2
import numpy as np
from PIL import Image
import logging

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.model_service import ModelService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_emotion_pipeline():
    """Test the complete emotion detection pipeline"""
    
    print("\n" + "="*70)
    print("EMOTION DETECTION PIPELINE TEST")
    print("="*70 + "\n")
    
    # Initialize model service
    print("1. Loading models...")
    model_service = ModelService()
    try:
        model_service.load_models()
        print("✓ Models loaded successfully\n")
    except Exception as e:
        print(f"✗ Error loading models: {str(e)}")
        return
    
    # Load sample image or create test image
    print("2. Loading test image...")
    test_image_paths = [
        'test_face.jpg',
        'test_image.jpg',
        'sample_face.png',
        os.path.join('backend', 'test_face.jpg'),
        os.path.join('backend', 'backend', 'test_face.jpg'),
    ]
    
    test_image = None
    for path in test_image_paths:
        if os.path.exists(path):
            test_image = cv2.imread(path)
            print(f"✓ Loaded test image from: {path}")
            print(f"  Image shape: {test_image.shape}")
            break
    
    if test_image is None:
        print("⚠ No test image found, creating synthetic test image...")
        # Create a synthetic test image (gray square)
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
        # Add a simple "face" region (darker rectangle)
        test_image[150:350, 200:400] = 80
        print(f"  Created synthetic image: {test_image.shape}")
    
    print()
    
    # Step 3: Face Detection
    print("3. Face Detection...")
    faces = model_service.detect_faces(test_image)
    print(f"   Detected {len(faces)} face(s)")
    
    if len(faces) == 0:
        print("   ⚠ No faces detected - cannot continue pipeline test")
        print("   Please provide an image with a detectable face")
        return
    
    face = faces[0]
    bbox = face['bbox']
    print(f"   Face bbox: {bbox}")
    print(f"   Face confidence: {face['confidence']:.4f}\n")
    
    # Step 4: Face Preprocessing
    print("4. Face Preprocessing...")
    face_image, preprocess_error = model_service.preprocess_face(test_image, bbox)
    
    if preprocess_error is not None:
        print(f"   ✗ Preprocessing error: {preprocess_error}")
        return
    
    print(f"   ✓ Face preprocessed successfully")
    print(f"   Preprocessed shape: {face_image.shape}")
    print(f"   Preprocessed dtype: {face_image.dtype}")
    print(f"   Preprocessed range: [{face_image.min():.4f}, {face_image.max():.4f}]")
    print(f"   Preprocessed mean: {face_image.mean():.4f}, std: {face_image.std():.4f}\n")
    
    # Step 5: Model Inference (before softmax)
    print("5. Model Inference (Logits before softmax)...")
    
    # Get raw logits by accessing model directly
    try:
        if model_service.use_onnx:
            import onnxruntime as ort
            if isinstance(model_service.cnn_model, ort.InferenceSession):
                input_name = model_service.cnn_model.get_inputs()[0].name
                output_name = model_service.cnn_model.get_outputs()[0].name
                
                if face_image.dtype != np.float32:
                    face_image = face_image.astype(np.float32)
                
                outputs = model_service.cnn_model.run([output_name], {input_name: face_image})
                logits = np.asarray(outputs[0])
                
                if logits.ndim > 1:
                    logits = logits[0]
                
                print(f"   ✓ ONNX inference complete")
                print(f"   Logits shape: {logits.shape}")
                print(f"   Logits: {logits}")
                print(f"   Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
                print(f"   Logits mean: {logits.mean():.4f}, std: {logits.std():.4f}\n")
        elif isinstance(model_service.cnn_model, type(model_service.cnn_model)) and hasattr(model_service.cnn_model, '__call__'):
            import torch
            with torch.no_grad():
                if len(face_image.shape) == 4:
                    input_tensor = torch.from_numpy(face_image).float()
                else:
                    input_tensor = torch.from_numpy(face_image).float().unsqueeze(0)
                
                if model_service.device == 'cuda':
                    input_tensor = input_tensor.to(model_service.device)
                    model_service.cnn_model = model_service.cnn_model.to(model_service.device)
                
                logits = model_service.cnn_model(input_tensor)
                
                if isinstance(logits, torch.Tensor):
                    logits = logits.cpu().numpy()
                    if logits.ndim > 1:
                        logits = logits[0]
                
                print(f"   ✓ PyTorch inference complete")
                print(f"   Logits shape: {logits.shape}")
                print(f"   Logits: {logits}")
                print(f"   Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
                print(f"   Logits mean: {logits.mean():.4f}, std: {logits.std():.4f}\n")
        else:
            print("   ⚠ Cannot extract raw logits for this model type")
            logits = None
    except Exception as e:
        print(f"   ⚠ Error extracting logits: {str(e)}")
        logits = None
    
    # Step 6: Final Emotion Prediction
    print("6. Final Emotion Prediction (after softmax)...")
    emotions = model_service.predict_emotion(face_image)
    
    if emotions == "model_not_confident":
        print("   ✗ Model not confident - validation failed")
        print("   This indicates the model output was too uniform")
    else:
        print("   ✓ Emotion prediction complete")
        print(f"   Emotion probabilities:")
        for emotion, prob in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
            print(f"     {emotion}: {prob:.4f}")
        
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        print(f"\n   Dominant emotion: {dominant_emotion[0]} (confidence: {dominant_emotion[1]:.4f})")
        
        # Calculate scores
        engagement_score = model_service._calculate_engagement(emotions, dominant_emotion[0])
        concentration_score = model_service.calculate_concentration_score(emotions)
        print(f"   Engagement score: {engagement_score:.2f}")
        print(f"   Concentration score: {concentration_score:.2f}")
    
    print("\n" + "="*70)
    print("PIPELINE TEST COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_emotion_pipeline()



