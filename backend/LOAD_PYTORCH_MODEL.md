# Loading PyTorch (.pt) Models

## Your Model File

Your EfficientNetB0 model is located at:
```
backend/models/efficientnetb0_emotion_best.pt
```

## Automatic Detection

The model service will automatically detect and load your `.pt` file. It searches for the model in these locations (in order):

1. `backend/models/efficientnetb0_emotion_best.pt`
2. `models/efficientnetb0_emotion_best.pt`
3. `backend/backend/models/efficientnetb0_emotion_best.pt`
4. `models/emotion_cnn_model.pt`
5. `models/emotion_cnn_model.pkl`

## Manual Configuration (Optional)

You can also set the `CNN_MODEL_PATH` environment variable in your `.env` file:

```env
CNN_MODEL_PATH=backend/models/efficientnetb0_emotion_best.pt
```

## Model Requirements

The code has been updated to:
- ✅ Load `.pt` (PyTorch) model files
- ✅ Handle EfficientNetB0 preprocessing (224x224 RGB images)
- ✅ Automatically detect model type and adjust preprocessing
- ✅ Support both GPU and CPU inference

## Testing

After starting the backend, check the logs to see if the model loaded successfully:

```bash
# Look for these log messages:
# "PyTorch CNN model loaded from backend/models/efficientnetb0_emotion_best.pt"
# "Model type: <class 'torch.nn.modules...'>"
# "All models loaded successfully"
```

## Emotion Classes

The model expects these emotion classes (adjust if your model uses different classes):
- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

If your model uses different classes, update the `emotion_classes` list in `predict_emotion()` method.














