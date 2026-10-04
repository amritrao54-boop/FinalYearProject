import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from cnn_model import predict_disease_from_image as original_predict

import cv2
import numpy as np

def generate_gradcam(image_path: str, save_dir="uploads/gradcam") -> str:
    """
    Generates a Grad-CAM heatmap for Explainable AI.
    If the model is the fallback, it returns the original image.
    """
    import cnn_model
    os.makedirs(save_dir, exist_ok=True)
    
    if cnn_model._mode != 'trained' or cnn_model._model is None:
        return image_path
        
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image as keras_image
    
    model = cnn_model._model
    # Find last conv layer (this assumes a standard MobileNetV2 structure or similar)
    last_conv_layer = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4: # Typically (None, H, W, C)
            last_conv_layer = layer.name
            break
            
    if not last_conv_layer:
        return image_path
        
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer).output, model.output]
    )
    
    img = keras_image.load_img(image_path, target_size=cnn_model.IMG_SIZE_TRAINED)
    x = keras_image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(x)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
        
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    original_img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + original_img
    
    base_name = os.path.basename(image_path)
    gradcam_path = os.path.join(save_dir, f"cam_{base_name}")
    cv2.imwrite(gradcam_path, superimposed_img)
    
    return gradcam_path

def analyze_cattle_image(image_path: str):
    """
    Analyzes a cattle image using the existing CNN model and generates XAI Grad-CAM.
    """
    prediction, confidence = original_predict(image_path)
    gradcam_img = generate_gradcam(image_path)
    
    return {
        "prediction": prediction,
        "probabilities": {},
        "confidence": confidence,
        "model": "MobileNetV2",
        "gradcam_path": gradcam_img
    }
