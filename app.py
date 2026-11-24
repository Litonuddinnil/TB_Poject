
import os
import pickle
import gradio as gr
import cv2
import numpy as np

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms


# ==========================================
# 1. Configuration
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "TB_CNN_Model.pkl"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)


# ==========================================
# 2. CNN Architecture
# Same architecture as train.py
# ==========================================

class MyCNN(nn.Module):

    def __init__(self, num_classes):

        super(MyCNN, self).__init__()

        # Feature Extraction
        self.features = nn.Sequential(

            # Convolution Block 1
            nn.Conv2d(
                3, 32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm2d(32),

            nn.MaxPool2d(2),

            # Convolution Block 2
            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm2d(64),

            nn.MaxPool2d(2),

            # Convolution Block 3
            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm2d(128),

            nn.MaxPool2d(2)
        )

        # Classification Layer
        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 16 * 16, 128),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ==========================================
# 3. Load Previously Trained Model
# ==========================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}\n"
        "Make sure TB_CNN_Model.pkl is inside the project folder."
    )


print("\nLoading trained CNN model...")


# Load trusted pickle checkpoint
with open(MODEL_PATH, "rb") as file:

    checkpoint = pickle.load(file)


# Load saved information
CLASSES = checkpoint["classes"]

IMG_SIZE = checkpoint["img_size"]


# Initialize CNN architecture
# No training is performed here
model = MyCNN(
    num_classes=len(CLASSES)
)


# Load trained weights
model.load_state_dict(
    checkpoint["model_state_dict"]
)


# Move model to CPU/GPU
model = model.to(device)

# Evaluation mode
model.eval()


print("Model loaded successfully!")

print("Classes:", CLASSES)

print("Image Size:", IMG_SIZE)


# ==========================================
# 4. Load Saved Evaluation Metrics
# ==========================================

accuracy = checkpoint.get("accuracy", 0)

precision = checkpoint.get("precision", 0)

recall = checkpoint.get("recall", 0)

f1_score = checkpoint.get("f1_score", 0)


print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy:  {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall:    {recall:.4f}")

print(f"F1 Score:  {f1_score:.4f}")


# ==========================================
# 5. DIP Image Preprocessing
# Same preprocessing as train.py
# ==========================================

def preprocess_xray(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Convert PIL image to NumPy
    image = np.array(image)

    # Convert RGB to Grayscale for DIP
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # Median Filtering
    filtered = cv2.medianBlur(gray, 3)

    # CLAHE Contrast Enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(filtered)

    # Resize Image
    resized = cv2.resize(
        enhanced,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # Convert Grayscale back to RGB
    rgb_image = cv2.cvtColor(
        resized,
        cv2.COLOR_GRAY2RGB
    )

    # Convert NumPy to PIL
    processed_image = Image.fromarray(rgb_image)

    # Transform
    transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    image_tensor = transform(processed_image)

    # Add Batch Dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor
# ==========================================
# 6. Prediction Function
# ==========================================

def predict_tb(image):

    if image is None:

        return None, "Please upload a chest X-ray image."

    try:

        # Apply DIP preprocessing
        image_tensor = preprocess_xray(image)

        # Move image to CPU/GPU
        image_tensor = image_tensor.to(device)

        # Prediction - No Training
        with torch.no_grad():

            outputs = model(image_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )[0]

        # Get predicted class
        predicted_index = torch.argmax(
            probabilities
        ).item()

        predicted_class = CLASSES[predicted_index]

        # Get predicted class score
        confidence = probabilities[
            predicted_index
        ].item() * 100

        # Prepare class probabilities for Gradio
        prediction_result = {

            CLASSES[i]: float(probabilities[i].item())

            for i in range(len(CLASSES))
        }

        # Result message
        if predicted_class == "TB":

            result_message = f"""
## Prediction Result: TB Detected

**Predicted Class:** {predicted_class}

**Model Score:** {confidence:.2f}%

The model detected image patterns associated with Tuberculosis.

This is an experimental AI prediction, not a medical diagnosis.
Please consult a qualified medical professional.
"""

        else:

            result_message = f"""
## Prediction Result: Normal

**Predicted Class:** {predicted_class}

**Model Score:** {confidence:.2f}%

The model classified this X-ray as Normal.

A Normal prediction does not rule out Tuberculosis.
Please consult a qualified medical professional if symptoms or concerns persist.
"""

        return prediction_result, result_message

    except Exception as e:

        return None, f"Prediction Error: {str(e)}"


# ==========================================
# 7. Model Information
# ==========================================

model_information = f"""
### CNN Model Performance

| Metric | Score |
|---|---:|
| Accuracy | {accuracy * 100:.2f}% |
| Precision | {precision * 100:.2f}% |
| Recall | {recall * 100:.2f}% |
| F1 Score | {f1_score * 100:.2f}% |

**Model:** Custom CNN (PyTorch)

**Input Image Size:** {IMG_SIZE} × {IMG_SIZE}

**Input Channels:** 1 (Grayscale)

**Preprocessing:** Median Filter + CLAHE

**Classes:** {", ".join(CLASSES)}
"""


# ==========================================
# 8. Gradio Interface
# ==========================================

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="TB Detection Using CNN"
) as demo:

    gr.Markdown(
        """
        # 🫁 Tuberculosis Detection System

        ### CNN-Based Chest X-ray Classification

        Upload a chest X-ray image to get an AI-generated
        prediction of Normal or Tuberculosis.
        """
    )

    with gr.Row():

        with gr.Column():

            image_input = gr.Image(
                type="pil",
                label="Upload Chest X-ray",
                height=350
            )

            with gr.Row():

                predict_button = gr.Button(
                    "Predict",
                    variant="primary"
                )

                clear_button = gr.ClearButton(
                    components=[],
                    value="Clear"
                )

        with gr.Column():

            prediction_output = gr.Label(
                label="Prediction Probabilities",
                num_top_classes=2
            )

            result_output = gr.Markdown(
                label="Prediction Result"
            )

    gr.Markdown("---")

    gr.Markdown(model_information)

    gr.Markdown(
        """
        ### Disclaimer

        This project is developed for academic and research purposes.

        It is not a clinically validated diagnostic system.
        Do not use its predictions as a substitute for professional
        medical examination or radiological assessment.
        """
    )

    # Prediction button
    predict_button.click(

        fn=predict_tb,

        inputs=image_input,

        outputs=[
            prediction_output,
            result_output
        ]
    )

    # Clear button
    clear_button = gr.ClearButton(
        components=[
            image_input,
            prediction_output,
            result_output
        ],
        value="Clear"
    )


# ==========================================
# 9. Launch Application
# ==========================================

if __name__ == "__main__":

    print("\nStarting Gradio Application...")

    demo.launch()
