
# ============================================================
# TUBERCULOSIS DETECTION USING CNN (PYTORCH)
# DIP BASED IMAGE PREPROCESSING
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import shutil
import random
import pickle
import copy

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image, UnidentifiedImageError

from torchvision import transforms
from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEED = 42

IMG_SIZE = 128

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 0.001

TEST_SIZE = 0.15

VALIDATION_SIZE = 0.15

NUM_WORKERS = 0

PATIENCE = 5

# DIP Parameters

MEDIAN_KERNEL = 3

CLAHE_CLIP_LIMIT = 2.0

CLAHE_TILE_SIZE = (8, 8)


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 4. DEVICE CONFIGURATION
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nUsing Device:", device)


# ============================================================
# 5. DATASET PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "Dataset of Tuberculosis Chest X-rays Images"
)

TB_PATH = os.path.join(
    DATASET_PATH,
    "TB Chest X-rays"
)

NORMAL_PATH = os.path.join(
    DATASET_PATH,
    "Normal Chest X-rays"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "Processed_TB_Dataset"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "TB_CNN_Model.pth"
)

PICKLE_PATH = os.path.join(
    BASE_DIR,
    "TB_CNN_Model.pkl"
)

CONFUSION_MATRIX_PATH = os.path.join(
    BASE_DIR,
    "TB_Confusion_Matrix.png"
)

HISTORY_PATH = os.path.join(
    BASE_DIR,
    "Training_History.png"
)

PREPROCESSING_PATH = os.path.join(
    BASE_DIR,
    "Preprocessing_Comparison.png"
)


# ============================================================
# 6. CHECK DATASET
# ============================================================

if not os.path.isdir(TB_PATH):

    raise FileNotFoundError(
        f"TB folder not found: {TB_PATH}"
    )

if not os.path.isdir(NORMAL_PATH):

    raise FileNotFoundError(
        f"Normal folder not found: {NORMAL_PATH}"
    )


VALID_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
)


# ============================================================
# 7. GET IMAGE FILES
# ============================================================

def get_image_files(folder):

    image_files = []

    for filename in sorted(os.listdir(folder)):

        file_path = os.path.join(
            folder,
            filename
        )

        if (
            os.path.isfile(file_path)
            and filename.lower().endswith(VALID_EXTENSIONS)
        ):

            image_files.append(filename)

    return image_files


tb_images = get_image_files(TB_PATH)

normal_images = get_image_files(NORMAL_PATH)


print("\n========== ORIGINAL DATASET ==========")

print("TB Images:", len(tb_images))

print("Normal Images:", len(normal_images))

print(
    "Total Images:",
    len(tb_images) + len(normal_images)
)


if len(tb_images) < 3 or len(normal_images) < 3:

    raise ValueError(
        "Each class must contain at least 3 images."
    )


# ============================================================
# 8. DIP PREPROCESSING FUNCTION
# ============================================================

def preprocess_xray(image_path):

    # Step 1: Read image

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise ValueError("Unable to read image")


    # Step 2: Median Filtering

    filtered = cv2.medianBlur(
        image,
        MEDIAN_KERNEL
    )


    # Step 3: CLAHE Contrast Enhancement

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_SIZE
    )

    enhanced = clahe.apply(filtered)


    # Step 4: Resize Image

    resized = cv2.resize(
        enhanced,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )


    return resized


# ============================================================
# 9. PREPROCESSING VISUALIZATION
# ============================================================

def save_preprocessing_comparison(image_path):

    original = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if original is None:
        return

    median = cv2.medianBlur(
        original,
        MEDIAN_KERNEL
    )

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_SIZE
    )

    enhanced = clahe.apply(median)

    resized = cv2.resize(
        enhanced,
        (IMG_SIZE, IMG_SIZE)
    )

    plt.figure(figsize=(14, 4))

    images = [
        original,
        median,
        enhanced,
        resized
    ]

    titles = [
        "Original X-ray",
        "Median Filtering",
        "CLAHE Enhancement",
        "Final 128x128"
    ]

    for i in range(4):

        plt.subplot(1, 4, i + 1)

        plt.imshow(
            images[i],
            cmap="gray"
        )

        plt.title(titles[i])

        plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        PREPROCESSING_PATH,
        dpi=300
    )

    plt.close()


# Save one sample preprocessing visualization

sample_image = (
    os.path.join(TB_PATH, tb_images[0])
)

save_preprocessing_comparison(sample_image)


# ============================================================
# 10. VALIDATE ORIGINAL IMAGES
# ============================================================

def validate_images(files, source_folder):

    valid_files = []

    for filename in files:

        image_path = os.path.join(
            source_folder,
            filename
        )

        try:

            with Image.open(image_path) as img:

                img.verify()

            valid_files.append(filename)

        except (
            OSError,
            ValueError,
            UnidentifiedImageError
        ):

            print(
                "Skipping corrupted image:",
                filename
            )

    return valid_files


tb_images = validate_images(
    tb_images,
    TB_PATH
)

normal_images = validate_images(
    normal_images,
    NORMAL_PATH
)


print("\n========== IMAGE VALIDATION ==========")

print("Valid TB Images:", len(tb_images))

print("Valid Normal Images:", len(normal_images))


if len(tb_images) < 3 or len(normal_images) < 3:

    raise ValueError(
        "Not enough valid images for splitting."
    )


# ============================================================
# 11. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# Split each class independently to preserve
# approximate class distribution.

tb_train, tb_temp = train_test_split(
    tb_images,
    test_size=TEST_SIZE + VALIDATION_SIZE,
    random_state=SEED
)

tb_val, tb_test = train_test_split(
    tb_temp,
    test_size=TEST_SIZE / (TEST_SIZE + VALIDATION_SIZE),
    random_state=SEED
)


normal_train, normal_temp = train_test_split(
    normal_images,
    test_size=TEST_SIZE + VALIDATION_SIZE,
    random_state=SEED
)

normal_val, normal_test = train_test_split(
    normal_temp,
    test_size=TEST_SIZE / (TEST_SIZE + VALIDATION_SIZE),
    random_state=SEED
)


# ============================================================
# 12. CREATE OUTPUT DIRECTORIES
# ============================================================

if os.path.exists(OUTPUT_PATH):

    shutil.rmtree(OUTPUT_PATH)


splits = [
    "Train",
    "Validation",
    "Test"
]

classes = [
    "Normal",
    "TB"
]

for split in splits:

    for category in classes:

        os.makedirs(
            os.path.join(
                OUTPUT_PATH,
                split,
                category
            ),
            exist_ok=True
        )


# ============================================================
# 13. APPLY DIP AND SAVE PROCESSED IMAGES
# ============================================================

def process_and_save(
    filenames,
    source_folder,
    destination_folder
):

    saved_count = 0

    for filename in filenames:

        source_path = os.path.join(
            source_folder,
            filename
        )

        try:

            processed_image = preprocess_xray(
                source_path
            )

            output_filename = (
                os.path.splitext(filename)[0] + ".png"
            )

            output_path = os.path.join(
                destination_folder,
                output_filename
            )

            success = cv2.imwrite(
                output_path,
                processed_image
            )

            if success:

                saved_count += 1

            else:

                print(
                    "Unable to save:",
                    filename
                )

        except Exception as error:

            print(
                "Preprocessing failed:",
                filename,
                error
            )

    return saved_count


# Train

tb_train_count = process_and_save(
    tb_train,
    TB_PATH,
    os.path.join(OUTPUT_PATH, "Train", "TB")
)

normal_train_count = process_and_save(
    normal_train,
    NORMAL_PATH,
    os.path.join(OUTPUT_PATH, "Train", "Normal")
)


# Validation

tb_val_count = process_and_save(
    tb_val,
    TB_PATH,
    os.path.join(OUTPUT_PATH, "Validation", "TB")
)

normal_val_count = process_and_save(
    normal_val,
    NORMAL_PATH,
    os.path.join(OUTPUT_PATH, "Validation", "Normal")
)


# Test

tb_test_count = process_and_save(
    tb_test,
    TB_PATH,
    os.path.join(OUTPUT_PATH, "Test", "TB")
)

normal_test_count = process_and_save(
    normal_test,
    NORMAL_PATH,
    os.path.join(OUTPUT_PATH, "Test", "Normal")
)


print("\n========== PREPROCESSING RESULTS ==========")

print("Train TB:", tb_train_count)

print("Train Normal:", normal_train_count)

print("Validation TB:", tb_val_count)

print("Validation Normal:", normal_val_count)

print("Test TB:", tb_test_count)

print("Test Normal:", normal_test_count)


# ============================================================
# 14. PYTORCH TRANSFORMATION
# ============================================================

train_transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=1),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )

])

eval_transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=1),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )

])


# ============================================================
# 15. LOAD DATASET
# ============================================================

train_dataset = ImageFolder(
    os.path.join(OUTPUT_PATH, "Train"),
    transform=train_transform
)

val_dataset = ImageFolder(
    os.path.join(OUTPUT_PATH, "Validation"),
    transform=eval_transform
)

test_dataset = ImageFolder(
    os.path.join(OUTPUT_PATH, "Test"),
    transform=eval_transform
)


print("\n========== DATASET INFORMATION ==========")

print("Classes:", train_dataset.classes)

print("Class Mapping:", train_dataset.class_to_idx)

print("Training Images:", len(train_dataset))

print("Validation Images:", len(val_dataset))

print("Testing Images:", len(test_dataset))


if len(val_dataset) == 0 or len(test_dataset) == 0:

    raise ValueError(
        "Validation or Test dataset is empty."
    )


# ============================================================
# 16. CLASS IMBALANCE HANDLING
# ============================================================

train_labels = [
    label for _, label in train_dataset.samples
]

class_counts = np.bincount(
    train_labels,
    minlength=len(train_dataset.classes)
)

if np.any(class_counts == 0):

    raise ValueError(
        "One class is missing from training data."
    )

class_weights = (
    len(train_labels) /
    (len(train_dataset.classes) * class_counts)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)


print("\nClass Counts:", class_counts)

print("Class Weights:", class_weights)


# ============================================================
# 17. DATALOADERS
# ============================================================

pin = device.type == "cuda"

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=pin
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin
)


# ============================================================
# 18. CNN ARCHITECTURE
# ============================================================

class MyCNN(nn.Module):

    def __init__(self, num_classes):

        super(MyCNN, self).__init__()

        # Feature Extraction

        self.features = nn.Sequential(

            # Block 1

            nn.Conv2d(
                1, 32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm2d(32),

            nn.MaxPool2d(2),


            # Block 2

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm2d(64),

            nn.MaxPool2d(2),


            # Block 3

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm2d(128),

            nn.MaxPool2d(2)

        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 16 * 16,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.4),

            nn.Linear(
                64,
                num_classes
            )

        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# 19. INITIALIZE MODEL
# ============================================================

num_classes = len(train_dataset.classes)

model = MyCNN(
    num_classes=num_classes
).to(device)


print("\n========== CNN MODEL ==========")

print(model)


# ============================================================
# 20. LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# 21. TRAINING HISTORY
# ============================================================

history = {

    "train_loss": [],

    "train_accuracy": [],

    "val_loss": [],

    "val_accuracy": []

}


best_val_loss = float("inf")

best_model_weights = copy.deepcopy(
    model.state_dict()
)

early_stopping_counter = 0


# ============================================================
# 22. MODEL TRAINING
# ============================================================

print("\n========== MODEL TRAINING ==========")


for epoch in range(EPOCHS):

    # -------------------------
    # Training
    # -------------------------

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


        # Forward Pass

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(dim=1)

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()


    train_loss = running_loss / total

    train_accuracy = correct / total


    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    val_running_loss = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            val_total += labels.size(0)

            val_correct += (
                predictions == labels
            ).sum().item()


    val_loss = val_running_loss / val_total

    val_accuracy = val_correct / val_total


    # -------------------------
    # Save History
    # -------------------------

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_accuracy
    )


    scheduler.step(val_loss)


    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {val_accuracy:.4f}"
    )


    # -------------------------
    # Best Model Saving
    # -------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        early_stopping_counter = 0

        print("Best model updated!")

    else:

        early_stopping_counter += 1

        print(
            "Early Stopping Counter:",
            early_stopping_counter
        )


    # -------------------------
    # Early Stopping
    # -------------------------

    if early_stopping_counter >= PATIENCE:

        print("\nEarly stopping activated!")

        break


# ============================================================
# 23. LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    best_model_weights
)

model.eval()

print("\nBest validation loss:", best_val_loss)


# ============================================================
# 24. FINAL TEST EVALUATION
# ============================================================

print("\n========== FINAL TEST EVALUATION ==========")

y_true = []

y_pred = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        y_true.extend(
            labels.numpy().tolist()
        )

        y_pred.extend(
            predictions.cpu().numpy().tolist()
        )


# ============================================================
# 25. PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0
)


print("\n========== PERFORMANCE RESULTS ==========")

print(f"Accuracy:  {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall:    {recall:.4f}")

print(f"F1 Score:  {f1:.4f}")


# ============================================================
# 26. CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(

    classification_report(

        y_true,

        y_pred,

        labels=[0, 1],

        target_names=[
            "Normal",
            "TB"
        ],

        zero_division=0

    )

)


# ============================================================
# 27. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)


plt.figure(figsize=(7, 6))

plt.imshow(
    cm,
    interpolation="nearest",
    cmap="Blues"
)

plt.title("TB Detection Confusion Matrix")

plt.colorbar()

plt.xticks(
    [0, 1],
    ["Normal", "TB"]
)

plt.yticks(
    [0, 1],
    ["Normal", "TB"]
)

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="black",
            fontsize=14
        )


plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300
)

plt.close()


# ============================================================
# 28. TRAINING HISTORY VISUALIZATION
# ============================================================

epochs_range = range(
    1,
    len(history["train_loss"]) + 1
)


plt.figure(figsize=(12, 5))


# Loss

plt.subplot(1, 2, 1)

plt.plot(
    epochs_range,
    history["train_loss"],
    label="Training Loss"
)

plt.plot(
    epochs_range,
    history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid()


# Accuracy

plt.subplot(1, 2, 2)

plt.plot(
    epochs_range,
    history["train_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid()


plt.tight_layout()

plt.savefig(
    HISTORY_PATH,
    dpi=300
)

plt.close()


# ============================================================
# 29. SAVE PYTORCH MODEL
# ============================================================

checkpoint = {

    "model_state_dict": model.cpu().state_dict(),

    "classes": train_dataset.classes,

    "class_to_idx": train_dataset.class_to_idx,

    "img_size": IMG_SIZE,

    "input_channels": 1,

    "normalization_mean": [0.5],

    "normalization_std": [0.5],

    "preprocessing": {

        "grayscale": True,

        "median_kernel": MEDIAN_KERNEL,

        "clahe_clip_limit": CLAHE_CLIP_LIMIT,

        "clahe_tile_size": CLAHE_TILE_SIZE,

        "resize": IMG_SIZE

    },

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1_score": f1,

    "best_validation_loss": best_val_loss

}


torch.save(
    checkpoint,
    MODEL_PATH
)


# ============================================================
# 30. SAVE PICKLE MODEL INFORMATION
# ============================================================

with open(
    PICKLE_PATH,
    "wb"
) as file:

    pickle.dump(
        checkpoint,
        file,
        protocol=pickle.HIGHEST_PROTOCOL
    )


# ============================================================
# 31. FINAL PROJECT SUMMARY
# ============================================================

print("\n==========================================")

print("       TB DETECTION PROJECT COMPLETED")

print("==========================================")

print("\nPreprocessing:")

print("1. Grayscale Conversion")

print("2. Median Filtering")

print("3. CLAHE Enhancement")

print("4. Image Resizing")

print("5. Normalization")


print("\nDataset:")

print("Train:", len(train_dataset))

print("Validation:", len(val_dataset))

print("Test:", len(test_dataset))


print("\nModel Performance:")

print(f"Accuracy:  {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall:    {recall:.4f}")

print(f"F1 Score:  {f1:.4f}")


print("\nGenerated Files:")

print("1.", MODEL_PATH)

print("2.", PICKLE_PATH)

print("3.", CONFUSION_MATRIX_PATH)

print("4.", HISTORY_PATH)

print("5.", PREPROCESSING_PATH)

print("\n==========================================")
