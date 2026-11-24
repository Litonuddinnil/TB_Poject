
 

# ==============================
# 1. Import Libraries
# ==============================

import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt

import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==============================
# 2. Configuration
# ==============================

SEED = 42

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

TEST_SIZE = 0.20

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==============================
# 3. Device Configuration
# ==============================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nUsing Device:", device)


# ==============================
# 4. Dataset Paths
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "Dataset of Tuberculosis Chest X-rays Images"
)

DISEASE_PATH = os.path.join(
    DATASET_PATH,
    "TB Chest X-rays"
)

NORMAL_PATH = os.path.join(
    DATASET_PATH,
    "Normal Chest X-rays"
)

OUTPUT_PATH = os.path.join(
    DATASET_PATH,
    "Filtered_TB_Dataset"
)

TRAIN_PATH = os.path.join(
    OUTPUT_PATH,
    "Train"
)

TEST_PATH = os.path.join(
    OUTPUT_PATH,
    "Test"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "TB_CNN_Model.pth"
)


# ==============================
# 5. Check Dataset
# ==============================

if not os.path.isdir(DISEASE_PATH):
    raise FileNotFoundError(
        f"TB dataset folder not found: {DISEASE_PATH}"
    )

if not os.path.isdir(NORMAL_PATH):
    raise FileNotFoundError(
        f"Normal dataset folder not found: {NORMAL_PATH}"
    )


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def get_image_files(folder):

    files = []

    for filename in sorted(os.listdir(folder)):

        file_path = os.path.join(folder, filename)

        if (
            os.path.isfile(file_path)
            and filename.lower().endswith(VALID_EXTENSIONS)
        ):

            files.append(filename)

    return files


tb_images = get_image_files(DISEASE_PATH)

normal_images = get_image_files(NORMAL_PATH)


print("\n========== DATASET INFORMATION ==========")

print("TB Images:", len(tb_images))
print("Normal Images:", len(normal_images))

print("Total Images:", len(tb_images) + len(normal_images))


if len(tb_images) < 2 or len(normal_images) < 2:
    raise ValueError(
        "Both TB and Normal folders must contain at least 2 valid images."
    )


# ==============================
# 6. Train-Test Split
# ==============================

# Remove previous generated split
# This prevents duplicate images after rerunning.

if os.path.exists(OUTPUT_PATH):

    shutil.rmtree(OUTPUT_PATH)


# Create fresh folders

for split in ["Train", "Test"]:

    for category in ["TB", "Normal"]:

        os.makedirs(
            os.path.join(OUTPUT_PATH, split, category),
            exist_ok=True
        )


# Split TB images

tb_train, tb_test = train_test_split(
    tb_images,
    test_size=TEST_SIZE,
    random_state=SEED
)


# Split Normal images

normal_train, normal_test = train_test_split(
    normal_images,
    test_size=TEST_SIZE,
    random_state=SEED
)


# ==============================
# 7. Copy Dataset Images
# ==============================

def copy_images(files, source, destination):

    for filename in files:

        source_file = os.path.join(source, filename)

        destination_file = os.path.join(
            destination,
            filename
        )

        shutil.copy2(
            source_file,
            destination_file
        )


# Copy TB images

copy_images(
    tb_train,
    DISEASE_PATH,
    os.path.join(TRAIN_PATH, "TB")
)

copy_images(
    tb_test,
    DISEASE_PATH,
    os.path.join(TEST_PATH, "TB")
)


# Copy Normal images

copy_images(
    normal_train,
    NORMAL_PATH,
    os.path.join(TRAIN_PATH, "Normal")
)

copy_images(
    normal_test,
    NORMAL_PATH,
    os.path.join(TEST_PATH, "Normal")
)


print("\n========== DATASET SPLITTING ==========")

print("TB Train:", len(tb_train))
print("TB Test:", len(tb_test))

print("Normal Train:", len(normal_train))
print("Normal Test:", len(normal_test))

print("\nDataset splitting completed successfully!")


# ==============================
# 8. Image Transformation
# ==============================

transform = transforms.Compose([

    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

])


# ==============================
# 9. Custom Dataset Class
# ==============================

class TBCNN_Dataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.samples = []

        self.transform = transform

        # Fixed class order:
        # Normal = 0
        # TB = 1

        self.classes = ["Normal", "TB"]

        self.class_to_idx = {
            "Normal": 0,
            "TB": 1
        }

        for class_name in self.classes:

            class_path = os.path.join(
                root_dir,
                class_name
            )

            if not os.path.isdir(class_path):

                raise FileNotFoundError(
                    f"Class folder not found: {class_path}"
                )

            for img_name in sorted(os.listdir(class_path)):

                if not img_name.lower().endswith(
                    VALID_EXTENSIONS
                ):
                    continue

                img_path = os.path.join(
                    class_path,
                    img_name
                )

                if not os.path.isfile(img_path):
                    continue

                # Skip corrupted images

                try:

                    with Image.open(img_path) as img:
                        img.verify()

                    label = self.class_to_idx[class_name]

                    self.samples.append(
                        (img_path, label)
                    )

                except (
                    OSError,
                    ValueError,
                    UnidentifiedImageError
                ):

                    print("Skipping corrupted image:", img_path)

        if len(self.samples) == 0:

            raise ValueError(
                f"No valid images found in {root_dir}"
            )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        img_path, label = self.samples[idx]

        with Image.open(img_path) as img:

            image = img.convert("RGB")

        if self.transform:

            image = self.transform(image)

        return image, label


# ==============================
# 10. Load Train-Test Dataset
# ==============================

train_dataset = TBCNN_Dataset(
    TRAIN_PATH,
    transform=transform
)

test_dataset = TBCNN_Dataset(
    TEST_PATH,
    transform=transform
)


num_classes = len(train_dataset.classes)

print("\n========== LOADED DATASET ==========")

print("Classes:", train_dataset.classes)

print("Number of Classes:", num_classes)

print("Training Images:", len(train_dataset))

print("Testing Images:", len(test_dataset))


# ==============================
# 11. Class Imbalance Handling
# ==============================

train_labels = [
    label for _, label in train_dataset.samples
]

class_counts = np.bincount(
    train_labels,
    minlength=num_classes
)

class_weights = (
    len(train_labels) /
    (num_classes * class_counts)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)

print("\nClass Weights:", class_weights)


# ==============================
# 12. DataLoader
# ==============================

pin = device.type == "cuda"

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=pin
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=pin
)


# ==============================
# 13. CNN Architecture
# ==============================

class MyCNN(nn.Module):

    def __init__(self, num_classes):

        super(MyCNN, self).__init__()

        # Feature Extraction

        self.features = nn.Sequential(

            # Convolution Block 1

            nn.Conv2d(
                3, 32,
                kernel_size=3,
                padding="same"
            ),

            nn.ReLU(),

            nn.BatchNorm2d(32),

            nn.MaxPool2d(2),


            # Convolution Block 2

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding="same"
            ),

            nn.ReLU(),

            nn.BatchNorm2d(64),

            nn.MaxPool2d(2),


            # Convolution Block 3

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding="same"
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


# ==============================
# 14. Initialize Model
# ==============================

model = MyCNN(
    num_classes=num_classes
).to(device)

print("\n========== CNN MODEL ==========")

print(model)


# ==============================
# 15. Loss and Optimizer
# ==============================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==============================
# 16. Model Training
# ==============================

print("\n========== MODEL TRAINING ==========")


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    correct = 0

    total = 0


    for batch_features, batch_labels in train_loader:

        batch_features = batch_features.to(device)

        batch_labels = batch_labels.to(device)


        # Forward Pass

        outputs = model(batch_features)

        loss = criterion(
            outputs,
            batch_labels
        )


        # Backpropagation

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        # Calculate Loss

        total_loss += loss.item()


        # Training Accuracy

        _, predicted = torch.max(outputs, 1)

        total += batch_labels.size(0)

        correct += (
            predicted == batch_labels
        ).sum().item()


    avg_loss = total_loss / len(train_loader)

    train_accuracy = correct / total


    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {avg_loss:.4f} "
        f"Accuracy: {train_accuracy:.4f}"
    )


# ==============================
# 17. Model Evaluation
# ==============================

print("\n========== MODEL EVALUATION ==========")

model.eval()

y_true = []

y_pred = []


with torch.no_grad():

    for batch_features, batch_labels in test_loader:

        batch_features = batch_features.to(device)

        outputs = model(batch_features)

        _, predicted = torch.max(outputs, 1)

        y_true.extend(
            batch_labels.numpy().tolist()
        )

        y_pred.extend(
            predicted.cpu().numpy().tolist()
        )


# ==============================
# 18. Performance Metrics
# ==============================

accuracy = accuracy_score(
    y_true,
    y_pred
)

# Positive class = TB (label 1)

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


# ==============================
# 19. Classification Report
# ==============================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "TB"],
        zero_division=0
    )
)


# ==============================
# 20. Confusion Matrix
# ==============================

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
    os.path.join(BASE_DIR, "TB_Confusion_Matrix.png"),
    dpi=300
)

plt.show()


# ==============================
# 21. Save Model
# ==============================

torch.save(

    {
        "model_state_dict": model.state_dict(),

        "classes": train_dataset.classes,

        "img_size": IMG_SIZE,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1_score": f1

    },

    MODEL_PATH

)


print("\n========== MODEL SAVING ==========")

print("Model saved successfully!")

print("Model Path:", MODEL_PATH)

print("Confusion Matrix saved successfully!")

print("\n========== PROJECT COMPLETED ==========")
 
# ==============================
# 22. Save Model as Pickle
# ==============================
PICKLE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "TB_CNN_Model.pkl"
)

model.eval()

# Save model information
model_data = {
    "model_state_dict": model.cpu().state_dict(),
    "classes": train_dataset.classes,
    "img_size": IMG_SIZE,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}

# Generate Pickle File
with open(PICKLE_MODEL_PATH, "wb") as file:
    pickle.dump(model_data, file, protocol=pickle.HIGHEST_PROTOCOL)

print("\n========== MODEL SAVING ==========")
print("Pickle model saved successfully!")
print("Model Path:", PICKLE_MODEL_PATH)

print("\n========== PROJECT COMPLETED ==========")