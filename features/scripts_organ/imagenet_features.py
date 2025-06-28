# ==========================================
# Import Required Libraries
# ==========================================

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
import torchvision.models as models

# Ensure high precision for matrix multiplications in PyTorch
torch.set_float32_matmul_precision('high')


# ==========================================
# Model Initialization
# ==========================================
def initialize_model(model_name, num_classes, feature_extracting, input_size=(224, 224)):
    """
    Initialize a ConvNeXt model for feature extraction.

    Args:
        model_name (str): Name of the model.
        num_classes (int): Number of output classes.
        feature_extracting (bool): Whether to freeze feature extraction layers.
        input_size (tuple): Input size for the model.

    Returns:
        model (torch.nn.Module): Initialized model.
        CNN_family (str): Model family name.
    """
    model_ft = None

    if model_name == "convnext_tiny":
        model_ft = models.convnext_tiny(weights="ConvNeXt_Tiny_Weights.IMAGENET1K_V1")
        set_parameter_requires_grad(model_ft, feature_extracting)
        num_ftrs = model_ft.classifier[2].in_features
        model_ft.classifier[2] = nn.Linear(num_ftrs, num_classes)
        CNN_family = "ConvNeXt"
    else:
        print(f"❌ Model not found: {model_name}")
        sys.exit(1)

    return model_ft, CNN_family


def set_parameter_requires_grad(model, feature_extracting):
    """
    Freeze layers in the model if feature extraction is enabled.

    Args:
        model (torch.nn.Module): Model instance.
        feature_extracting (bool): Whether to freeze feature extraction layers.
    """
    classifier_name = None

    for attr in ["fc", "classifier", "head", "heads"]:
        if hasattr(model, attr):
            classifier_name = attr
            break

    if classifier_name is None:
        print("❌ No valid classifier layer found in the model.")
        sys.exit(1)

    if feature_extracting:
        for param in getattr(model, classifier_name).parameters():
            param.requires_grad = True


class Identity(nn.Module):
    """Identity layer for feature extraction."""
    def forward(self, x):
        return x


# ==========================================
# Dataset for Video Frames
# ==========================================
class VideoFrameDataset(Dataset):
    """Dataset wrapper for video frames."""
    
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index]


def load_frames_from_video(video_path, transform):
    """
    Load and preprocess frames from a video file.

    Args:
        video_path (str): Path to the video file.
        transform (torchvision.transforms): Transformations to apply.

    Returns:
        torch.Tensor: Stacked frames as a tensor.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(transform(img))

    cap.release()
    return torch.stack(frames)


# ==========================================
# Define Inference Model
# ==========================================
class ModelInference(pl.LightningModule):
    """Simplified model for inference."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


# ==========================================
# Command-line Arguments Parser
# ==========================================
def get_args_parser():
    """Define and parse command-line arguments."""
    
    parser = argparse.ArgumentParser(description="Extract Features from ConvNeXt", add_help=False)
    
    parser.add_argument('--name_model', type=str, default="convnext_tiny", help="Name of the model to use")
    parser.add_argument('--num_classes', type=int, default=1000, help="Number of output classes")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for data loading")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for DataLoader")

    # Dataset paths
    parser.add_argument('--path_data', type=str, default=os.path.join('..', '..', 'data', '15FPS'),
                        help="Path to source video files")
    parser.add_argument('--dataframe', type=str, default=os.path.join('..', '..', 'data', 'offcial_splits',
                                                                      'videoendoscopy-metadata.json'),
                        help="Path to JSON dataframe with video metadata")
    parser.add_argument('--output_dir', type=str, default=os.path.join("..", "15FPS", "ImageNet"),
                        help="Directory to save extracted features")

    return parser


# ==========================================
# Main Execution
# ==========================================
if __name__ == '__main__':
    # Parse command-line arguments
    parser = get_args_parser()
    args = parser.parse_args()

    # Validate paths
    for path, label in [(args.path_data, "data"), (args.dataframe, "dataframe"), (args.output_dir, "output")]:
        if not os.path.exists(path):
            if label == "output":
                os.makedirs(path, exist_ok=True)
                print(f"✅ Created output directory: {path}")
            else:
                print(f"❌ Error: {label} path '{path}' does not exist. Please verify.")
                sys.exit(1)
        else:
            print(f"✅ {label.capitalize()} path verified: {path}")

    # ==========================================
    # Device Handling
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ==========================================
    # Load Dataset
    # ==========================================
    df = pd.read_json(args.dataframe).reset_index(drop=True)

    # ==========================================
    # Load Model
    # ==========================================
    model_ft, CNN_family = initialize_model(args.name_model, args.num_classes, feature_extracting=True)
    model_ft = model_ft.to(device)

    for param in model_ft.parameters():
        param.requires_grad = False

    model = ModelInference(model_ft).to(device)
    model.eval()

    # ==========================================
    # Feature Extraction
    # ==========================================
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize([0.5990, 0.3664, 0.2769], [0.2847, 0.2190, 0.1772])
    ])

    for idx in tqdm(df.index):
        num_patient = str(int(df["num patient"].loc[idx]))    
        name_img = df["filename"].loc[idx]
        name_hash = os.path.splitext(name_img)[0]

        # Define save path
        path_save_feat = os.path.join(args.output_dir, num_patient, name_hash)
        os.makedirs(path_save_feat, exist_ok=True)

        video_path = os.path.join(args.path_data, num_patient, name_img)
        if not os.path.exists(video_path):
            print(f"❌ Error: Video '{video_path}' not found. Skipping...")
            continue

        frames_tensor = load_frames_from_video(video_path, transform)
        dataset = VideoFrameDataset(frames_tensor)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        frame_index = 0
        for batch in dataloader:
            batch = batch.to(device)

            with torch.no_grad():
                features = model(batch).cpu().numpy()

            for i, feature in enumerate(features):
                torch.save(torch.tensor(feature), os.path.join(path_save_feat, f'{frame_index:05d}.pt'))
                frame_index += 1

    print("🚀 Feature extraction complete!")
