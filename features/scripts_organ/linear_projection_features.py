# ==========================================
# Import Required Libraries
# ==========================================

import argparse
import sys
import os
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch import nn

# Ensure high precision for matrix multiplications in PyTorch
torch.set_float32_matmul_precision('high')


# ==========================================
# Direct Patch Embedding (No Projection)
# ==========================================
class PatchEmbedDirect(nn.Module):
    """Embed image patches without a projection layer."""
    
    def __init__(self, embed_dim=768):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)  # Normalize embeddings

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten: [batch, 3, 16, 16] → [batch, 768]
        return self.norm(x)  # Apply LayerNorm


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
    """Load and preprocess frames from a video file."""
    
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convert to PIL
        frames.append(transform(img))
    
    cap.release()
    return torch.stack(frames)  # Return a tensor of frames


# ==========================================
# Command-line Arguments Parser
# ==========================================
def get_args_parser():
    """Define and parse command-line arguments."""
    
    parser = argparse.ArgumentParser(description="Extract Linear Projection Features", add_help=False)
    
    # Image parameters
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for DataLoader')

    # Dataset paths
    parser.add_argument('--path_data', type=str, default=os.path.join('..', '..', 'data', '15FPS'),
                        help='Path to source video files')
    parser.add_argument('--dataframe', type=str, default=os.path.join('..', '..', 'data', 'offcial_splits','videoendoscopy-metadata.json'),
                        help='Path to JSON dataframe with video metadata')
    parser.add_argument('--output_dir', type=str, default=os.path.join("..", "15FPS", "LinearProjection"),
                        help='Directory to save extracted features')

    return parser


# ==========================================
# Main Execution
# ==========================================
if __name__ == '__main__':
    # Parse command-line arguments
    parser = get_args_parser()
    args = parser.parse_args()

    # Validate dataset directory
    if not os.path.exists(args.path_data):
        print(f"❌ Error: Data path '{args.path_data}' does not exist. Please verify.")
        sys.exit(1)
    print(f"✅ Data path verified: '{args.path_data}'")

    # Validate dataframe file existence
    if not os.path.exists(args.dataframe):
        print(f"❌ Error: Dataframe file '{args.dataframe}' is missing. Please verify.")
        sys.exit(1)
    print(f"✅ Dataframe file verified: '{args.dataframe}'")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"✅ Output directory is set: '{args.output_dir}'")

    # ==========================================
    # Device Setup
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ==========================================
    # Load Dataset
    # ==========================================
    df = pd.read_json(args.dataframe).reset_index(drop=True)

    # ==========================================
    # Load Model (No Projection Layer)
    # ==========================================
    model = PatchEmbedDirect(embed_dim=768).to(device)

    # ==========================================
    # Define Image Transformations
    # ==========================================
    transform = transforms.Compose([
        transforms.Resize((16, 16)),  # Resize image to 16×16 patches
        transforms.ToTensor(),
        transforms.Normalize([0.5990, 0.3664, 0.2769], [0.2847, 0.2190, 0.1772])
    ])

    # ==========================================
    # Feature Extraction Process
    # ==========================================
    for idx in tqdm(df.index):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        num_patient = str(int(df["num patient"].loc[idx]))    
        name_img = df["filename"].loc[idx]
        name_hash = os.path.splitext(name_img)[0]

        # Define save path for extracted features
        path_save_feat = os.path.join(args.output_dir, num_patient, name_hash)
        os.makedirs(path_save_feat, exist_ok=True)

        video_path = os.path.join(args.path_data, num_patient, name_img)
        if not os.path.exists(video_path):
            print(f"❌ Error: Video '{video_path}' does not exist. Skipping...")
            continue

        # Skip if features already exist
        if df["num frames"].loc[idx] == len(os.listdir(path_save_feat)):
            print(f"⚠️ Skipped: Features already exist for '{name_img}'")
            continue

        frames_tensor = load_frames_from_video(video_path, transform)
        dataset = VideoFrameDataset(frames_tensor)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        frame_index = 0

        for batch in dataloader:
            batch = batch.to(device)

            # Extract features
            with torch.no_grad():
                features = model(batch)  # Extract embeddings
                features = features.cpu().numpy()

            # Save each frame's feature
            for feature in features:
                save_path = os.path.join(path_save_feat, f'{frame_index:05d}.pt')
                torch.save(torch.tensor(feature), save_path)
                frame_index += 1

    print("🚀 Feature extraction complete!")
