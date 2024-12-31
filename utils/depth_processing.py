import torch
import sys
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os  # For path checking

sys.path.append('/home/oshkosh/Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

def load_depth_model(encoder='vits'):
    model = DepthAnythingV2(**model_configs[encoder])
    checkpoint_path = f'/home/oshkosh/Third-Eye/models/depth-anything/depth_anything_v2_vits.pth'
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.to(device).eval()

def estimate_depth(model, frame):
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize and normalize frame
    resized_frame = cv2.resize(frame_rgb, (640, 480))
    tensor_frame = torch.from_numpy(resized_frame).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    
    # Infer depth map
    with torch.no_grad():
        depth_map = model.infer_image(tensor_frame.squeeze(0).permute(1, 2, 0).cpu().numpy())
    
    # Normalize depth map for visualization (Optional)
    normalized_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    return normalized_depth_map
