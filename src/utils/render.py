#!/usr/bin/env python3
"""
Rendering module for ReactDiff project.
Handles the conversion of 3DMM parameters to rendered videos using PIRender.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import sys
from pathlib import Path
from typing import Optional, Union, Tuple, List
import logging
from .utils import torch_img_to_np, _fix_image, torch_img_to_np2

# Add external directory to path for PIRender imports
current_dir = Path(__file__).parent
external_dir = current_dir.parent / "external"
pirender_dir = external_dir / "PIRender"
sys.path.insert(0, str(external_dir))
sys.path.insert(0, str(pirender_dir))

try:
    from PIRender.face_model import FaceGenerator
    # these utils are your project’s utils, not PIRender’s
    PI_RENDER_AVAILABLE = True
except Exception as e:
    import traceback
    PI_RENDER_AVAILABLE = False
    logging.warning("PIRender not available. Using placeholder rendering.")
    logging.warning(f"PIRender import error: {e}\n{traceback.format_exc()}")


def obtain_seq_index(index, num_frames, semantic_radius=13):
    """Obtain sequence indices for semantic transformation."""
    seq = list(range(index - semantic_radius, index + semantic_radius + 1))
    seq = [min(max(item, 0), num_frames - 1) for item in seq]
    return seq

#
# def transform_semantic(semantic):
#     """Transform semantic parameters for PIRender."""
#     semantic_list = []
#     for i in range(semantic.shape[0]):
#         index = obtain_seq_index(i, semantic.shape[0])
#         semantic_item = semantic[index, :].unsqueeze(0)
#         semantic_list.append(semantic_item)
#     semantic = torch.cat(semantic_list, dim=0)
#     return semantic.transpose(1, 2)

def transform_semantic(semantic: torch.Tensor) -> torch.Tensor:
    # Accept (T,58) or (1,T,58)
    if semantic.dim() == 3 and semantic.shape[0] == 1:
        semantic = semantic.squeeze(0)
    assert semantic.dim() == 2 and semantic.shape[1] == 58, \
        f"transform_semantic expects (T,58), got {tuple(semantic.shape)}"
    # ... then your original implementation
    semantic_list = []
    for i in range(semantic.shape[0]):
        index = obtain_seq_index(i, semantic.shape[0])
        semantic_item = semantic[index, :].unsqueeze(0)  # (1, 27, 58)
        semantic_list.append(semantic_item)
    semantic = torch.cat(semantic_list, dim=0)           # (T, 27, 58)
    return semantic.transpose(1, 2).contiguous()         # (T, 58, 27)


class Render(nn.Module):
    """
    Renderer for converting 3DMM parameters to video frames using PIRender.
    
    This class handles the conversion of 3DMM (3D Morphable Model) parameters
    to rendered video frames, with support for both PIRender integration and
    fallback placeholder rendering.
    """
    
    def __init__(self, device: str = 'cpu', use_pirender: bool = True):
        """
        Initialize the renderer.
        
        Args:
            device: Device to run rendering on ('cpu' or 'cuda')
            use_pirender: Whether to use PIRender for actual 3D rendering
        """
        super().__init__()
        self.device = device
        self.use_pirender = use_pirender and PI_RENDER_AVAILABLE
        
        # Load reference face parameters
        self.reference_face = self._load_reference_face()
        
        # Initialize PIRender if available
        if self.use_pirender:
            self._init_pirender()
        else:
            logging.warning("Using placeholder rendering. Install PIRender for actual 3D face rendering.")
    
    def _load_reference_face(self) -> np.ndarray:
        """Load reference face parameters from file."""
        reference_path = external_dir / "reference_full.npy"
        
        try:
            if reference_path.exists():
                reference_face = np.load(reference_path)
                print(f"Loaded reference face from {reference_path}")
                return reference_face
            else:
                logging.warning(f"Reference face file not found at {reference_path}")
        except Exception as e:
            logging.error(f"Error loading reference face: {e}")
        
        # Return default reference face (neutral expression)
        print("Using default reference face parameters")
        return np.zeros((1, 58), dtype=np.float32)
    
    def _init_pirender(self):
        """Initialize PIRender face generator."""
        try:
            self.face_generator = FaceGenerator().to(self.device)
            self.face_generator.eval()
            
            # Load pre-trained weights
            checkpoint_path = external_dir / "PIRender" / "cur_model_fold.pth"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.face_generator.load_state_dict(checkpoint['state_dict'])
                print("PIRender checkpoint loaded successfully")
            else:
                logging.warning(f"PIRender checkpoint not found at {checkpoint_path}")
            
            # Load mean and std face for 3DMM transformation
            mean_face_path = external_dir / "FaceVerse" / "mean_face.npy"
            std_face_path = external_dir / "FaceVerse" / "std_face.npy"
            
            if mean_face_path.exists() and std_face_path.exists():
                self.mean_face = torch.FloatTensor(
                    np.load(mean_face_path).astype(np.float32)).view(1, 1, -1).to(self.device)
                self.std_face = torch.FloatTensor(
                    np.load(std_face_path).astype(np.float32)).view(1, 1, -1).to(self.device)

                self._reverse_transform_3dmm = lambda e: (
                    (e * self.std_face + self.mean_face).squeeze(0)  # std/mean are (1,1,C)
                    if e.dim() == 3 else (e * self.std_face + self.mean_face)
                )
                # Create reverse transform for 3DMM parameters
                print("FaceVerse mean/std loaded successfully")
            else:
                logging.warning("FaceVerse mean/std files not found, using identity transform")
                self._reverse_transform_3dmm = lambda e: e
                
            print("PIRender initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize PIRender: {e}")
            self.use_pirender = False
    
    def rendering(self, 
                  output_dir: str, 
                  video_name: str, 
                  listener_3dmm: torch.Tensor, 
                  listener_reference: torch.Tensor,
                  fps: int = 25,
                  save_frames: bool = False,
                  chunk_size: int = 32) -> str:
        """
        Render listener 3DMM parameters to video frames.
        
        Args:
            output_dir: Directory to save rendered frames and video
            video_name: Name for the output video (without extension)
            listener_3dmm: 3DMM parameters for listener (T, 58)
            listener_reference: Reference frame for listener (3, H, W) or (H, W, 3)
            fps: Frames per second for output video
            save_frames: Whether to save individual frames
            chunk_size: Number of frames to process at once (to avoid OOM)
            
        Returns:
            Path to the generated video file
            
        Raises:
            ValueError: If input tensors have invalid shapes
            RuntimeError: If rendering fails
        """
        try:
            # Validate inputs
            self._validate_inputs(listener_3dmm, listener_reference)
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Convert tensors to appropriate format
            listener_3dmm_np = self._prepare_3dmm_data(listener_3dmm)
            listener_reference_np = self._prepare_reference_data(listener_reference)
            
            # Render frames
            if self.use_pirender:
                frames = self._render_with_pirender(listener_3dmm_np, listener_reference_np, chunk_size)
            else:
                frames = self._render_placeholder(listener_3dmm_np, listener_reference_np)
            
            # Save frames if requested
            frame_dir = None
            if save_frames:
                frame_dir = output_path / f"{video_name}_frames"
                frame_dir.mkdir(exist_ok=True)
                for i, frame in enumerate(frames):
                    frame_path = frame_dir / f"frame_{i:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
            
            # Create video from frames
            video_path = output_path / f"{video_name}.mp4"
            self._create_video_from_frames(frames, str(video_path), fps)
            
            print(f"Video saved to: {video_path}")
            return str(video_path)
            
        except Exception as e:
            logging.error(f"Rendering failed: {e}")
            raise RuntimeError(f"Rendering failed: {e}")
    
    def _validate_inputs(self, listener_3dmm: torch.Tensor, listener_reference: torch.Tensor):
        """Validate input tensor shapes and types."""
        if not isinstance(listener_3dmm, torch.Tensor):
            raise ValueError("listener_3dmm must be a torch.Tensor")
        if not isinstance(listener_reference, torch.Tensor):
            raise ValueError("listener_reference must be a torch.Tensor")
        
        if listener_3dmm.dim() != 2 or listener_3dmm.shape[1] != 58:
            raise ValueError(f"listener_3dmm must have shape (T, 58), got {listener_3dmm.shape}")
        
        if listener_reference.dim() not in [3, 4]:
            raise ValueError(f"listener_reference must have 3 or 4 dimensions, got {listener_reference.dim()}")
    
    def _prepare_3dmm_data(self, listener_3dmm: torch.Tensor) -> np.ndarray:
        """Convert 3DMM tensor to numpy array."""
        if listener_3dmm.is_cuda:
            listener_3dmm = listener_3dmm.detach().cpu()
        return listener_3dmm.numpy()
    
    def _prepare_reference_data(self, listener_reference: torch.Tensor) -> np.ndarray:
        """Convert reference tensor to numpy array in HWC format."""
        if listener_reference.is_cuda:
            listener_reference = listener_reference.detach().cpu()
        
        reference_np = listener_reference.numpy()
        
        # Handle different input formats
        if reference_np.ndim == 4:  # (B, C, H, W) or (B, H, W, C)
            reference_np = reference_np[0]  # Take first batch
        
        if reference_np.shape[0] == 3:  # CHW format
            reference_np = np.transpose(reference_np, (1, 2, 0))
        
        # Denormalize if needed (assuming normalization to [-1, 1])
        if reference_np.min() < 0:
            reference_np = (reference_np + 1) * 127.5
        
        reference_np = np.clip(reference_np, 0, 255).astype(np.uint8)
        
        # Convert BGR to RGB if needed
        if reference_np.shape[2] == 3:
            reference_np = cv2.cvtColor(reference_np, cv2.COLOR_RGB2BGR)
        
        return reference_np
    
    def _render_with_pirender(self, listener_3dmm: np.ndarray, listener_reference: np.ndarray,
                              chunk_size: int = 32) -> List[np.ndarray]:
        """
        listener_3dmm: (T, 58)  numpy
        listener_reference: (H, W, 3) BGR uint8
        """
        with torch.no_grad():
            T = int(listener_3dmm.shape[0])

            # ---- 3DMM -> tensor (T,58) ----
            a = torch.from_numpy(listener_3dmm).float().to(self.device)   # (T,58)
            a = self._reverse_transform_3dmm(a)
            if a.dim() == 3:
                # fix the accidental (1,T,58)
                a = a.squeeze(0)
            assert a.shape == (T, 58), f"3DMM shape must be (T,58), got {tuple(a.shape)}"

            # ---- PIRender semantics (T,58,27) ----
            semantics = transform_semantic(a).to(self.device).contiguous()
            assert semantics.dim() == 3 and semantics.shape[0] == T, \
                f"semantics must be (T,58,27), got {tuple(semantics.shape)}"

            # ---- reference HWC(BGR uint8) -> NCHW float32 [-1,1] ----
            ref_rgb = cv2.cvtColor(listener_reference, cv2.COLOR_BGR2RGB)           # (H,W,3)
            ref_chw = torch.from_numpy(ref_rgb).permute(2, 0, 1).contiguous()       # (3,H,W)
            ref_chw = (ref_chw.float() / 255.0) * 2.0 - 1.0                         # [-1,1]
            ref_chw = ref_chw.unsqueeze(0).repeat(T, 1, 1, 1).to(self.device)       # (T,3,H,W)

            # ---- chunked forward (10 chunks like the original) ----
            out_chunks: List[np.ndarray] = []
            duration = max(1, T // 10)
            for i in range(10):
                start = i * duration
                end   = T if i == 9 else (i + 1) * duration
                if start >= T:
                    break

                ref_i = ref_chw[start:end].contiguous()
                sem_i = semantics[start:end].contiguous()  # (N,58,27)

                try:
                    out = self.face_generator(ref_i, sem_i)         # expects (N,3,H,W) & (N,58,27)
                    fake = out["fake_image"]                        # (N,3,H,W), [-1,1] or [0,1]
                    fake = fake.clamp(-1, 1) * 0.5 + 0.5            # -> [0,1]
                    fake_np = (fake * 255.0).byte().cpu().permute(0, 2, 3, 1).numpy()  # (N,H,W,3) RGB
                    out_chunks.append(fake_np)
                except Exception as err:
                    logging.warning(f"PIRender failed for chunk {i}, using reference: {err}")
                    n = max(0, end - start)
                    if n > 0:
                        out_chunks.append(np.tile(ref_rgb[None, ...], (n, 1, 1, 1)))   # (N,H,W,3) RGB

            if not out_chunks:
                raise RuntimeError("No chunks produced in PIRender rendering.")

            vid_rgb = np.concatenate(out_chunks, axis=0)[:T]   # (T,H,W,3) RGB
            frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in vid_rgb]
            print(f"Generated {len(frames)} frames for rendering")
            return frames



    
    
    def _render_placeholder(self, 
                           listener_3dmm: np.ndarray, 
                           listener_reference: np.ndarray) -> List[np.ndarray]:
        """Render placeholder frames (just copies of reference)."""
        frames = []
        for _ in range(len(listener_3dmm)):
            frames.append(listener_reference.copy())
        
        print(f"Generated {len(frames)} placeholder frames")
        return frames
    
    def _create_video_from_frames(self, 
                                 frames: List[np.ndarray], 
                                 output_video: str, 
                                 fps: int = 25):
        """Create video from list of frames."""
        if not frames:
            raise ValueError("No frames provided for video creation")
        
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise RuntimeError(f"Failed to create video writer for {output_video}")
        
        try:
            # Write frames
            for frame in frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                out.write(frame)
            
            expected_duration = len(frames) / fps
            print(f"Video created with {len(frames)} frames at {fps} FPS")
            print(f"Expected video duration: {expected_duration:.2f} seconds")
            
        finally:
            out.release()
    
    def render_batch(self, 
                    output_dir: str, 
                    video_names: List[str], 
                    listener_3dmm_batch: torch.Tensor, 
                    listener_reference_batch: torch.Tensor,
                    fps: int = 25,
                    chunk_size: int = 32) -> List[str]:
        """
        Render multiple videos in batch.
        
        Args:
            output_dir: Directory to save rendered videos
            video_names: List of video names
            listener_3dmm_batch: Batch of 3DMM parameters (B, T, 58)
            listener_reference_batch: Batch of reference frames (B, C, H, W)
            fps: Frames per second for output videos
            chunk_size: Number of frames to process at once (to avoid OOM)
            
        Returns:
            List of paths to generated video files
        """
        video_paths = []
        
        for i, (video_name, listener_3dmm, listener_reference) in enumerate(
            zip(video_names, listener_3dmm_batch, listener_reference_batch)
        ):
            try:
                video_path = self.rendering(
                    output_dir, 
                    video_name, 
                    listener_3dmm, 
                    listener_reference, 
                    fps,
                    chunk_size=chunk_size
                )
                video_paths.append(video_path)
                print(f"Batch {i+1}/{len(video_names)} completed: {video_name}")
                
            except Exception as e:
                logging.error(f"Failed to render {video_name}: {e}")
                video_paths.append(None)
        
        return video_paths
    
    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage."""
        memory_info = {}
        if torch.cuda.is_available():
            memory_info = {
                'allocated': torch.cuda.memory_allocated(self.device) / 1024**3,  # GB
                'reserved': torch.cuda.memory_reserved(self.device) / 1024**3,    # GB
                'max_allocated': torch.cuda.max_memory_allocated(self.device) / 1024**3,  # GB
            }
        return memory_info
    
    def clear_memory_cache(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cache cleared")


def create_renderer(device: str = 'cpu', use_pirender: bool = True) -> Render:
    """
    Factory function to create a renderer instance.
    
    Args:
        device: Device to run rendering on ('cpu' or 'cuda')
        use_pirender: Whether to use PIRender for actual 3D rendering
        
    Returns:
        Render instance
    """
    return Render(device=device, use_pirender=use_pirender)


# For backward compatibility
def render_video(output_dir: str, 
                video_name: str, 
                listener_3dmm: torch.Tensor, 
                listener_reference: torch.Tensor,
                device: str = 'cpu',
                fps: int = 25,
                chunk_size: int = 32) -> str:
    """
    Convenience function for rendering a single video.
    
    Args:
        output_dir: Directory to save rendered video
        video_name: Name for the output video
        listener_3dmm: 3DMM parameters for listener (T, 58)
        listener_reference: Reference frame for listener
        device: Device to run rendering on
        fps: Frames per second for output video
        chunk_size: Number of frames to process at once (to avoid OOM)
        
    Returns:
        Path to the generated video file
    """
    renderer = create_renderer(device)
    return renderer.rendering(output_dir, video_name, listener_3dmm, listener_reference, fps, chunk_size=chunk_size)
