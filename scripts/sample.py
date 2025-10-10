#!/usr/bin/env python3
"""
Main sampling script for ReactDiff.
Generates listener reactions from speaker audio and visual cues.
"""

import argparse
import os
import sys

import accelerate
import torch
import numpy as np
from tqdm import tqdm

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from models import k_diffusion as K
from utils.render import Render
from data.dataset import get_dataloader
from external.wav2vec2focctc import Wav2Vec2ForCTC


def main():
    p = argparse.ArgumentParser(description="Main sampling script for ReactDiff",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size', type=int, default=50,
                   help='the batch size')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='the checkpoint to use')
    p.add_argument('--config', type=str, required=True,
                   help='the model config')
    p.add_argument('-n', type=int, default=64,
                   help='the number of images to sample')
    p.add_argument('-t', type=int, default=256,
                   help='the number of timesteps to sample')
    p.add_argument('--prefix', type=str, default='out',
                   help='the output prefix')
    p.add_argument('--steps', type=int, default=50,
                   help='the number of denoising steps')
    p.add_argument('--num-workers', type=int, default=8,
                   help='the number of data loader workers')
    p.add_argument('--momentum', type=float, default=0.9,
                   help='the momentum for sampling')
    p.add_argument('--gpu-ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    p.add_argument('--out-path', default="./results/sample_output", type=str, help="output path")
    p.add_argument('--window-size', default=16, type=int, help="prediction window-size for online mode")
    p.add_argument('--sampling', type=str, default='sde', help='sampling strategy, ode, sde')
    p.add_argument('--multi-gpu', action='store_true',
                   help='enable multi-GPU sampling (auto-detected if multiple GPUs available)')
    p.add_argument('--mixed-precision', type=str, default='fp16',
                   help='the mixed precision type (fp16, bf16, or no)')

    args = p.parse_args()

    config = K.config.load_config(open(args.config))
    model_config = config['model']
    dataset_config = config['dataset']
    
    # Add dataset parameters from config to args
    args.dataset_path = dataset_config['location']
    args.img_size = dataset_config.get('img_size', 256)
    args.crop_size = dataset_config.get('crop_size', 224)
    args.clip_length = dataset_config.get('clip_length', 750)

    # Initialize Accelerate with multi-GPU support
    # Auto-detect multi-GPU if not explicitly specified
    if not args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs. Use --multi-gpu to enable multi-GPU sampling.")
    
    # Set up accelerate configuration
    if args.multi_gpu and torch.cuda.device_count() > 1:
        # Multi-GPU configuration
        accelerator = accelerate.Accelerator(
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=1,
            log_with=None,
            project_dir=None
        )
    else:
        # Single GPU configuration
        accelerator = accelerate.Accelerator(
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=1,
            log_with=None,
            project_dir=None
        )
    
    device = accelerator.device
    print('Using device:', device, flush=True)
    print(f'Number of processes: {accelerator.num_processes}', flush=True)
    print(f'Mixed precision: {accelerator.mixed_precision}', flush=True)

    # Load model
    inner_model = K.config.make_model(config).eval().requires_grad_(False).to(device)
    # inner_model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model_ema'])
    accelerator.print('Parameters:', K.utils.n_params(inner_model))
    model = K.Denoiser(inner_model, sigma_data=model_config['sigma_data'])

    # Load data
    val_dl = get_dataloader(args, "test", load_audio=True, load_video_s=False, load_video_l=False, 
                           load_emotion_s=False, load_emotion_l=False, load_3dmm_s=True, 
                           load_3dmm_l=False, load_ref=True)
    
    # Prepare data loader with Accelerate for multi-GPU support
    val_dl = accelerator.prepare(val_dl)

    # Setup rendering
    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']
    
    if torch.cuda.is_available():
        render = Render('cuda', use_pirender=True)  # Enable PIRender
        print("✅ PIRender enabled for GPU rendering")
    else:
        render = Render('cpu', use_pirender=False)  # Disable PIRender on CPU
        print("⚠️  PIRender disabled - using placeholder rendering on CPU")
    
    # Log memory usage if on GPU
    if torch.cuda.is_available():
        memory_info = render.get_memory_usage()
        print(f"GPU Memory: {memory_info.get('allocated', 0):.2f}GB allocated")

    # Load audio encoder
    audio_encoder = Wav2Vec2ForCTC.from_pretrained("../external/facebook/wav2vec2-base-960h").to(device)
    audio_encoder.freeze_feature_extractor()

    # Create output directory
    if accelerator.is_main_process:
        if not os.path.exists(args.out_path):
            os.makedirs(args.out_path)

        out_dir = os.path.join(args.out_path, 'test')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    
    # Wait for all processes to create directories
    accelerator.wait_for_everyone()
    
    # Create process-specific output directory for multi-GPU
    if args.multi_gpu and accelerator.num_processes > 1:
        out_dir = os.path.join(args.out_path, f'test_process_{accelerator.process_index}')
    else:
        out_dir = os.path.join(args.out_path, 'test')
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Sampling parameters
    sigmas = K.sampling.get_sigmas_karras(args.steps, sigma_min, sigma_max, rho=7., device=device)
    
    @torch.no_grad()
    @K.utils.eval_mode(model)
    def run_sampling():
        """Run the sampling process."""
        listener_3dmm_list = []
        speaker_3dmm_list = []

        video_dir = os.path.join(out_dir, 'video')
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
            
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_dl, disable=not accelerator.is_main_process)):
                with accelerator.accumulate(model):
                    _, speaker_audio_clip, _, speaker_3dmm, _, _, _, _, listener_references, listener_video_path = batch
                    
                    speaker_3dmm = speaker_3dmm.to(device)
                    listener_references = listener_references.to(device)
                    frame_num = speaker_3dmm.shape[1]
                    
                    # Debug: Print frame information
                    if accelerator.is_main_process:
                        print(f"Processing batch {batch_idx}: {frame_num} frames (expected ~{frame_num/25:.1f}s at 25fps)")
                        print(f"Window size: {args.window_size}, Clip length: {args.clip_length}")
                    
                    # Encode audio
                    speaker_audio_clip = audio_encoder(speaker_audio_clip.to(device), frame_num=frame_num)[:, : 2 * frame_num]

                    b, t, c = speaker_3dmm.shape
                    interval_num = t // args.window_size
                    x_0_out = None
                    past_frame = torch.zeros((speaker_3dmm.shape[0], 1, speaker_3dmm.shape[2])).to(speaker_3dmm.get_device())
                    
                    if accelerator.is_main_process:
                        print(f"Will process {interval_num} intervals of size {args.window_size}")
                        print(f"Total frames to generate: {t}")
                    
                    # Process in windows
                    for i in range(0, interval_num):
                        if i != (interval_num - 1):
                            interval = args.window_size
                            cond_dict = {}
                            cond_speaker_3dmm = speaker_3dmm[:, i * args.window_size: (i + 1) * args.window_size]
                            cond_speaker_audio = speaker_audio_clip[:, 2 * i * args.window_size: 2 * (i + 1) * args.window_size]
                            cond_dict['speaker_audio'] = cond_speaker_audio
                            cond_dict['speaker_3dmm'] = cond_speaker_3dmm
                        else:
                            interval = args.window_size + t % args.window_size
                            cond_speaker_3dmm = speaker_3dmm[:, i * args.window_size:]
                            cond_speaker_audio = speaker_audio_clip[:, 2 * i * args.window_size:]
                            cond_dict['speaker_audio'] = cond_speaker_audio
                            cond_dict['speaker_3dmm'] = cond_speaker_3dmm

                        # Sample
                        x = torch.randn([b, interval, model_config['input_channels']], device=device) * sigma_max
                        temporal_cond = torch.arange(i * args.window_size, i * args.window_size + interval)
                        temporal_cond = temporal_cond.view(1, -1, 1).repeat(b, 1, 1).view(b, -1, 1).to(device)

                        x_0 = K.sampling.sample_dpmpp_2m_sde(model, x, sigmas, past_frame, cond_dict, temporal_cond, disable=True)

                        # Apply momentum
                        if i != 0:
                            x_0[:,:,52:] = args.momentum * past_frame[:,:,52:] + (1 - args.momentum) * x_0[:,:,52:]
                            x_0_out = torch.cat((x_0_out, x_0), 1)
                        else:
                            x_0_out = x_0

                        past_frame = x_0[:, -1, :].unsqueeze(1)

                    # Render results with PIRender
                    if accelerator.is_main_process:
                        print(f"Generated {x_0_out.shape[1]} frames for rendering")
                        print(f"Expected video duration: {x_0_out.shape[1]/25:.1f} seconds")
                    
                    for j, out in enumerate(x_0_out):
                        video_path = listener_video_path[j]
                        video_path = video_path.split('/')
                        video_path = '_'.join(video_path)
                        
                        if accelerator.is_main_process:
                            print(f"Rendering video {j+1}/{x_0_out.shape[0]}: {out.shape[0]} frames")
                        
                        # Use PIRender with reference frame and chunked processing for long sequences
                        chunk_size = 32 if out.shape[0] > 100 else 64  # Adaptive chunk size
                        render.rendering(
                            output_dir=video_dir, 
                            video_name=video_path, 
                            listener_3dmm=out, 
                            listener_reference=listener_references[j],
                            fps=25,
                            chunk_size=chunk_size
                        )
                        
                listener_3dmm_list.append(x_0_out.cpu())
                speaker_3dmm_list.append(speaker_3dmm.cpu())

        listener_3dmm = torch.cat(listener_3dmm_list, dim=0)
        speaker_3dmm = torch.cat(speaker_3dmm_list, dim=0)

        return listener_3dmm, speaker_3dmm

    print("Starting sampling process...")
    try:
        listener_3dmm, speaker_3dmm = run_sampling()
        
        # Save results
        coeffs_out_dir = os.path.join(out_dir, 'coeffs')
        if not os.path.exists(coeffs_out_dir):
            os.makedirs(coeffs_out_dir)
            
        np.save(os.path.join(coeffs_out_dir, 'listener_3dmm.npy'), listener_3dmm.cpu().numpy().astype(np.float32))
        np.save(os.path.join(coeffs_out_dir, 'speaker_3dmm.npy'), speaker_3dmm.cpu().numpy().astype(np.float32))
        
        print(f"Sampling completed. Results saved to {out_dir}")
        
    except KeyboardInterrupt:
        print("Sampling interrupted by user.")
    except Exception as e:
        print(f"Error during sampling: {e}")


if __name__ == '__main__':
    main()
