import os
import torch
from torch.utils import data
from torchvision import transforms
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random, math
import time
import pandas as pd
from PIL import Image
import soundfile as sf
import cv2
from torch.utils.data import DataLoader
import logging
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from external.wav2vec2focctc import Wav2Vec2ForCTC
import librosa
from multiprocessing import Pool
from scipy.io import loadmat

from functools import cmp_to_key


def validate_video_file(video_path):
    """
    Validate if a video file can be read properly.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not os.path.exists(video_path):
        return False, f"File does not exist: {video_path}"
    
    if not os.path.isfile(video_path):
        return False, f"Path is not a file: {video_path}"
    
    # Check file size
    file_size = os.path.getsize(video_path)
    if file_size == 0:
        return False, f"File is empty: {video_path}"
    
    # Try to open with OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return False, f"Cannot open video file: {video_path}"
    
    # Try to read first frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None or frame.size == 0:
        return False, f"Cannot read frame from video: {video_path}"
    
    return True, "Valid"


class Transform(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img):
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            normalize
        ])
        img = transform(img)
        return img


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def extract_video_features(video_path, img_transform):
    video_list = []
    video = cv2.VideoCapture(video_path)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame = img_transform(Image.fromarray(frame[:, :, ::-1])).unsqueeze(0)
        video_list.append(frame)
    video_clip = torch.cat(video_list, axis=0)
    return video_clip, fps, n_frames


class ReactionDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, root_path, split, img_size=256, crop_size=224, clip_length=751, total_length = 751, fps=25,
                 load_audio=True, load_video_s=True, load_video_l=True, load_emotion_s=False, load_emotion_l=False,
                 load_3dmm_s=False, load_3dmm_l=False, load_ref=True,  repeat_mirrored=False):
        """
        Args:
            root_path: (str) Path to the data folder.
            split: (str) 'train' or 'val' or 'test' split.
            img_size: (int) Size of the image.
            crop_size: (int) Size of the crop.
            clip_length: (int) Number of frames in a clip.
            fps: (int) Frame rate of the video.
            load_audio: (bool) Whether to load audio features.
            load_video_s: (bool) Whether to load speaker video features.
            load_video_l: (bool) Whether to load listener video features.
            load_emotion: (bool) Whether to load emotion labels.
            load_3dmm: (bool) Whether to load 3DMM parameters.
            repeat_mirrored: (bool) Whether to extend dataset with mirrored speaker/listener. This is used for val/test.
        """

        self._root_path = root_path
        self._img_loader = pil_loader
        self._clip_length = clip_length
        self._fps = fps
        self._split = split
        self._total_length = total_length
        self.img_size = img_size

        # self._data_path = os.path.join(self._root_path, self._split)
        self._data_path = self._root_path
        self._list_path = pd.read_csv(os.path.join(self._root_path, 'paper-list', self._split + '.csv'), header=None,
                                      delimiter=',')
        self._list_path = self._list_path.drop(0)

        self.load_audio = load_audio
        self.load_video_s = load_video_s
        self.load_video_l = load_video_l
        self.load_3dmm_s = load_3dmm_s
        self.load_3dmm_l = load_3dmm_l
        self.load_emotion_s = load_emotion_s
        self.load_emotion_l = load_emotion_l
        self.load_ref = load_ref

        self._audio_path = os.path.join(self._data_path, 'Audio_files')
        self._video_path = os.path.join(self._data_path, 'Video_files')
        self._emotion_path = os.path.join(self._data_path, 'Emotion')
        self._3dmm_path = os.path.join(self._data_path, '3D_FV_files')

        self.processor = Wav2Vec2Processor.from_pretrained("../external/facebook/wav2vec2-base-960h")

        self.mean_face = torch.FloatTensor(
            np.load('../external/FaceVerse/mean_face.npy').astype(np.float32)).view(1, -1)
        self.std_face = torch.FloatTensor(
            np.load('../external/FaceVerse/std_face.npy').astype(np.float32)).view(1, -1)

        self._transform = Transform(img_size, crop_size)

        speaker_path = list(self._list_path.values[:, 1])
        listener_path = list(self._list_path.values[:, 2])

        speaker_path_tmp = speaker_path + listener_path
        listener_path_tmp = listener_path + speaker_path
        speaker_path = speaker_path_tmp
        listener_path = listener_path_tmp

        self.data_list = []
        for i, (sp, lp) in enumerate(zip(speaker_path, listener_path)):
            ab_speaker_video_path = os.path.join(self._video_path, sp + '.mp4')
            ab_speaker_audio_path = os.path.join(self._audio_path, sp + '.wav')
            ab_speaker_emotion_path = os.path.join(self._emotion_path, sp + '.csv')
            ab_speaker_3dmm_path = os.path.join(self._3dmm_path, sp + '.npy')

            ab_listener_video_path = os.path.join(self._video_path, lp + '.mp4')
            ab_listener_audio_path = os.path.join(self._audio_path, lp + '.wav')
            ab_listener_emotion_path = os.path.join(self._emotion_path, lp + '.csv')
            ab_listener_3dmm_path = os.path.join(self._3dmm_path, lp + '.npy')

            self.data_list.append(
                {'speaker_video_path': ab_speaker_video_path, 'speaker_audio_path': ab_speaker_audio_path,
                 'speaker_emotion_path': ab_speaker_emotion_path, 'speaker_3dmm_path': ab_speaker_3dmm_path,
                 'listener_video_path': ab_listener_video_path, 'listener_audio_path': ab_listener_audio_path,
                 'listener_emotion_path': ab_listener_emotion_path, 'listener_3dmm_path': ab_listener_3dmm_path})

        self._len = len(self.data_list)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        data = self.data_list[index]

        # ========================= Data Augmentation ==========================

        speaker_prefix = 'speaker'
        listener_prefix = 'listener'

        # ========================= Load Speaker & Listener video clip ==========================
        speaker_video_path = data[f'{speaker_prefix}_video_path']
        listener_video_path = data[f'{listener_prefix}_video_path']

        if self._split == 'train':
            start = random.randint(0, self._total_length - 1 - self._clip_length) if self._clip_length < self._total_length else 0
            end = start + self._clip_length
        else:
            start = 0
            end = start + self._clip_length


        speaker_video_clip = 0
        if self.load_video_s:
            clip = []
            cap = cv2.VideoCapture(speaker_video_path)
            total_length = cap.get(7)
            idx = 0
            while True:
                ret = cap.grab()
                if not ret:
                    break
                if (idx >= start) and (idx <= end):
                    ret, frame = cap.retrieve()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    clip.append(self._transform(frame).unsqueeze(0))
                    if frame is None:
                        break
                elif idx > end:
                    break
                idx += 1
            cap.release()
            speaker_video_clip = torch.cat(clip, dim=0)

        # listener video clip only needed for val/test
        listener_video_clip = 0
        if self.load_video_l:
            cap = cv2.VideoCapture(listener_video_path)
            clip = []
            idx = 0
            while True:
                ret = cap.grab()
                if not ret:
                    break
                if (idx >= start) and (idx <= end):
                    ret, frame = cap.retrieve()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    clip.append(self._transform(frame).unsqueeze(0))
                    if frame is None:
                        break
                elif idx > end:
                    break
                idx += 1
            cap.release()
            listener_video_clip = torch.cat(clip, dim=0)

        # ========================= Load Speaker audio clip (listener audio is NEVER needed) ==========================
        listener_audio_clip, speaker_audio_clip = 0, 0
        if self.load_audio:
            speaker_audio_path = data[f'{speaker_prefix}_audio_path']
            speech_array, sampling_rate = librosa.load(speaker_audio_path, sr=16000)
            interval = sampling_rate // self._fps
            speaker_audio_clip = torch.FloatTensor(
                np.squeeze(self.processor(speech_array, sampling_rate=16000).input_values))
            speaker_audio_clip = speaker_audio_clip[int(start * interval): int((start + self._clip_length) * interval)].detach()

        # ========================= Load Speaker & Listener emotion ==========================
        listener_emotion, speaker_emotion = 0, 0
        if self.load_emotion_l:
            listener_emotion_path = data[f'{listener_prefix}_emotion_path']
            listener_emotion = pd.read_csv(listener_emotion_path, header=None, delimiter=',')
            listener_emotion = torch.from_numpy(np.array(listener_emotion.drop(0)).astype(np.float32))[
                               start: start + self._clip_length]

        if self.load_emotion_s:
            speaker_emotion_path = data[f'{speaker_prefix}_emotion_path']
            speaker_emotion = pd.read_csv(speaker_emotion_path, header=None, delimiter=',')
            speaker_emotion = torch.from_numpy(np.array(speaker_emotion.drop(0)).astype(np.float32))[
                              start: start + self._clip_length]

        # ========================= Load Listener 3DMM ==========================
        listener_3dmm = 0
        if self.load_3dmm_l:
            listener_3dmm_path = data[f'{listener_prefix}_3dmm_path']
            listener_3dmm = torch.FloatTensor(np.load(listener_3dmm_path)).squeeze()
            listener_3dmm = listener_3dmm[start: start + self._clip_length]
            listener_3dmm = (listener_3dmm - self.mean_face)  / self.std_face


        speaker_3dmm = 0
        if self.load_3dmm_s:
            speaker_3dmm_path = data[f'{speaker_prefix}_3dmm_path']
            speaker_3dmm = torch.FloatTensor(np.load(speaker_3dmm_path)).squeeze()
            speaker_3dmm = speaker_3dmm[start: start + self._clip_length]
            speaker_3dmm = (speaker_3dmm - self.mean_face)  / self.std_face

        # ========================= Load Listener Reference ==========================
        listener_reference = 0
        if self.load_ref:
            try:
                # Validate video file first
                is_valid, error_msg = validate_video_file(listener_video_path)
                
                if not is_valid:
                    print(f"Warning: {error_msg}")
                    # Create a default reference frame (neutral gray image)
                    listener_reference = torch.zeros((3, self.img_size, self.img_size))
                else:
                    # Video is valid, proceed with loading
                    cap = cv2.VideoCapture(listener_video_path)
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret and frame is not None and frame.size > 0:
                        # Successfully read frame, convert and transform
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        listener_reference = self._transform(frame)
                    else:
                        print(f"Warning: Failed to read frame from valid video: {listener_video_path}")
                        listener_reference = torch.zeros((3, self.img_size, self.img_size))
                            
            except Exception as e:
                print(f"Error loading reference frame from {listener_video_path}: {e}")
                # Fallback to default reference frame
                listener_reference = torch.zeros((3, self.img_size, self.img_size))


        return speaker_video_clip, speaker_audio_clip, speaker_emotion, speaker_3dmm, listener_video_clip, listener_audio_clip, listener_emotion, listener_3dmm, listener_reference, listener_video_path

    def __len__(self):
        return self._len
    
    def validate_all_videos(self):
        """
        Validate all video files in the dataset and report problematic ones.
        Useful for debugging dataset issues.
        """
        print("Validating all video files in the dataset...")
        problematic_videos = []
        
        for idx in tqdm(range(len(self)), desc="Validating videos"):
            try:
                data = self._data_list[idx]
                listener_video_path = data['listener_video_path']
                
                is_valid, error_msg = validate_video_file(listener_video_path)
                if not is_valid:
                    problematic_videos.append((idx, listener_video_path, error_msg))
                    
            except Exception as e:
                problematic_videos.append((idx, "Unknown", f"Error during validation: {e}"))
        
        if problematic_videos:
            print(f"\nFound {len(problematic_videos)} problematic videos:")
            for idx, path, error in problematic_videos[:10]:  # Show first 10
                print(f"  Index {idx}: {error}")
            if len(problematic_videos) > 10:
                print(f"  ... and {len(problematic_videos) - 10} more")
        else:
            print("All videos are valid!")
        
        return problematic_videos


def get_dataloader(conf, split, load_audio=False, load_video_s=False, load_video_l=False, load_emotion_s=False,
                   load_emotion_l=False, load_3dmm_s=False, load_3dmm_l=False, load_ref=False, repeat_mirrored=False):
    assert split in ["train", "val", "test"], "split must be in [train, val, test]"
    dataset = ReactionDataset(conf.dataset_path, split, img_size=conf.img_size, crop_size=conf.crop_size,
                              clip_length=conf.clip_length,
                              load_audio=load_audio, load_video_s=load_video_s, load_video_l=load_video_l,
                              load_emotion_s=load_emotion_s, load_emotion_l=load_emotion_l, load_3dmm_s=load_3dmm_s,
                              load_3dmm_l=load_3dmm_l, load_ref=load_ref,
                              repeat_mirrored=repeat_mirrored)
    shuffle = False
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=shuffle, num_workers=conf.num_workers, persistent_workers=True)
    return dataloader
