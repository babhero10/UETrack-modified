import os
import json
import numpy as np
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
import cv2


class CustomManifest(BaseVideoDataset):
    """Custom dataset loader from JSON manifest file.
    
    Supports video files with annotations in bounding box format (x, y, width, height).
    
    JSON Structure:
    {
        "train": {
            "video_id": {
                "video_path": "path/to/video.mp4",
                "annotation_path": "path/to/annotation.txt",
                "n_frames": 100,
                ...
            }
        },
        "test": {...}
    }
    
    Annotation format: x,y,width,height (one per line)
    """
    
    def __init__(self, root=None, image_loader=jpeg4py_loader, manifest_path=None, 
                 split='train', data_fraction=None, multi_modal_vision=False, 
                 multi_modal_language=False, use_nlp=False):
        """
        args:
            root - Root path where videos and annotations are stored
            image_loader - Function to read images
            manifest_path - Path to the JSON manifest file
            split - 'train' or 'test'
            data_fraction - Fraction of dataset to use
        """
        root = env_settings().custom_dir if root is None else root
        super().__init__('CustomManifest', root, image_loader)
        
        # Load manifest
        if manifest_path is None:
            manifest_path = os.path.join(root, 'contestant_manifest.json')
        
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        self.split = split
        self.sequence_list = self._build_sequence_list()
        
        if data_fraction is not None:
            import random
            self.sequence_list = random.sample(self.sequence_list, 
                                              int(len(self.sequence_list) * data_fraction))
        
        self.sequence_meta = OrderedDict()
        self._load_metadata()
    
    def _build_sequence_list(self):
        """Build list of sequences from manifest."""
        sequences = []
        # Support 'train' and 'public_lb' splits
        split_name = self.split if self.split in self.manifest else 'train'
        if split_name in self.manifest:
            for video_id in self.manifest[split_name].keys():
                sequences.append(video_id)
        return sequences
    
    def _load_metadata(self):
        """Load metadata for all sequences."""
        for seq_id in self.sequence_list:
            meta = self.manifest[self.split][seq_id]
            self.sequence_meta[seq_id] = {
                'video_path': os.path.join(self.root, meta['video_path']),
                'annotation_path': os.path.join(self.root, meta['annotation_path']),
                'n_frames': meta.get('n_frames', None),
                'fps': meta.get('native_fps', 30)
            }
    
    def _read_annotations(self, annotation_path):
        """Read bounding box annotations from file.
        
        Format: x,y,width,height (one per line)
        
        returns:
            np.ndarray - Bounding boxes (N, 4) in format [x, y, w, h]
        """
        bboxes = []
        try:
            with open(annotation_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            # Try parsing as x,y,width,height
                            bbox = list(map(float, line.split(',')))
                            if len(bbox) == 4:
                                bboxes.append(bbox)
                        except:
                            pass
        except Exception as e:
            print(f"Error reading annotations from {annotation_path}: {e}")
        
        return np.array(bboxes, dtype=np.float32)
    
    def _extract_frames_from_video(self, video_path, frame_ids):
        """Extract frames from video file.
        
        args:
            video_path - Path to video file
            frame_ids - List of frame indices to extract
            
        returns:
            list - List of frame arrays
        """
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Failed to open video: {video_path}")
            
            for frame_id in frame_ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                else:
                    # If frame can't be read, use black frame
                    if len(frames) > 0:
                        frames.append(np.zeros_like(frames[-1]))
                    else:
                        frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
            
            cap.release()
        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")
            frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in frame_ids]
        
        return frames
    
    def get_name(self):
        """Dataset name."""
        return 'CustomManifest'
    
    def has_class_info(self):
        """Whether dataset has class information."""
        return False
    
    def has_occlusion_info(self):
        """Whether dataset has occlusion information."""
        return False
    
    def get_sequence_info(self, seq_id):
        """Get information about a sequence.
        
        args:
            seq_id - Sequence index
            
        returns:
            dict - Sequence information
        """
        seq_name = self.sequence_list[seq_id]
        meta = self.sequence_meta[seq_name]
        
        # Load annotations
        annotations = self._read_annotations(meta['annotation_path'])
        
        return {
            'bbox': annotations,  # (N, 4) in [x, y, w, h] format
            'valid': np.ones(len(annotations), dtype=bool),
            'visible': np.ones(len(annotations), dtype=bool),
            'n_frames': len(annotations),
            'fps': meta['fps'],
            'seq_name': seq_name
        }
    
    def get_frames(self, seq_id, frame_ids, anno=None):
        """Get frames from a sequence.
        
        args:
            seq_id - Sequence index
            frame_ids - List of frame indices
            anno - Annotations (optional)
            
        returns:
            list - List of frames
            dict - Annotations
        """
        seq_name = self.sequence_list[seq_id]
        meta = self.sequence_meta[seq_name]
        
        # Extract frames from video
        frames = self._extract_frames_from_video(meta['video_path'], frame_ids)
        
        # Get annotations if not provided
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        
        return frames, anno
    
    def is_video_sequence(self):
        """Whether this is a video dataset."""
        return True
    
    def is_synthetic_video_dataset(self):
        """Whether this is synthetic."""
        return False
