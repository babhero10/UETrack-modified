import numpy as np
import json
import os
import cv2
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class CustomManifestDataset(BaseDataset):
    """
    Custom dataset loaded from JSON manifest with video files and bbox annotations.
    
    Manifest Structure:
    {
        "test": {
            "sequence_name": {
                "video_path": "path/to/video.mp4",
                "annotation_path": "path/to/annotation.txt",
                "n_frames": 100
            }
        }
    }
    
    Annotation Format (per line):
        x,y,width,height (decimal format)
    """
    
    def __init__(self, manifest_dir):
        """
        Args:
            manifest_dir: Directory containing the manifest.json file
        """
        super().__init__()
        self.manifest_dir = manifest_dir
        
        # Load manifest
        manifest_path = os.path.join(manifest_dir, 'manifest.json')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest file not found at {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        # Get test sequences - prefer public_lb, fallback to test, then train
        self.sequences = []
        if 'public_lb' in self.manifest:
            self.sequences = list(self.manifest['public_lb'].keys())
        elif 'test' in self.manifest:
            self.sequences = list(self.manifest['test'].keys())
        elif 'train' in self.manifest:
            self.sequences = list(self.manifest['train'].keys())
        else:
            raise ValueError("Manifest must contain 'train', 'test', or 'public_lb' split")
        
        self.sequence_list = self._get_sequence_list()
    
    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])
    
    def _get_sequence_list(self):
        return self.sequences
    
    def _construct_sequence(self, sequence_name):
        """
        Construct a Sequence object from manifest entry.
        
        Args:
            sequence_name: Name of the sequence
            
        Returns:
            Sequence object with frames and ground truth bboxes
        """
        # Find sequence in manifest (check public_lb, test, then train)
        sequence_info = None
        for split in ('public_lb', 'test', 'train'):
            if split in self.manifest and sequence_name in self.manifest[split]:
                sequence_info = self.manifest[split][sequence_name]
                break
        if sequence_info is None:
            raise ValueError(f"Sequence {sequence_name} not found in manifest")
        
        # Get video and annotation paths
        video_path = sequence_info['video_path']
        annotation_path = sequence_info['annotation_path']
        n_frames = sequence_info.get('n_frames', None)
        
        # Make paths absolute if relative
        if not os.path.isabs(video_path):
            video_path = os.path.join(self.manifest_dir, video_path)
        if not os.path.isabs(annotation_path):
            annotation_path = os.path.join(self.manifest_dir, annotation_path)
        
        # Extract frames from video
        frames = self._extract_frames(video_path, n_frames)
        
        # Load annotations
        ground_truth_rect = self._load_annotations(annotation_path, n_frames)
        
        return Sequence(sequence_name, frames, 'custom_manifest', ground_truth_rect,
                       object_class='custom')
    
    def _extract_frames(self, video_path, n_frames=None):
        """
        Extract frame paths from video file.
        
        Args:
            video_path: Path to video file
            n_frames: Expected number of frames (for validation)
            
        Returns:
            List of frame file paths
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if n_frames is not None and total_frames != n_frames:
            print(f"Warning: Expected {n_frames} frames but video has {total_frames} frames")
        
        # Generate frame paths (frames are indexed in video)
        # We return video path with frame index for lazy loading
        frames = [{'video_path': video_path, 'frame_idx': i} for i in range(total_frames)]
        
        cap.release()
        return frames
    
    def _load_annotations(self, annotation_path, n_frames=None):
        """
        Load bounding box annotations from text file.
        
        Args:
            annotation_path: Path to annotation file
            n_frames: Expected number of frames
            
        Returns:
            Numpy array of shape (n_frames, 4) with [x, y, w, h] format
        """
        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        
        bboxes = []
        with open(annotation_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) != 4:
                    raise ValueError(f"Invalid annotation format: {line}. Expected x,y,w,h")
                try:
                    x, y, w, h = map(float, parts)
                    bboxes.append([x, y, w, h])
                except ValueError as e:
                    raise ValueError(f"Cannot parse annotation: {line}. Error: {e}")
        
        if n_frames is not None and len(bboxes) != n_frames:
            print(f"Warning: Expected {n_frames} annotations but got {len(bboxes)}")
        
        return np.array(bboxes, dtype=np.float64)
    
    def __len__(self):
        return len(self.sequences)
