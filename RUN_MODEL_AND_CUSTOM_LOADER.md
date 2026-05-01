# UETrack: Run the Model and Use the Custom Loader

This is the single guide to follow for:
- running UETrack
- enabling LASS
- using your custom video dataset loader

## 1) Set up the environment

```bash
conda create -n uetrack python=3.10
conda activate uetrack
bash install.sh
export PYTHONPATH=/home/bab/Main/mind_cloud_27/UETrack:$PYTHONPATH
```

Then create the default project path files:

```bash
cd /home/bab/Main/mind_cloud_27/UETrack
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```

## 2) Enable the model

Use the RGB tiny config if you want the simplest setup:
- [experiments/uetrack/uetrack_tiny.yaml](experiments/uetrack/uetrack_tiny.yaml)

Make sure these are set:
- `MODEL.ENCODER.USE_LASS: true`
- `MODEL.ENCODER.PRETRAIN_TYPE` points to your backbone checkpoint
- `TRAIN.TEACHER_PATH` points to your teacher checkpoint

For training, the dataset list is controlled by:
- `DATA.TRAIN.DATASETS_NAME`
- `DATA.TRAIN.DATASETS_RATIO`

## 3) Prepare your custom dataset

### Folder layout

Put the custom data under:

```text
data/custom_manifest/
├── contestant_manifest.json   # used by the training loader (train split)
├── manifest.json              # used by the evaluation loader (includes public_lb)
├── train/
│   └── sequence_1/
│       ├── sequence_1.mp4
│       └── annotation.txt
└── public_lb/
    └── sequence_101/
        ├── sequence_101.mp4
        └── annotation.txt (optional for public_lb, can be placeholders)
```

Splits supported:
- **train**: Used for training (required for `CUSTOM_MANIFEST` dataset)
- **public_lb**: Used for evaluation/leaderboard (public test set)
- **test**: Alternative name for evaluation (fallback if public_lb not found)

### Annotation format

Each line in `annotation.txt` must be:

```text
x,y,width,height
```

Example:

```text
109,443,37,81
110,445,36,82
111,447,38,80
```

### Manifest format

Example `contestant_manifest.json` (for training - train split):

```json
{
  "train": {
    "dataset1/sequence_1": {
      "video_path": "train/dataset1/sequence_1/sequence_1.mp4",
      "annotation_path": "train/dataset1/sequence_1/annotation.txt",
      "n_frames": 500
    },
    "dataset1/sequence_2": {
      "video_path": "train/dataset1/sequence_2/sequence_2.mp4",
      "annotation_path": "train/dataset1/sequence_2/annotation.txt",
      "n_frames": 678
    }
  }
}
```

Example `manifest.json` (for evaluation - includes public_lb split):

```json
{
  "train": {
    "dataset1/sequence_1": {
      "video_path": "train/dataset1/sequence_1/sequence_1.mp4",
      "annotation_path": "train/dataset1/sequence_1/annotation.txt",
      "n_frames": 500
    }
  },
  "public_lb": {
    "dataset1/sequence_101": {
      "video_path": "public_lb/dataset1/sequence_101/sequence_101.mp4",
      "annotation_path": "public_lb/dataset1/sequence_101/annotation.txt",
      "n_frames": 450
    },
    "dataset1/sequence_102": {
      "video_path": "public_lb/dataset1/sequence_102/sequence_102.mp4",
      "annotation_path": "public_lb/dataset1/sequence_102/annotation.txt",
      "n_frames": 500
    }
  }
}
```

## 4) Register the custom loader paths

### Training path

Edit [lib/train/admin/local.py](lib/train/admin/local.py) and set:

```python
self.custom_manifest_dir = '/home/YOUR_USERNAME/data/custom_manifest'
```

### Evaluation path

Edit [lib/test/evaluation/local.py](lib/test/evaluation/local.py) and set:

```python
settings.custom_manifest_path = '/home/YOUR_USERNAME/data/custom_manifest'
```

## 5) Use the custom loader for training

The training pipeline accepts the dataset name `CUSTOM_MANIFEST`.

### Use only your custom dataset

Set:

```yaml
DATA:
  TRAIN:
    DATASETS_NAME:
    - CUSTOM_MANIFEST
    DATASETS_RATIO:
    - 1
```

### Mix with standard datasets

Example:

```yaml
DATA:
  TRAIN:
    DATASETS_NAME:
    - LASOT
    - GOT10K_vottrain
    - COCO17
    - CUSTOM_MANIFEST
    DATASETS_RATIO:
    - 4
    - 4
    - 4
    - 4
```

## 6) Train the model

### Single GPU

```bash
python lib/train/run_training.py --script uetrack --config uetrack_tiny --save_dir .
```

### Multi-GPU

```bash
python -m torch.distributed.launch --nproc_per_node 2 lib/train/run_training.py --script uetrack --config uetrack_tiny --save_dir .
```

## 7) Test and evaluate

### Standard benchmarks

```bash
python tracking/test.py uetrack uetrack_tiny --dataset lasot --threads 2
python tracking/test.py uetrack uetrack_tiny --dataset got10k_test --threads 2
python tracking/test.py uetrack uetrack_tiny --dataset trackingnet --threads 2
```

### Custom dataset (test/public_lb)

#### Option 1: Generate CSV submission directly

Run evaluation on public_lb split and generate CSV:

```bash
python tracking/evaluate_custom_loader.py \
    --tracker_name uetrack \
    --tracker_param uetrack_tiny \
    --dataset_name custom_manifest \
    --split public_lb \
    --output_dir results \
    --csv_output submission.csv
```

#### Option 2: Manual tracking + CSV generation

First run tracking:

```bash
python tracking/test.py uetrack uetrack_tiny --dataset custom_manifest --threads 2
```

Then generate CSV from results:

```bash
python tracking/evaluate_custom_loader.py \
    --tracker_name uetrack \
    --tracker_param uetrack_tiny \
    --dataset_name custom_manifest \
    --skip_tracking \
    --csv_output submission.csv
```

#### CSV Output Format

The evaluation produces `submission.csv` with format:

```csv
id,x,y,w,h
dataset1/Car_video_0,109,443,37,81
dataset1/Car_video_1,110,445,36,82
dataset1/Car_video_2,111,447,38,80
```

Where:
- `id`: sequence name (from manifest public_lb split)
- `x,y,w,h`: predicted bounding box at first frame

## 8) Quick checklist

- [ ] Conda env created
- [ ] `install.sh` completed
- [ ] `PYTHONPATH` set
- [ ] `create_default_local_file.py` ran successfully
- [ ] `self.custom_manifest_dir` set in training local settings
- [ ] `settings.custom_manifest_path` set in test local settings
- [ ] `contestant_manifest.json` exists for training (train split)
- [ ] `manifest.json` exists for evaluation (includes public_lb split)
- [ ] `DATA.TRAIN.DATASETS_NAME` includes `CUSTOM_MANIFEST`
- [ ] `USE_LASS: true` if you want the efficient attention version
- [ ] Training folder structure has train/ with video files and annotations
- [ ] Evaluation folder structure has public_lb/ with video files and annotations

## 9) Minimal example workflow

```bash
# Setup
cd /home/bab/Main/mind_cloud_27/UETrack
conda activate uetrack
export PYTHONPATH=$PWD:$PYTHONPATH
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .

# Train
python lib/train/run_training.py --script uetrack --config uetrack_tiny --save_dir .

# Evaluate and generate CSV submission
python tracking/evaluate_custom_loader.py \
    --tracker_name uetrack \
    --tracker_param uetrack_tiny \
    --dataset_name custom_manifest \
    --split public_lb \
    --output_dir results \
    --csv_output submission.csv

# View submission
head submission.csv
```

## 10) Directory structure for complete setup

```text
UETrack/
├── data/
│   └── custom_manifest/
│       ├── contestant_manifest.json    (train split for training)
│       ├── manifest.json               (train + public_lb splits for eval)
│       ├── train/
│       │   └── dataset1/
│       │       ├── Car_video_0/
│       │       │   ├── Car_video_0.mp4
│       │       │   └── annotation.txt
│       │       └── Car_video_1/
│       │           ├── Car_video_1.mp4
│       │           └── annotation.txt
│       └── public_lb/
│           └── dataset1/
│               ├── Car_video_100/
│               │   ├── Car_video_100.mp4
│               │   └── annotation.txt (optional, can have dummy content)
│               └── Car_video_101/
│                   ├── Car_video_101.mp4
│                   └── annotation.txt
├── experiments/
│   └── uetrack/
│       ├── uetrack_tiny.yaml
│       ├── uetrack_small.yaml
│       └── uetrack_base.yaml
├── lib/
│   ├── train/
│   │   ├── admin/
│   │   │   ├── environment.py
│   │   │   └── local.py (auto-generated)
│   │   ├── dataset/
│   │   │   └── custom_manifest.py ← Training loader for CUSTOM_MANIFEST
│   │   └── run_training.py
│   └── test/
│       └── evaluation/
│           ├── custom_manifest_dataset.py ← Evaluation loader for custom_manifest
│           ├── datasets.py
│           ├── environment.py
│           └── local.py (auto-generated)
├── tracking/
│   ├── test.py
│   ├── train.py
│   └── evaluate_custom_loader.py ← Generate CSV submissions
└── submission.csv (output)
```

## 11) Dataset naming and splits

- **Training dataset name**: `CUSTOM_MANIFEST` (uppercase in config)
- **Evaluation dataset name**: `custom_manifest` (lowercase for test.py)
- **Training manifest**: `contestant_manifest.json` with train split
- **Evaluation manifest**: `manifest.json` with public_lb split (primary)

## 12) CSV submission format

Output `submission.csv`:
- Header: `id,x,y,w,h`
- Each row: `dataset_name/video_name,x,y,w,h`
- One prediction per video from public_lb split
- Use first frame bbox by default
