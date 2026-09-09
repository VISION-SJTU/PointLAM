# Waymo workflow

This release provides the PointLAM single-frame Waymo workflow. OpenPCDet's general dataset and model infrastructure is retained, but PointLAM configurations are supplied for Waymo only.

Complete [installation](INSTALL.md) first. Data preparation commands below run from the repository root; training and validation commands run from `tools/`.

## Data preparation

Obtain the single-frame Perception TFRecord data from the [Waymo Open Dataset](https://waymo.com/open/download/). Use the standard 798 training and 202 validation segments used by OpenPCDet. This loader reads the original `Frame` TFRecord format, rather than the modular Parquet format.

The repository includes the train/validation filename lists under `data/waymo/ImageSets/`. They are copied unchanged from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/tree/255db8f02a8bd07211d2c91f54602d63c4c93356/data/waymo/ImageSets) and match LION's lists. Keep train and validation membership separate. Place the extracted TFRecords together in `data/waymo/raw_data/`; a symlink to your data storage is also suitable.

```text
data/waymo/
├── ImageSets/
│   ├── train.txt
│   └── val.txt
└── raw_data/
    ├── segment-...tfrecord
    └── ...
```

Generate single-frame point clouds, annotations and the ground-truth database:

```bash
python -m pcdet.datasets.waymo.waymo_dataset \
    --func create_waymo_infos \
    --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml
```

This processes both splits at interval 1 and then builds the training database. The preparation entry uses GPU 0 for the database step. It writes:

```text
data/waymo/
├── waymo_processed_data_v0_5_0/                 # points and per-segment metadata
├── waymo_processed_data_v0_5_0_infos_train.pkl
├── waymo_processed_data_v0_5_0_infos_val.pkl
├── waymo_processed_data_v0_5_0_gt_database_train_sampled_1/
├── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl
└── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_global.npy
```

The PointLAM model config expects these single-frame files. The ground-truth sampler reads per-object files; its shared-memory cache remains disabled in this implementation. The generated global database array is not required by that runtime path. The dataset-wide shared-memory cache is a separate option and is disabled by default.

## Training from scratch

```bash
cd tools
bash scripts/waymo_train_pointlam.sh
```

The launcher defaults to GPUs `0,1,2,3`, 24 epochs, global batch 16, eight data workers per process, SyncBatchNorm, and experiment tag `default`. It keeps up to 12 epoch checkpoints. The recipe uses AdamW with the existing one-cycle schedule and peak learning rate 0.003; the last epoch disables the five configured data augmentations.

The launcher accepts an experiment tag, epoch count and GPU list:

```bash
bash scripts/waymo_train_pointlam.sh my_run 24 0,1,2,3
```

Changing the GPU count does not change the launcher's global batch size of 16. Use the four-GPU setting for the supplied recipe.

Outputs are written below the repository's `output/pointlam_models/pointlam_waymo/<experiment_tag>/` directory. Checkpoints are in its `ckpt/` subdirectory.

## Resume training

Use the same recipe and GPU count, and pass the saved checkpoint explicitly:

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29988 \
    train.py --launcher pytorch --tcp_port 29988 \
    --cfg_file cfgs/pointlam_models/pointlam_waymo.yaml \
    --batch_size 16 --epochs 24 --workers 8 --sync_bn \
    --max_ckpt_save_num 12 --extra_tag default \
    --ckpt /absolute/path/to/checkpoint_epoch_N.pth
```

The training entry restores the model, optimizer, epoch and iteration from the checkpoint. `--pretrained_model` initializes model weights and is not a substitute for resuming optimizer state.

## Validation

Evaluate a checkpoint you trained:

```bash
python test.py --cfg_file cfgs/pointlam_models/pointlam_waymo.yaml \
    --ckpt /absolute/path/to/checkpoint_epoch_24.pth
```

The inherited `test.py` filename is the validation entry. In the config, `DATA_SPLIT.test` maps to `val`. It evaluates every validation frame with the official Waymo metric implementation and writes predictions and metrics under the experiment output directory.

Without `--batch_size`, validation uses the config's batch size of 4 per GPU. An explicit batch size is interpreted as the global batch size. Point shuffling is enabled and the entry seeds NumPy with 1024; Torch sampling is not additionally fixed by this entry. Use the same batch size and random-state policy when comparing runs.

For distributed validation:

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29989 \
    test.py --launcher pytorch --tcp_port 29989 \
    --cfg_file cfgs/pointlam_models/pointlam_waymo.yaml --batch_size 4 \
    --ckpt /absolute/path/to/checkpoint_epoch_24.pth
```

The supplied legacy entry points accept the local-rank argument passed by `torch.distributed.launch`; use the commands above. They do not read Torchrun's `LOCAL_RANK` environment variable.

No official test-set preparation or submission workflow is included.
