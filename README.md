# Detecting Bark Beetle Outbreaks through Geospatial Foundation Models
University of Washington MSDS Capstone Project 2026

## Section 1: Environment and Setup

Follow setup instructions from Olmo segmentation docs here: 

https://github.com/allenai/olmoearth_projects/blob/main/docs/tutorials/FinetuneOlmoEarthSegmentation.md

Then:

```bash
cd olmoearth_projects && source .venv/bin/activate
```

## Section 2: Data Processing
We detail which datasets we use and how to correct process and format them for OlmoEarth ingest.

### Section 2.1: DEFID2
We will be using The Database of European Forest Insect and Disease Disturbances (DEFID2), a database that has harmonized georeferenced records of European forests that were disturbed by insects and/or diseases between 2018 and 2021. This is available for download through the European Commission’s Joint Research Centre Data Catalogue, and provides important spatiotemporal information about the spread of bark beetles in the form of polygons with respective geolocations as well as qualitative labels detailing canopy discoloration and defoliation.

Download here: https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/FOREST/DISTURBANCES/DEFID2/

Then navigate to ____ notebook to process the raw records into a geopandas dataframe.

### Section 2.2: USFS 2023

## Section 3: Model Training and Evaluation

### Section 3.1: Configuration

#### Section 3.1.1: Dataset Config

Example SuperDove configuration update:
```json
"superdove": {
  "band_sets": [
    {
      "bands": ["b01", "b02", "b03", "b04","b05", "b06", "b07", "b08"],
      "dtype": "uint16"
    }
  ],
  "data_source": {
    "time_offset": "120d",
    "duration": "184d",
    "cache_dir": "cache/superdove",
    "item_type_id": "PSScene",
    "asset_type_id": "ortho_analytic_8b_sr",
    "ingest": true,
    "name": "rslearn.data_sources.planet.Planet",
    "bands":["b01", "b02", "b03", "b04","b05", "b06", "b07", "b08"],
    "query_config": {
      "time_mode": "WITHIN",
      "max_matches": 1
    },
    "range_filters": {
      "cloud_cover": {"lte": 1.0}
    },
    "sort_by": "cloud_cover"
  },
  "type": "raster"
}
```


Example Dove-R configuration update:
```json
"doveR": {
  "band_sets": [
    {
      "bands": ["b01", "b02", "b03", "b04"],
      "dtype": "uint16"
    }
  ],
  "data_source": {
    "time_offset": "0d",
    "duration": null,
    "cache_dir": "cache/doveR",
    "item_type_id": "PSScene",
    "asset_type_id": "ortho_analytic_4b_sr",
    "ingest": true,
    "name": "rslearn.data_sources.planet.Planet",
    "bands":[
        "b01",
        "b02",
        "b03",
        "b04"
      ],
    "query_config": {
      "time_mode": "WITHIN",
      "max_matches": 1
    },
    "range_filters": {
      "cloud_cover": {"lte": 1.0}
    },
    "sort_by": "cloud_cover"
  },
  "type": "raster"
}
```

#### Section 3.1.2: Model Config

Example addition of **SuperDove** to model config:
```yaml
psb_sd:
  data_type: "raster"
  layers: [
            "superdove"
          ]
  bands: ["b01", "b02", "b03", "b04","b05", "b06", "b07", "b08"]
  passthrough: true
  dtype: FLOAT32
  load_all_layers: true
  load_all_item_groups: true
  required: true
```

Example addition of **Dove-R** to model config:
```yaml
ps2_sd:
  data_type: "raster"
  layers: [
            "doveR",
          ]
  bands: ["b01", "b02", "b03", "b04"]
  passthrough: true
  dtype: FLOAT32
  load_all_layers: true
  load_all_item_groups: true
  required: true
```

### Section 3.2: Building a Dataset

Run the OER scrit to generate annotations json; assuming $RAW is the dir holding all raw geodataframes
```bash
python ./scripts/oer_annotation_creation.py $RAW/cluster_all.gdb --outdir $PROJECT_PATH --id-col cluster_id --taskgeom-col task_geom_buff --buffer 0 --label-cols label
```

Prepare the labeled windows from the 

```bash
python -m olmoearth_projects.main olmoearth_run prepare_labeled_windows --project_path $PROJECT_PATH --scratch_path $OER_DATASET_PATH
```

Build the windows - most time intensive!
```bash
python -m olmoearth_projects.main olmoearth_run build_dataset_from_windows --project_path $PROJECT_PATH --scratch_path $OER_DATASET_PATH
```

### Section 3.3: Training OlmoEarth

After setting up the model configuration file, fine-tune OlmoEarth on the processed windows:

```bash
python -m olmoearth_projects.main olmoearth_run finetune --project_path $PROJECT_PATH --scratch_path $OER_DATASET_PATH
```

### Section 3.4: Testing
After fine-tuning is complete, evaluate on the validation or test set:

```bash
rslearn model test --config $PROJECT_PATH/model.yaml \
                   --data.init_args.path $OER_DATASET_PATH/dataset/ \
                   --ckpt_path $OER_DATASET_PATH/trainer_checkpoints/last.ckpt \
                   --trainer.callbacks=[] \
                   --data.init_args.num_workers 8
```
*Note: this is an example checkpoint path - insert whatever checkpoint file for this arg*

### Section 3.5: Output Prediction Generation

Update model config first to include the following trainer callback:

```yaml
- class_path: rslearn.train.prediction_writer.RslearnWriter
  init_args:
    path: placeholder 
    output_layer: output
    selector: ["burnscar_segmentation"] # <<< or whatever task selector you have 
    merger:
      class_path: rslearn.train.prediction_writer.RasterMerger
      init_args:
        padding: 8
```

```bash
rslearn model predict --config $PROJECT_PATH/model.yaml \
                      --data.init_args.path $OER_DATASET_PATH/dataset/ \
                      --ckpt_path $OER_DATASET_PATH/trainer_checkpoints/last.ckpt \
                      --data.init_args.num_workers 8
```

### Section 3.6: Embedding Generation

I would suggest creating a separate embedding-specific config, e.g.

```yaml
model:
  class_path: rslearn.train.lightning_module.RslearnLightningModule
  init_args:
    model:
      class_path: rslearn.models.singletask.SingleTaskModel
      init_args:
        encoder:
          - class_path: rslearn.models.olmoearth_pretrain.model.OlmoEarth
            init_args:
              model_id: "OLMOEARTH_V1_TINY"
              patch_size: 4
              selector: ["encoder"]
        decoder:
            - class_path: rslearn.train.tasks.embedding.EmbeddingHead
              init_args:
                feature_map_index: 0
    lr: 0.0001
    plateau: true
    plateau_factor: 0.2
    plateau_patience: 2
    plateau_min_lr: 0
    plateau_cooldown: 10
data:
  class_path: rslearn.train.data_module.RslearnDataModule
  init_args:
    path: ${DATASET_PATH}
    inputs:
      sentinel2_l2a:
          data_type: "raster"
          layers: [
                  "sentinel2_l2a"
                  ]
          bands: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
          passthrough: true
          dtype: FLOAT32
          load_all_layers: true
          load_all_item_groups: true
      label:
        data_type: "raster"
        layers: ["label"]
        bands: ["label"]
        dtype: INT32
        is_target: true
    task:
      # The EmbeddingTask is a dummy task setup so that the output feature map can be
      # written to the rslearn dataset during `model predict`.
      # burnscar_segmentation:
      class_path: rslearn.train.tasks.embedding.EmbeddingTask
    batch_size: 1
    num_workers: ${NUM_WORKERS}
    default_config:
      transforms:
        - class_path: rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize
          init_args:
            band_names:
              sentinel2_l2a: ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
      patch_size: 64
    train_config:
      groups: ["spatial_split_10km"]
      # By default (patch_size=null), data for the entire window bounds is read. This
      # can be cropped using transforms, but if a random crop is desired, it is more
      # efficient to crop it with this option, since this way the cropping will happen
      # when reading GeoTIFFs. However, setting it here is less flexible, since it only
      # supports random cropping.
      patch_size: 64
      # For validation, testing, and prediction, patch_size can be combined with
      # load_all_crops to perform sliding window inference. For training, it should
      # usually be left false so that each training epoch sees a different crop.
      load_all_patches: false # randomly select new patches every epoch from full size image
      # This should typically be enabled for predict_config, so that windows without
      # layers containing targets are skipped. For training, validation, and testing,
      # targets are needed so it should be false.
      skip_targets: false
      # split by tag in metadata.json
      tags:
        split: "train"
    val_config:
      groups: ["spatial_split_10km"]
      patch_size: 64
      load_all_patches: true #tmp
      tags:
        split: "val"
    test_config:
      groups: ["spatial_split_10km"]
      patch_size: 64
      load_all_patches: true
      tags:
        split: "test"
    predict_config:
      groups: ["spatial_split_10km"]
      patch_size: 64
      load_all_patches: true
      overlap_ratio: 0.5  # 16 / 64
      # skip_targets: true
trainer:
  max_epochs: 100
  check_val_every_n_epoch: 5
  callbacks:
    - class_path: lightning.pytorch.callbacks.LearningRateMonitor
      init_args:
        logging_interval: "epoch"
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        dirpath: ${TRAINER_DATA_PATH}
        save_top_k: 1
        save_last: true
        monitor: val_loss
        mode: min
    - class_path: rslearn.train.callbacks.freeze_unfreeze.FreezeUnfreeze
      init_args:
        module_selector: ["model", "encoder", 0]
        unfreeze_at_epoch: 10
        unfreeze_lr_factor: 10
    - class_path: rslearn.train.prediction_writer.RslearnWriter
      init_args:
        path: placeholder
        output_layer: embeddings
        merger:
          class_path: rslearn.train.prediction_writer.RasterMerger
          init_args:
            # This lets the merger know what output resolution to expect relative to
            # the window's resolution. Here, our output will be 1/patch_size relative
            # to the window resolution (input resolution), since we compute one
            # embedding per patch in the input, so we set the downsample_factor to
            # patch_size.
            downsample_factor: 4 #must match patch size!!! otherwise null region present in embedding raster
            padding: 0
  logger:
    class_path: lightning.pytorch.loggers.WandbLogger
    init_args:
      project: ${WANDB_PROJECT}
      name: ${WANDB_NAME}
      entity: ${WANDB_ENTITY}
# unused: ${EXTRA_FILES_PATH}
```

```bash
 rslearn model predict --config /data/datasets/olmo/usfs/processed/model_emb.yaml --data.init_args.path /data/datasets/olmo/usfs/oerun/dataset/ --ckpt_path /data/datasets/olmo/defid2/oerun_cluster_30d_trim/trainer_checkpoints_GA5DWODMA4/last_rev.ckpt  --data.init_args.num_workers 8
```

## Section 4: Output Prediction Visualization

TBD

## Section 5: Embedding Analysis

TBD
