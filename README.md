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

Then the output model predictions will be saved in the target directory specified in the model config.

### Section 3.6: Embedding Generation

I would suggest creating a separate embedding-specific config, e.g. [config](examples/model_emb.yaml)

```bash
 rslearn model predict --config $PROJECT_PATH/model_emb.yaml \
                       --data.init_args.path $OER_DATASET_PATH/dataset/ \
                       --ckpt_path $OER_DATASET_PATH/trainer_checkpoints/last.ckpt  \
                       --data.init_args.num_workers 8
```

Then the embeddings will be saved in the target directory specified in the model config.

## Section 4: Output Prediction Visualization

TBD

## Section 5: Embedding Analysis

TBD
