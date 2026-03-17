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

Then navigate to [proc_defid2.ipynb](data_proc/proc_defid2.ipynb) notebook to process the raw records into a geopandas dataframe.

### Section 2.2: USFS 2023

Please refer to the previous team for this dataset: https://github.com/DSHydro/Insect_Forest_Infestation

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

#### Section 3.1.3 Env Variables

I would suggest setting these to facilitate cmds and API calls
```bash
export CUDA_VISIBLE_DEVICES=<GPU ID>
export WANDB_PROJECT="PROJECT-ID"
export WANDB_ENTITY="ACCOUNT-NAME"
export WANDB_NAME="RUN-NAME"
export OER_DATASET_PATH=<OERUN-PATH>
export PROJECT_PATH=<PROJECT-PATH>
export PL_API_KEY=<API KEY>
```

### Section 3.2: Building a Dataset

Run the OER scrit to generate annotations json; assuming $RAW is the dir holding all raw geodataframes
```bash
python ./scripts/oer_annotation_creation.py $RAW/cluster_all.gdb \
            --outdir $PROJECT_PATH \
            --id-col cluster_id \
            --taskgeom-col task_geom_buff \
            --buffer 0 \
            --label-cols label
```

Prepare the labeled windows from the 

```bash
python -m olmoearth_projects.main olmoearth_run prepare_labeled_windows \
        --project_path $PROJECT_PATH --scratch_path $OER_DATASET_PATH
```

Build the windows - most time intensive!
```bash
python -m olmoearth_projects.main olmoearth_run build_dataset_from_windows \
        --project_path $PROJECT_PATH --scratch_path $OER_DATASET_PATH
```

### Section 3.3: Training OlmoEarth

After setting up the model configuration file, fine-tune OlmoEarth on the processed windows:

```bash
python -m olmoearth_projects.main olmoearth_run finetune \
        --project_path $PROJECT_PATH --scratch_path $OER_DATASET_PATH
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

This section walks through the vizualization tools provided in the vizualization.ipynb notebook. This notebook is designed to be run end to end with minimal configuration by the user and can be found in the Visualization folder here https://github.com/rrichards7/uw-capstone-beetles/blob/main/Visualizations/visualizations.ipynb. The notebook walks through a series of visualizations for evaluating model predictions against ground‑truth labels using Sentinel‑2, DoveR, and SuperDove imagery. It begins with basic raster alignment and side‑by‑side visualization of the raw satellite image, labels, and predictions, then explores pixel‑intensity distributions and overlays contour comparisons. It also extracts polygon shapes from both masks to classify true positives, false positives, and false negatives, rendering them as colored outlines on the satellite image. Finally, it blends TP/FP/FN error maps directly onto multi‑sensor and time‑series imagery, including interactive widgets for switching sensors, spectral bands, and scrolling through monthly Sentinel‑2 scenes.

### Section 4.1: Data Curation/Configuration:
The first key step for visualization configuration is curating the ground truth raster, the predictions, and the sensor data to be used for the visuals. All three of these should be a result of the previous model training and inference, but for convenience there is sample data provided under the visualization folder within google drive. 

### Section 4.2: Notebook Configuration
To run this notebook on your own machine, you’ll need to update four separate sets of file paths. First, configure the standalone dataset paths that point to a single prediction raster, a single ground‑truth label, and a single Sentinel‑2 RGB image. Next, update the time‑series Sentinel‑2 directory, which contains month‑by‑month band‑group folders used by the interactive slider. Third, adjust the multi‑sensor directory that stores DoveR, Sentinel‑2, and SuperDove GeoTIFFs for the sensor‑switching visualizer. Make sure each directory mirrors the expected folder structure and contains the required band‑group files. The final set of filepaths is for a side by side visualization used on our poster which points to a particular task id of the superdove sensor data with ground truth and prediction rasters, this is not necessarily required to be unique, but is required to be defined, or change the variable names used to match pre defined predictions, ground truths, and sensor data.

## Section 5: Embedding Analysis

This section utilizes the high-dimensional latent space of the fine-tuned OlmoEarth model to quantify forest degradation and validate model sensitivity to bark beetle disturbances. All notebooks used for analysis can be found in the embeddings_analysis folder.


### Section 5.1: Spatiotemporal Event Alignment
We synchronize satellite imagery with ground-truth disturbance records to establish a "Healthy vs. Infested" benchmark. 
* **Workflow:** Extracts WGS84 bounding boxes from target GeoTIFFs to query the `defid2.sqlite` database. Informs when observations were made and when to generate embeddings. 

### Section 5.2: Latent Space Trajectories
We analyze the "drift" in embeddings over multiple years to detect the spectral and structural shifts characteristic of beetle-driven canopy decline.
* **Dimensionality Reduction:** Uses Global Average Pooling and PCA to project complex feature vectors into 2D trajectories, visualizing the transition of forest patches from healthy to infested states.
* **Distance Metrics:** Employs Cosine Distance to quantify the magnitude of shift between years. An increasing distance between the 2019 (Healthy) and 2021 (Infested) embeddings serves as a quantitative proxy for infestation severity.

### Section 5.3: Representation Quality & Interpretability
To ensure the fine-tuned model captures biological signals rather than sensor artifacts, we apply several diagnostic metrics:
* **SVD Energy Distribution:** Evaluates representation "organization" by measuring variance concentration in top singular values.
* **Change Kurtosis:** Analyzes the "peakedness" of yearly differences. High kurtosis indicates the model has learned to focus on specific reactive spectral bands (discoloration) rather than global image noise.
* **Model Divergence:** Measures the cosine distance between the Base OlmoEarth and Fine-Tuned representations to quantify the impact of domain-specific training.

---
