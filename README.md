# Detecting Bark Beetle Outbreaks through Geospatial Foundation Models
University of Washington MSDS Capstone Project 2026

## Section 1: Environment and Repo Setup

```bash
conda
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

#### Section 3.1.2: Model Config

### Section 3.2: Building a Dataset

```bash
rslearn prepare
```

```bash
rslearn build
```

### Section 3.3: Training OlmoEarth

```bash
model fit
```

### Section 3.4: Testing
or rslearn model validate

```bash
rslearn model test
```

### Section 3.5: Output Prediction Generation

Update model config

```bash
rslearn model test
```

### Section 3.6: Embedding Generation

Update model config

```bash
rslearn model test
```

## Section 4: Output Prediction Visualization


## Section 5: Embedding Analysis
