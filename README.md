# Image Retrieval with Bag of Visual Words

A content-based image retrieval (CBIR) system that finds visually similar images in the **Paris6k** landmark dataset. Images are described with **SIFT** local features, quantized into a **Bag of Visual Words (BoVW)** vocabulary using **K-Means**, weighted with **TF-IDF**, and ranked by **cosine similarity**. Retrieval quality is measured with Precision@K, Recall@K, and Average Precision.

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Report](#report)

## Overview

Given a query image of a Paris landmark, the system returns the top-K most similar images from the database. Each result is labeled with its landmark class and shown next to the query, with correct matches highlighted in green and incorrect matches in red.

## Pipeline

| Stage | Script | Description |
|---|---|---|
| 1. Data check | `prepare_data.py` | Recursively scans the dataset and verifies images can be read |
| 2. Feature extraction | `extract_features.py` | Extracts 128-D SIFT descriptors from each grayscale image |
| 3. Vocabulary | `build_codebook.py` | Samples up to 100,000 descriptors and clusters them with K-Means (K = 500) |
| 4. Encoding | `build_histograms.py` | Maps each image's descriptors to visual words and builds a normalized 500-bin histogram |
| 5. Weighting | `compute_idf.py` | Computes inverse document frequency: `idf = log((N + 1) / (df + 1))` |
| 6. Retrieval | `retrieve.py` | Encodes the query, applies TF-IDF with L2 normalization, ranks by cosine similarity, evaluates, and visualizes |

`search.py` is a simpler baseline that ranks by cosine similarity on raw BoVW histograms without TF-IDF. `visualize_results.py` provides a standalone result-grid plot.

## Dataset

The project uses the **Paris6k** dataset (Oxford Visual Geometry Group), which contains about 6,400 images across 12 landmark folders:

`defense`, `eiffel`, `general`, `invalides`, `louvre`, `moulinrouge`, `museedorsay`, `notredame`, `pantheon`, `pompidou`, `sacrecoeur`, `triomphe`

Image files follow the pattern `paris_<landmark>_<id>.jpg`, and the landmark name is used as the ground-truth label.

The images are not included in this repository because of their size. Download them from:

https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/

Then extract them into this layout:

```
data/images/visionset/Paris6k/
├── defense/
├── eiffel/
├── ...
└── triomphe/
```

Place query images in `data/queries/`, named with the same `paris_<landmark>_<id>.jpg` pattern so they can be evaluated.

## Installation

**Requirements:** Python 3.8 or later.

```bash
git clone https://github.com/hala-sabobeh/image-retrieval-bovw.git
cd image-retrieval-bovw
pip install opencv-python numpy scikit-learn joblib matplotlib
```

SIFT is included in `opencv-python` 4.4 and later.

## Usage

The first three indexing scripts use paths relative to the project root, while the remaining scripts use paths relative to `src/`. Run each one from the directory shown.

**Build the index** (from the project root):

```bash
python src/prepare_data.py
python src/extract_features.py
python src/build_codebook.py
python src/build_histograms.py
```

**Compute IDF and run retrieval** (from `src/`):

```bash
cd src
python compute_idf.py
python retrieve.py
```

`retrieve.py` prompts for:

1. The query image file name, for example `paris_eiffel_000123.jpg`
2. The number of top matches to return

It prints the ranked results with similarity scores, the evaluation metrics, and the retrieval time, then displays the query alongside the retrieved images.

## Evaluation

A retrieved image is relevant if it belongs to the same landmark class as the query.

| Metric | Definition |
|---|---|
| Precision@K | Fraction of the top K results that are relevant |
| Recall@K | Relevant images in the top K divided by all relevant images in the database |
| Average Precision (AP) | Mean of the precision values at each rank where a relevant image appears |
| Retrieval Time | Time to encode the query and rank the database |

## Project Structure

```
.
├── data/
│   ├── images/visionset/Paris6k/   # Dataset (download separately)
│   └── queries/                    # Query images
├── features/
│   ├── descriptors/                # SIFT descriptors per image (generated)
│   ├── histograms/                 # BoVW histograms per image (generated)
│   └── models/
│       ├── kmeans_model.pkl        # Trained K-Means vocabulary
│       ├── codebook.npy            # Cluster centers (500 x 128)
│       └── idf.npy                 # IDF weights
├── src/
│   ├── prepare_data.py
│   ├── extract_features.py
│   ├── build_codebook.py
│   ├── build_histograms.py
│   ├── compute_idf.py
│   ├── retrieve.py
│   ├── search.py
│   └── visualize_results.py
└── vision_report.pdf
```

## Configuration

| Parameter | Location | Default |
|---|---|---|
| Vocabulary size (K) | `build_codebook.py` | 500 |
| Max descriptors for K-Means | `build_codebook.py` | 100,000 |
| K-Means random state | `build_codebook.py` | 42 |
| Top-K results | prompted in `retrieve.py` | user-defined |

## Report

See `vision_report.pdf` for the full methodology, experiments, and analysis of results.

## Author

**Hala Sabobeh**
