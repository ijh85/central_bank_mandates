# Replication Code for Bertsch et al. (2024).

This repository contains replication code for Bertsch et al. (2024).

## Overview

This code reproduces the following results:
- Asset return regressions
- Text regressions
- Time series plots of text features
- Word cloud visualizations of speaker concerns
- Shapley value plots of variable importance

## Requirements

### Data
The following data is located in the `data/` directory:
- `data_60_21.csv`: Full sample (1960-2021)
- `data_60_83.csv`: Pre-great moderation sample (1960-1983)
- `data_84_21.csv`: Great moderation and post-crisis sample (1984-2021)
- `speech_counts.csv`: Annual frequency speech data
- `aggregate_data.csv`: Aggregated data
- `text.csv`: Text features
- `labels.csv`: District labels

## Code Structure

### `scripts/plots.py`
Generates visualizations including:
- Speech frequency histograms
- Time series plots of key metrics
- Word clouds of financial stability concerns
- SHAP value variable importance plots

### `scripts/regressions.py`
Runs regression analyses including:
- Text regressions
- Asset return regressions

## Output

Results are exported to following directories:
- `results/plots/`: Generated figures in vector format
- `results/tables/`: Descriptive statistics in CSV format
- `results/regressions/`: Regression tables in tex format

## Usage

1. Install required Python packages:

```bash
pip install -r requirements.txt
```

2. Put input data in `data/` directory.

3. Run scripts:

```
bash
python scripts/plots.py
python scripts/regressions.py
```

