# <Project Name>

Linear regression model to predict house rents. Preprocessing pipeline with scaling + one-hot encoding. Implemented as coursework at Imperial, cleaned and published here with demo data.

## Quickstart

```bash
pip install -r requirements.txt

# Train on your CSV (example column names)
#   - target column: price
#   - numeric: area bedrooms
#   - categorical: area_rate
PYTHONPATH=src python -m src.train --csv data/train.csv --target price --numeric area bedrooms --categorical area_rate --min-count-col area --min-count 6
