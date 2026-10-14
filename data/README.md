# Data Documentation: Equity Index ML Ensembles

## Directory Structure
- `raw/`: Raw international index histories and macro features.
- `processed/`: Normalized feature matrices, PCA component loadings, and label vectors.
- Root data file: `weekly_msci_indices.csv` (contains weekly index levels across 35 international equity markets).

## Data Sources & Acquisition
- **Primary Dataset**: `weekly_msci_indices.csv` provides weekly closing values for 35 MSCI global equity market indices across developed, emerging, and frontier markets.
- **Feature Engineering & Preprocessing**:
  1. Weekly percentage changes and log returns are calculated.
  2. Target variable: 1-week forward return of target index (e.g., LUXXX).
  3. Feature standardization via `StandardScaler` and `MinMaxScaler`.
  4. Unsupervised clustering using k-Means to group co-moving international markets.
  5. PCA decomposition to extract global systematic equity factors (`src/features.py`).
