# Financial Machine Learning

Supervised and unsupervised methods for weekly international equity-index returns, followed by model selection and ensemble comparisons.

## Research modules

- Regression, clustering, PCA, classification trees, discriminant analysis, SVM, and neural networks.
- Validation curves, hyperparameter search, bias?variance analysis, learning curves, bagging, random forests, and boosting.

## Data and execution

The supplied MSCI dataset is in `data/weekly_msci_indices.csv`. Install dependencies from `requirements.txt` and run notebooks from the repository root.

Some classification results use shuffled splits despite a future-return target. Re-evaluate chronologically before making claims about future performance.
