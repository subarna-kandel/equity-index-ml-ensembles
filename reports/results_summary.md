# Research & Methodology Summary: Equity Index ML Ensembles

## 1. Executive Overview
This study evaluates supervised, unsupervised, and ensemble machine learning techniques across 35 international MSCI equity indices. We focus on dimensionality reduction, market co-movement clustering, and predicting one-week-ahead index return direction.

## 2. Experimental Framework
- **Data Universe**: Weekly closing prices across 35 developed, emerging, and frontier markets.
- **Dimensionality Reduction**: Principal Component Analysis (PCA) decomposes return covariance into orthogonal factor exposures.
- **Unsupervised Segmentation**: k-Means clustering isolates market groupings by risk-return profiles.
- **Predictive Modeling**:
  - Regularized Linear Models (LASSO, Ridge) with cross-validated shrinkage.
  - Non-linear classifiers: Decision Trees, Linear Discriminant Analysis (LDA), and Support Vector Machines (SVM).
  - Ensembles: Bootstrap Aggregation (Bagging), Random Forests, and Gradient Boosting.

## 3. Empirical Findings
- **Factor Concentration**: The first 3 principal components account for >70% of total cross-market variance, capturing global market beta, regional divergence, and emerging market risk premia.
- **Ensemble Dominance**: Gradient Boosted trees and Random Forests consistently outperformed unconstrained linear regression and individual decision trees by curbing prediction variance.
- **Bias-Variance Tradeoff**: Learning curve analysis confirmed that regularized ensemble methods mitigated overfitting on small sample time-series data.

## 4. Limitations
- Macro-regime instability alters cross-asset correlations over time, requiring adaptive retraining.
