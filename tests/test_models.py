"""Unit tests for equity index feature extraction, PCA factors, and ML metrics."""

import sys
import os
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import create_target_and_features
from src.features import fit_pca_factors, cluster_equity_markets
from src.evaluation import evaluate_regression, compute_directional_accuracy


class TestEquityIndexMLPipeline(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n_obs = 150
        dates = pd.date_range('2020-01-01', periods=n_obs, freq='W')
        data = {f'Index_{i}': np.exp(np.cumsum(np.random.normal(0.001, 0.02, n_obs))) for i in range(5)}
        self.df = pd.DataFrame(data, index=dates)

    def test_create_target_and_features(self):
        X, y = create_target_and_features(self.df, target_col='Index_0', horizon=1)
        self.assertEqual(len(X), len(y))
        self.assertNotIn('Index_0', X.columns)
        self.assertEqual(len(X.columns), 4)

    def test_fit_pca_factors(self):
        rets = self.df.pct_change().dropna()
        pca, factors, loadings = fit_pca_factors(rets, n_components=3)
        self.assertEqual(factors.shape[1], 3)
        self.assertEqual(loadings.shape, (5, 3))
        self.assertAlmostEqual(np.sum(pca.explained_variance_ratio_), sum(pca.explained_variance_ratio_))

    def test_cluster_equity_markets(self):
        rets = self.df.pct_change().dropna()
        kmeans, r_df = cluster_equity_markets(rets, n_clusters=2)
        self.assertIn('Cluster', r_df.columns)
        self.assertEqual(len(r_df), 5)
        self.assertEqual(len(set(r_df['Cluster'])), 2)

    def test_evaluate_regression(self):
        y_true = np.array([0.01, -0.02, 0.015, 0.005])
        y_pred = np.array([0.008, -0.018, 0.012, 0.006])
        res = evaluate_regression(y_true, y_pred)
        self.assertIn('RMSE', res)
        self.assertIn('R2', res)
        self.assertGreater(res['R2'], 0.8)

    def test_compute_directional_accuracy(self):
        y_true = np.array([0.02, -0.01, 0.03, -0.02])
        y_pred = np.array([0.01, -0.04, 0.01, 0.01])  # 3 of 4 correct sign
        acc = compute_directional_accuracy(y_true, y_pred)
        self.assertEqual(acc, 0.75)


if __name__ == '__main__':
    unittest.main()
