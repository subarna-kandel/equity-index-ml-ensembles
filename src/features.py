"""Unsupervised clustering, PCA factor decomposition, and standardization."""

import numpy as np
import pandas as pd


def fit_pca_factors(returns_df, n_components=3):
    """Decompose cross-sectional equity returns into orthogonal principal component factors."""
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        scaler = StandardScaler()
        scaled_rets = scaler.fit_transform(returns_df)
        pca = PCA(n_components=n_components)
        factors = pca.fit_transform(scaled_rets)
        loadings = pd.DataFrame(pca.components_.T, index=returns_df.columns,
                                columns=[f'PC{i+1}' for i in range(n_components)])
        return pca, factors, loadings
    except ImportError:
        # NumPy SVD fallback
        vals = returns_df.values
        normed = (vals - np.mean(vals, axis=0)) / (np.std(vals, axis=0) + 1e-10)
        U, S, Vt = np.linalg.svd(normed, full_matrices=False)
        factors = U[:, :n_components] * S[:n_components]
        loadings = pd.DataFrame(Vt[:n_components].T, index=returns_df.columns,
                                columns=[f'PC{i+1}' for i in range(n_components)])
        class MockPCA:
            explained_variance_ratio_ = (S[:n_components]**2) / np.sum(S**2)
        return MockPCA(), factors, loadings


def cluster_equity_markets(returns_df, n_clusters=4, random_state=42):
    """Group international markets based on risk-return profile using k-Means."""
    ann_return = returns_df.mean() * 52.0
    ann_vol = returns_df.std() * np.sqrt(52.0)
    risk_return_df = pd.DataFrame({'AnnualizedReturn': ann_return, 'AnnualizedVol': ann_vol})

    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        scaler = StandardScaler()
        scaled_feats = scaler.fit_transform(risk_return_df)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        risk_return_df['Cluster'] = kmeans.fit_predict(scaled_feats)
        return kmeans, risk_return_df
    except ImportError:
        # Quantile-based deterministic grouping fallback
        risk_return_df['Cluster'] = pd.qcut(risk_return_df['AnnualizedReturn'], q=n_clusters, labels=False)
        return None, risk_return_df
