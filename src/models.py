"""Linear, penalized, and ensemble modeling architectures."""

from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier


def fit_lasso_regression(X_train, y_train, cv=5):
    """Fit LASSO regression with cross-validated alpha regularization."""
    model = LassoCV(cv=cv, random_state=42)
    model.fit(X_train, y_train)
    return model


def fit_random_forest_ensemble(X_train, y_train, n_estimators=100, max_depth=5):
    """Fit Random Forest regressor ensemble to international return features."""
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)
    return rf


def fit_gradient_boosting_ensemble(X_train, y_train, n_estimators=100, learning_rate=0.05):
    """Fit Gradient Boosting ensemble minimizing residual forecasting bias."""
    gb = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    gb.fit(X_train, y_train)
    return gb
