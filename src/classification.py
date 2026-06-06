"""Classical classifiers + MLP + Autoencoder for churn prediction & anomaly detection.

Methodology guarantees
-----------------------
* The supervised frame is built only from pre-cutoff transactions (see
  ``preprocessing.make_classification_features``), so no future information
  reaches the features.
* Feature encoding (scaling + country one-hot) lives inside the pipeline via
  :func:`src.features.make_preprocessor`, so it is fit on the training folds
  only — ``GridSearchCV`` re-fits it per fold and on the final train split.
* SMOTE runs *after* the preprocessor and *only* at fit time (imblearn pipeline),
  never touching the held-out test set.

All metrics: AUC, F1, precision, recall, accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .features import NUMERIC_FEATURES, TARGET, make_preprocessor
from .utils import MODELS_DIR, SEED, get_logger, seed_all

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Classical model zoo + hyperparameter grids                                  #
# --------------------------------------------------------------------------- #

def build_model_grid(seed: int = SEED) -> Dict[str, Tuple[object, Dict]]:
    """Return {name: (estimator, param_grid_for_GridSearch)}."""
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, random_state=seed),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=seed),
            {"clf__max_depth": [None, 6, 12], "clf__min_samples_leaf": [1, 5]},
        ),
        "RandomForest": (
            RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
            {"clf__max_depth": [None, 10], "clf__min_samples_leaf": [1, 5]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [5, 11, 21]},
        ),
        "SVM_RBF": (
            SVC(kernel="rbf", probability=True, random_state=seed),
            {"clf__C": [1.0, 10.0], "clf__gamma": ["scale"]},
        ),
        "GaussianNB": (
            GaussianNB(),
            {},
        ),
    }


def make_pipeline(estimator, preprocessor: Optional[ColumnTransformer] = None,
                  seed: int = SEED) -> ImbPipeline:
    """Pipeline: preprocessor -> SMOTE -> classifier.

    The preprocessor and SMOTE only fire at ``fit`` time, so the held-out test
    set is transformed with statistics learned on the training data alone.
    """
    pre = preprocessor if preprocessor is not None else make_preprocessor()
    return ImbPipeline(steps=[
        ("pre", pre),
        ("smote", SMOTE(random_state=seed)),
        ("clf", estimator),
    ])


def _metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "AUC": float(roc_auc_score(y_true, y_proba)),
        "F1": float(f1_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _proba(estimator, X) -> np.ndarray:
    """Positive-class score, falling back to decision_function when needed."""
    try:
        return estimator.predict_proba(X)[:, 1]
    except (AttributeError, NotImplementedError):
        return estimator.decision_function(X)


@dataclass
class ModelOutcome:
    name: str
    best_params: dict
    cv_mean_auc: float
    cv_std_auc: float
    test_metrics: dict
    confusion: np.ndarray
    estimator: object  # fitted pipeline (preprocessor + smote + clf)


def train_classical(
    frame: pd.DataFrame,
    *,
    feature_columns: Optional[List[str]] = None,
    target_column: str = TARGET,
    test_size: float = 0.2,
    n_splits: int = 5,
    seed: int = SEED,
) -> Tuple[List[ModelOutcome], pd.DataFrame, dict, dict]:
    """Train all six classical models with CV + held-out test on a *raw* frame.

    Parameters
    ----------
    frame
        Output of :func:`src.features.build_supervised_frame`: raw feature
        columns (incl. ``DominantCountry``) plus the target. Encoding is done
        inside the pipeline, so pass raw values — do **not** pre-encode.

    Returns
    -------
    (outcomes, table, splits, metadata)
        ``splits`` carries both the raw train/test frames (for the fitted
        pipelines) and the transformed matrices + fitted preprocessor (for the
        MLP / SHAP). ``metadata`` records everything needed to reproduce the run.
    """
    seed_all(seed)
    feature_columns = feature_columns or [
        c for c in frame.columns if c not in (target_column, "CustomerID")
    ]
    X = frame[feature_columns].reset_index(drop=True)
    y = frame[target_column].astype(int).reset_index(drop=True)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    outcomes: List[ModelOutcome] = []
    rows = []

    for name, (est, grid) in build_model_grid(seed).items():
        pipe = make_pipeline(est, make_preprocessor(), seed=seed)
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
        gs.fit(X_tr, y_tr)
        best = gs.best_estimator_

        idx = gs.best_index_
        cv_mean = float(gs.cv_results_["mean_test_score"][idx])
        cv_std = float(gs.cv_results_["std_test_score"][idx])

        y_pred = best.predict(X_te)
        y_proba = _proba(best, X_te)
        m = _metrics(y_te, y_pred, y_proba)
        cm = confusion_matrix(y_te, y_pred)

        outcomes.append(ModelOutcome(name, gs.best_params_, cv_mean, cv_std, m, cm, best))
        rows.append({
            "Model": name,
            "CV_AUC_mean": round(cv_mean, 4),
            "CV_AUC_std": round(cv_std, 4),
            **{k: round(v, 4) for k, v in m.items()},
            "best_params": str(gs.best_params_),
        })
        logger.info("%-20s CV-AUC=%.4f  Test-AUC=%.4f", name, cv_mean, m["AUC"])

    table = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)

    # Fit a standalone preprocessor on the full training split for the MLP/SHAP.
    preprocessor = make_preprocessor().fit(X_tr)
    X_tr_t = preprocessor.transform(X_tr)
    X_te_t = preprocessor.transform(X_te)
    transformed_names = list(preprocessor.get_feature_names_out())

    splits = {
        "X_train_raw": X_tr, "X_test_raw": X_te,
        "X_train_t": np.asarray(X_tr_t), "X_test_t": np.asarray(X_te_t),
        "y_train": y_tr, "y_test": y_te,
        "preprocessor": preprocessor,
        "raw_feature_columns": feature_columns,
        "transformed_feature_names": transformed_names,
    }

    best_outcome = max(outcomes, key=lambda o: o.test_metrics["AUC"])
    metadata = {
        "feature_columns": feature_columns,
        "numeric_features": NUMERIC_FEATURES,
        "target_column": target_column,
        "seed": seed,
        "test_size": test_size,
        "n_splits": n_splits,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "churn_rate": float(y.mean()),
        "best_model_name": best_outcome.name,
        "best_model_auc": float(best_outcome.test_metrics["AUC"]),
        "transformed_feature_names": transformed_names,
    }
    return outcomes, table, splits, metadata


def save_models(outcomes: List[ModelOutcome], folder: Path = MODELS_DIR) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for o in outcomes:
        joblib.dump(o.estimator, folder / f"{o.name}.joblib")
        logger.info("saved %s.joblib", o.name)


# --------------------------------------------------------------------------- #
# MLP (PyTorch)                                                                #
# --------------------------------------------------------------------------- #

def build_mlp(input_dim: int):
    """Return a small MLP for tabular churn classification."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 1),
    )


def train_mlp(
    X_train, y_train,
    X_val, y_val,
    epochs: int = 60, batch_size: int = 256, lr: float = 1e-3, patience: int = 8,
    seed: int = SEED,
) -> Tuple[object, list, list, StandardScaler]:
    """Train MLP with SMOTE + early stopping on val AUC.

    ``X_train``/``X_val`` should already be the *transformed* numeric matrices
    (e.g. ``splits['X_train_t']``). The function is self-seeding: it calls
    :func:`seed_all` and uses a seeded ``DataLoader`` generator, so it no longer
    depends on the notebook having seeded earlier.

    Returns ``(model, train_losses, val_aucs, scaler)``.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train).ravel()
    y_val = np.asarray(y_val).ravel()

    # Scale (idempotent if already standardized) + SMOTE on train only
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_train_s, y_train_arr = SMOTE(random_state=seed).fit_resample(X_train_s, y_train)

    train_ds = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(y_train_arr.astype(np.float32)).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_s, dtype=torch.float32),
        torch.tensor(y_val.astype(np.float32)).unsqueeze(1),
    )
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_mlp(X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    train_losses, val_aucs = [], []
    best_auc, best_state, since = 0.0, None, 0

    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward(); opt.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_ds)
        train_losses.append(epoch_loss)

        model.eval()
        probs, ys = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                probs.append(torch.sigmoid(model(xb)).cpu().numpy())
                ys.append(yb.numpy())
        probs = np.concatenate(probs).ravel()
        ys = np.concatenate(ys).ravel()
        auc = roc_auc_score(ys, probs)
        val_aucs.append(auc)

        if auc > best_auc:
            best_auc, since = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                logger.info("early stopping at epoch %d (best val AUC=%.4f)", ep, best_auc)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_aucs, scaler


def mlp_metadata(metadata: dict, *, threshold: float = 0.5, scaler: StandardScaler | None = None,
                 seed: int = SEED) -> dict:
    """Build a JSON-serialisable metadata dict to ship alongside a saved MLP."""
    meta = {
        "input_features": metadata.get("transformed_feature_names"),
        "seed": seed,
        "threshold": threshold,
        "churn_cutoff": "2011-09-01",
    }
    if scaler is not None:
        meta["scaler_mean"] = np.asarray(scaler.mean_).tolist()
        meta["scaler_scale"] = np.asarray(scaler.scale_).tolist()
    return meta


def mlp_proba_fn(model, scaler: StandardScaler):
    """Return ``f(X_2d) -> proba_1d`` for SHAP's model-agnostic KernelExplainer.

    Wraps the trained torch MLP + its scaler into a plain NumPy callable, so the
    same transformed feature space used for training is explained.
    """
    import torch

    device = next(model.parameters()).device

    def _f(X: np.ndarray) -> np.ndarray:
        Xs = scaler.transform(np.asarray(X, dtype=np.float32))
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(Xs, dtype=torch.float32, device=device))
            return torch.sigmoid(logits).cpu().numpy().ravel()

    return _f


def mlp_evaluate(model, scaler: StandardScaler, X, y, threshold: float = 0.5) -> dict:
    import torch
    device = next(model.parameters()).device
    model.eval()
    Xs = scaler.transform(np.asarray(X, dtype=np.float32))
    with torch.no_grad():
        logits = model(torch.tensor(Xs, dtype=torch.float32, device=device))
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
    pred = (probs >= threshold).astype(int)
    y = np.asarray(y).ravel()
    m = _metrics(y, pred, probs)
    cm = confusion_matrix(y, pred)
    return {"metrics": m, "confusion": cm, "probs": probs, "pred": pred}


# --------------------------------------------------------------------------- #
# Autoencoder for anomalous-customer detection                                #
# --------------------------------------------------------------------------- #

def build_autoencoder(input_dim: int, bottleneck: int = 4):
    import torch.nn as nn
    encoder = nn.Sequential(
        nn.Linear(input_dim, 16), nn.ReLU(),
        nn.Linear(16, 8), nn.ReLU(),
        nn.Linear(8, bottleneck),
    )
    decoder = nn.Sequential(
        nn.Linear(bottleneck, 8), nn.ReLU(),
        nn.Linear(8, 16), nn.ReLU(),
        nn.Linear(16, input_dim),
    )
    return nn.Sequential(encoder, decoder)


def train_autoencoder(
    X, epochs: int = 80, batch_size: int = 256, lr: float = 1e-3, seed: int = SEED,
) -> Tuple[object, list, StandardScaler, np.ndarray]:
    """Train autoencoder on (scaled) features. Returns model, loss curve, scaler,
    per-sample reconstruction error.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = np.asarray(X, dtype=np.float32)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    ds = TensorDataset(torch.tensor(Xs, dtype=torch.float32))
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)

    model = build_autoencoder(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    losses = []
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            xr = model(xb)
            loss = loss_fn(xr, xb)
            loss.backward(); opt.step()
            ep_loss += loss.item() * xb.size(0)
        losses.append(ep_loss / len(ds))

    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(Xs, dtype=torch.float32, device=device)
        recon = model(Xt).cpu().numpy()
    err = ((Xs - recon) ** 2).mean(axis=1)
    return model, losses, scaler, err
