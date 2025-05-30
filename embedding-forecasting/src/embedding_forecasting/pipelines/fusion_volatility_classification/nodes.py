import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import itertools
from kedro.framework.session import KedroSession
from sklearn.svm import SVC

CLASSIFIERS = {
    # "RandomForest": lambda multiclass=False: RandomForestClassifier(n_estimators=100, random_state=42),
    # "GradientBoosting": lambda multiclass=False: GradientBoostingClassifier(n_estimators=100, random_state=42),
    # "SVM (RBF kernel)": lambda multiclass=False: SVC(kernel='rbf', probability=True, random_state=42),
    "K-Nearest Neighbors": lambda multiclass=False: KNeighborsClassifier(n_neighbors=5),
}

WINDOW_SIZE = 90
SKIP_IMAGES = {"gaf", "mtf", "rp"}

def create_raw_windows(series, window_size: int):
    arr = np.asarray(series, dtype=float)
    return [arr[i : i + window_size] for i in range(len(arr) - window_size + 1)]

def realized_volatility(windows):
    return np.array([np.std(np.diff(np.log(w))) for w in windows])

def weighted_f1(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    return f1_score(y_test, clf.predict(X_test), average="weighted")

def fusion_classification_node(train_prices, test_prices) -> dict:
    """
    Evaluate all 2-way combinations of embeddings via concatenation,
    on both binary and multi-class volatility thresholds.
    """
    # -- load catalog
    session = KedroSession.create()
    catalog = session.load_context().catalog

    # -- build vol labels (same as classification_node)
    price_tr = train_prices["price_close"].astype(float).values
    price_te = test_prices["price_close"].astype(float).values
    wins_tr  = create_raw_windows(price_tr, WINDOW_SIZE)
    wins_te  = create_raw_windows(price_te, WINDOW_SIZE)
    vols_tr  = realized_volatility(wins_tr)
    vols_te  = realized_volatility(wins_te)

    thr25    = np.percentile(vols_tr, 75)
    low33, high66 = np.percentile(vols_tr, [33, 66])
    y_bin_tr = (vols_tr >= thr25).astype(int)
    y_bin_te = (vols_te >= thr25).astype(int)
    y_mul_tr = np.digitize(vols_tr, bins=[low33, high66], right=True)
    y_mul_te = np.digitize(vols_te, bins=[low33, high66], right=True)

    # -- discover embeddings
    all_ds   = catalog.list()
    train_ds = [n for n in all_ds if n.endswith("_train")]
    prefixes = sorted({
        ds.rsplit("_", 1)[0]
        for ds in train_ds
        if ds.rsplit("_", 1)[0] not in SKIP_IMAGES
    })

    # -- evaluate combinations
    results = {"binary": [], "multi": []}
    for combo in itertools.combinations(prefixes, 2):
        # load and stack
        parts_tr, parts_te = [], []
        for p in combo:
            Xp_tr = np.asarray(catalog.load(f"{p}_train"), dtype=float)
            Xp_te = np.asarray(catalog.load(f"{p}_test"),  dtype=float)
            if Xp_tr.ndim == 1: Xp_tr = Xp_tr[:, None]
            if Xp_te.ndim == 1: Xp_te = Xp_te[:, None]
            parts_tr.append(Xp_tr)
            parts_te.append(Xp_te)
        X_tr = np.hstack(parts_tr)
        X_te = np.hstack(parts_te)

        # binary
        scores = [
            weighted_f1(make_clf(False), X_tr, y_bin_tr, X_te, y_bin_te)
            for make_clf in CLASSIFIERS.values()
        ]
        results["binary"].append((combo, float(np.mean(scores))))

        # multi
        scores = [
            weighted_f1(make_clf(True), X_tr, y_mul_tr, X_te, y_mul_te)
            for make_clf in CLASSIFIERS.values()
        ]
        results["multi"].append((combo, float(np.mean(scores))))

    # sort results (keep all combinations)
    for task in results:
        results[task].sort(key=lambda x: x[1], reverse=True)

    return results