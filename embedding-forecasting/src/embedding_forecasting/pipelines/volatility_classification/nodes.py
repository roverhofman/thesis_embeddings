import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from kedro.framework.session import KedroSession
from sklearn.svm import SVC

# — PARAMETERS —
WINDOW_SIZE = 90
SKIP_IMAGES = {"gaf", "mtf", "rp"}

def create_raw_windows(series, window_size: int):
    arr = np.asarray(series, dtype=float)
    return [arr[i : i + window_size] for i in range(len(arr) - window_size + 1)]

def realized_volatility(windows):
    return np.array([np.std(np.diff(np.log(w))) for w in windows])

CLASSIFIERS = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM (RBF kernel)": SVC(kernel='rbf', probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
}

def weighted_f1(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    return f1_score(y_test, clf.predict(X_test), average="weighted")

def classification_node(train_prices, test_prices) -> dict:
    """
    Runs volatility-based thresholding and evaluates multiple classifiers
    across all embedding datasets in the catalog.
    """
    # load catalog
    session = KedroSession.create()
    catalog = session.load_context().catalog

    # discover embeddings via catalog
    train_ds = catalog.list(regex_search="_train$")
    prefixes = sorted(
        name[:-6]  # strip off "_train"
        for name in train_ds
        if name[:-6] not in SKIP_IMAGES
    )

    # 1) extract price arrays
    price_train = train_prices["price_close"].astype(float).values
    price_test  = test_prices["price_close"].astype(float).values

    # 2) compute realized vols & thresholds on train
    raw_wins_train = create_raw_windows(price_train, WINDOW_SIZE)
    vols_train     = realized_volatility(raw_wins_train)
    thresh_25      = np.percentile(vols_train, 75)
    low_thr, high_thr = np.percentile(vols_train, [33, 66])
    y_binary_train = (vols_train >= thresh_25).astype(int)
    y_multi_train  = np.digitize(vols_train, bins=[low_thr, high_thr], right=True)

    # 3) apply thresholds on test
    raw_wins_test  = create_raw_windows(price_test, WINDOW_SIZE)
    vols_test      = realized_volatility(raw_wins_test)
    y_binary_test  = (vols_test >= thresh_25).astype(int)
    y_multi_test   = np.digitize(vols_test, bins=[low_thr, high_thr], right=True)

    # 4) loop over embeddings → run both tasks
    results = {}
    for prefix in prefixes:
        X_tr = np.asarray(catalog.load(f"{prefix}_train"), dtype=float)
        X_te = np.asarray(catalog.load(f"{prefix}_test"),  dtype=float)

        # ensure 2D
        if X_tr.ndim == 1:
            X_tr = X_tr.reshape(-1, 1)
        if X_te.ndim == 1:
            X_te = X_te.reshape(-1, 1)

        binary_scores = {
            name: weighted_f1(clf, X_tr, y_binary_train, X_te, y_binary_test)
            for name, clf in CLASSIFIERS.items()
        }

        multi_scores = {}
        for name, clf in CLASSIFIERS.items():
            if name == "LogisticRegression":
                clf = LogisticRegression(
                    multi_class="multinomial", solver="lbfgs", max_iter=500
                )
            multi_scores[name] = weighted_f1(
                clf, X_tr, y_multi_train, X_te, y_multi_test
            )

        results[prefix] = {"binary": binary_scores, "multi": multi_scores}

    return results
