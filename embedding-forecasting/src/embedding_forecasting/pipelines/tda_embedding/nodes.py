import numpy as np
import pandas as pd
import ts2vg
import networkx as nx
from scipy.stats import kurtosis, skew
from gudhi import CubicalComplex
from gudhi.wasserstein import wasserstein_distance
from gudhi.hera import bottleneck_distance
from gudhi.representations import Landscape


def compute_weight_statistics_tda(weights):
    statistics = {}
    weights_array = np.array(weights)
    weights_array_nonzero = weights_array[weights_array > 0]
    weights_sum = np.sum(weights_array_nonzero)
    probabilities = weights_array_nonzero / weights_sum if weights_sum > 0 else np.zeros_like(weights_array_nonzero)
    statistics['entropy'] = -np.sum(probabilities * np.log2(probabilities + 1e-10))

    statistics['variance'] = np.var(weights_array)
    statistics['standard_deviation'] = np.std(weights_array)
    statistics['mean'] = np.mean(weights_array)
    statistics['median'] = np.median(weights_array)
    statistics['5th_percentile'] = np.percentile(weights_array, 5)
    statistics['25th_percentile'] = np.percentile(weights_array, 25)
    statistics['75th_percentile'] = np.percentile(weights_array, 75)
    statistics['95th_percentile'] = np.percentile(weights_array, 95)
    statistics['RMS'] = np.sqrt(np.mean(weights_array**2))
    statistics['kurtosis'] = kurtosis(weights_array)
    statistics['skewness'] = skew(weights_array)

    # Calculate zero crossings
    zero_crossings = np.where(np.diff(np.sign(weights_array)))[0]
    statistics['zero_crossings'] = len(zero_crossings)

    # Calculate mean crossings
    mean_value = np.mean(weights_array)
    mean_crossings = np.where(np.diff(np.sign(weights_array - mean_value)))[0]
    statistics['mean_crossings'] = len(mean_crossings)

    return statistics


def compute_graph_metrics_tda(G: nx.Graph) -> dict:
    """
    Compute graph metrics for a given NetworkX graph using TDA weight stats.
    """
    # basic degree stats
    degs = [d for _, d in G.degree()]
    metrics = {
        'average_degree': np.mean(degs),
        'density': nx.density(G),
        'transitivity': nx.transitivity(G),
        'degree_assortativity': nx.degree_assortativity_coefficient(G)
    }

    # weight-based statistics
    weights = [data['weight'] for _, _, data in G.edges(data=True)]
    metrics.update(compute_weight_statistics_tda(weights))
    return metrics


def feature_vector_tda(time_series):
    """Given a 1D array, compute TDA feature vector."""
    cc = CubicalComplex(dimensions=[len(time_series)], top_dimensional_cells=list(time_series))
    cc.compute_persistence()
    pd0 = np.array(cc.persistence_intervals_in_dimension(0))
    pd0 = pd0[pd0[:,1] != np.inf]
    if pd0.size == 0:
        # return zero vector if no persistence
        return [0.] * 7

    maxd = max(pd0[:,1].max(), time_series.max())
    null = np.vstack([np.linspace(0, maxd*1.1, len(pd0))]*2).T

    w1 = wasserstein_distance(pd0, null, order=1, keep_essential_parts=False)
    bnd = bottleneck_distance(pd0, null)

    lifetimes = pd0[:,1] - pd0[:,0]
    p = lifetimes / lifetimes.sum()
    entropy = -np.sum(p * np.log(p + 1e-10))

    ts_len = len(time_series)
    times = np.arange(ts_len)
    betti = ((pd0[:,0][None,:] <= times[:,None]) &
             (times[:,None] < pd0[:,1][None,:])).sum(axis=1)
    b1 = betti.sum()
    b2 = np.sqrt((betti**2).sum())

    lc = Landscape(resolution=1000)
    lc.fit([pd0])
    L = lc.transform([pd0])[0]
    l1 = np.abs(L).sum()
    l2 = np.sqrt((L**2).sum())

    return [w1, bnd, entropy, b1, b2, l1, l2]


def features_tda(df_list):
    """Apply feature_vector_tda to each time series in each DataFrame"""
    return [feature_vector_tda(row) for df in df_list for row in df.values]


def create_vg_tda(ts, graph_type='natural'):
    vg = getattr(ts2vg, f"{graph_type.capitalize()}VG")(weighted="abs_angle")
    vg.build(ts)
    G = vg.as_networkx()
    return np.array(sorted(dict(G.degree()).values()))


def features_vg_tda(data, graph_type='natural'):
    feats = [feature_vector_tda(create_vg_tda(row, graph_type)) for row in data.values]
    return pd.DataFrame(feats)


def create_vg2_tda(data, graph_type='natural') -> pd.DataFrame:
    """Compute graph metrics (using compute_graph_metrics_tda) for each series."""
    all_metrics = []
    for row in data.values:
        vg = getattr(ts2vg, f"{graph_type.capitalize()}VG")(weighted="abs_angle")
        vg.build(row)
        G = vg.as_networkx()
        metrics = compute_graph_metrics_tda(G)
        all_metrics.append(metrics)
    return pd.DataFrame(all_metrics)


def merging_tda(*dfs):
    return pd.concat(dfs, axis=1, ignore_index=True)


def tda_embedding(train_sc, valid_sc, test_sc):
    # raw TDA
    tr1 = pd.DataFrame(features_tda([train_sc]))
    vl1 = pd.DataFrame(features_tda([valid_sc]))
    te1 = pd.DataFrame(features_tda([test_sc]))

    # VG + TDA (natural)
    tr2  = features_vg_tda(train_sc, 'natural')
    tr2b = create_vg2_tda(train_sc, 'natural')
    vl2  = features_vg_tda(valid_sc, 'natural')
    vl2b = create_vg2_tda(valid_sc, 'natural')
    te2  = features_vg_tda(test_sc, 'natural')
    te2b = create_vg2_tda(test_sc, 'natural')

    # VG + TDA (horizontal)
    tr3  = features_vg_tda(train_sc, 'horizontal')
    tr3b = create_vg2_tda(train_sc, 'horizontal')
    vl3  = features_vg_tda(valid_sc, 'horizontal')
    vl3b = create_vg2_tda(valid_sc, 'horizontal')
    te3  = features_vg_tda(test_sc, 'horizontal')
    te3b = create_vg2_tda(test_sc, 'horizontal')

    train_tda = merging_tda(tr1, tr2, tr2b, tr3, tr3b)
    valid_tda = merging_tda(vl1, vl2, vl2b, vl3, vl3b)
    test_tda  = merging_tda(te1, te2, te2b, te3, te3b)

    return train_tda, valid_tda, test_tda
