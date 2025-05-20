# nodes.py
import pandas as pd
import numpy as np
import networkx as nx
import ts2vg
from networkx.algorithms import community
from scipy.stats import kurtosis, skew


def compute_weight_statistics(weights):
    statistics = {}
    w = np.array(weights)
    w_nonzero = w[w > 0]
    total = w_nonzero.sum()
    probs = w_nonzero / total
    statistics['entropy'] = -np.sum(probs * np.log2(probs))

    statistics['variance'] = np.var(w)
    statistics['standard_deviation'] = np.std(w)
    statistics['mean'] = np.mean(w)
    statistics['median'] = np.median(w)
    for p in (5, 25, 75, 95):
        statistics[f'{p}th_percentile'] = np.percentile(w, p)
    statistics['RMS'] = np.sqrt(np.mean(w**2))
    statistics['kurtosis'] = kurtosis(w)
    statistics['skewness'] = skew(w)

    # zero and mean crossings
    zc = np.where(np.diff(np.sign(w)))[0]
    mc = np.where(np.diff(np.sign(w - statistics['mean'])))[0]
    statistics['zero_crossings'] = len(zc)
    statistics['mean_crossings'] = len(mc)

    return statistics


def compute_graph_metrics(G: nx.Graph) -> dict:
    metrics = {}
    degs = [d for _, d in G.degree()]
    metrics['average_degree'] = np.mean(degs)

    if nx.is_connected(G):
        metrics['average_shortest_path_length'] = nx.average_shortest_path_length(G)
        metrics['diameter'] = nx.diameter(G)
    else:
        metrics['average_shortest_path_length'] = float('inf')
        metrics['diameter'] = float('inf')

    metrics['global_efficiency'] = nx.global_efficiency(G)
    metrics['average_clustering_coefficient'] = nx.average_clustering(G)
    metrics['degree_assortativity_coefficient'] = nx.degree_assortativity_coefficient(G)
    metrics['density'] = nx.density(G)
    metrics['transitivity'] = nx.transitivity(G)

    # community/modularity
    parts = community.greedy_modularity_communities(G)
    metrics['modularity'] = community.modularity(G, parts)

    # edge-weight stats
    weights = [data['weight'] for _, _, data in G.edges(data=True)]
    metrics.update(compute_weight_statistics(weights))

    return metrics


def create_vg(data):
    """
    Build a visibility graph for each time-series in `data`
    and compute its graph metrics.
    """
    # If `data` is a DataFrame, convert to NumPy so we iterate rows:
    arr = data.values if hasattr(data, "values") else data

    all_metrics = []
    for series in arr:
        vg = ts2vg.NaturalVG(weighted="abs_angle")
        vg.build(series)
        G = vg.as_networkx()
        all_metrics.append(compute_graph_metrics(G))

    return pd.DataFrame(all_metrics)

def graph_embedding(train_sc, valid_sc, test_sc):
    """
    Apply visibility-graph embedding to train/validation/test splits.
    Returns three DataFrames of graph metrics.
    """
    train_gr = create_vg(train_sc)
    val_gr = create_vg(valid_sc)
    test_gr = create_vg(test_sc)
    return train_gr, val_gr, test_gr
