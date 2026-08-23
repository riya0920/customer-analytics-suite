"""Choosing k, and what happens when you stop choosing it -- HDBSCAN.

THE GAP
-------
"No HDBSCAN, no k selection procedure -- k=5 is asserted, and only stability and
forward-separation are measured, not optimality."

WHY 'OPTIMALITY' IS THE WRONG WORD, AND THE SECTION SAYS SO
------------------------------------------------------------
There is no optimal k. There is a k that maximises silhouette, a k that maximises
the Calinski-Harabasz ratio, a k the elbow suggests, a k that is stable under
resampling, and a k whose segments separate FUTURE behaviour. They disagree, and
they disagree for principled reasons -- silhouette rewards compact spheres,
stability rewards coarse partitions, and forward separation rewards whatever
correlates with the outcome.

So this module computes all five and reports the disagreement instead of
averaging it away. The one that should win is forward separation, because it is
the only criterion tied to what the segments are FOR; the others measure whether
the geometry is tidy, which is a question nobody in the business asked.

HDBSCAN IS A DIFFERENT ANSWER, NOT A BETTER ONE
------------------------------------------------
It does not take k. It finds however many dense regions exist and labels the rest
NOISE, which is the honest response to a customer base that genuinely has a
diffuse middle -- and it is also completely unusable for a campaign, because
"unassigned" is not a segment anyone can target. The noise share is the number
that decides whether it is applicable at all, so it is reported first.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)


def k_selection(X: np.ndarray, ks=(2, 3, 4, 5, 6, 7, 8), seed: int = 0,
                sample: int = 3000) -> list[dict]:
    """Five criteria per k. They will not agree, and that is the finding."""
    rng = np.random.default_rng(seed)
    idx = (rng.choice(len(X), sample, replace=False)
           if len(X) > sample else np.arange(len(X)))
    Xs = X[idx]
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        lab = km.labels_
        rows.append(dict(
            k=k,
            inertia=float(km.inertia_),
            silhouette=float(silhouette_score(Xs, lab[idx])),
            calinski_harabasz=float(calinski_harabasz_score(X, lab)),
            davies_bouldin=float(davies_bouldin_score(X, lab)),
            smallest_cluster_share=float(np.bincount(lab).min() / len(lab)),
        ))
    return rows


def elbow_k(rows: list[dict]) -> int:
    """The knee of the inertia curve, by maximum distance to the endpoint chord.

    Written out rather than eyeballed because "the elbow" is the criterion most
    often quoted and least often computed -- and once computed it is visibly a
    weak signal on a curve this smooth, which is worth seeing.
    """
    ks = np.array([r["k"] for r in rows], float)
    y = np.array([r["inertia"] for r in rows], float)
    p1, p2 = np.array([ks[0], y[0]]), np.array([ks[-1], y[-1]])
    d = p2 - p1
    d = d / np.linalg.norm(d)
    best, best_dist = int(ks[0]), -1.0
    for i in range(len(ks)):
        v = np.array([ks[i], y[i]]) - p1
        dist = float(np.linalg.norm(v - np.dot(v, d) * d))
        if dist > best_dist:
            best, best_dist = int(ks[i]), dist
    return best


def stability_k(X: np.ndarray, ks=(2, 3, 4, 5, 6, 7, 8), n_boot: int = 12,
                seed: int = 0) -> list[dict]:
    """Mean ARI between clusterings of bootstrap resamples, per k.

    Stability is biased toward SMALL k almost by construction: k=2 is stable on
    nearly any data because there is little to disagree about. That bias is the
    reason it is reported alongside the others rather than used alone.
    """
    rng = np.random.default_rng(seed)
    out = []
    for k in ks:
        base = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        scores = []
        for b in range(n_boot):
            idx = rng.choice(len(X), len(X), replace=True)
            km = KMeans(n_clusters=k, n_init=10, random_state=seed + b + 1).fit(X[idx])
            scores.append(adjusted_rand_score(base.labels_[idx], km.labels_))
        out.append(dict(k=k, mean_ari=float(np.mean(scores)),
                        min_ari=float(np.min(scores))))
    return out


def forward_separation(X: np.ndarray, future: np.ndarray,
                       ks=(2, 3, 4, 5, 6, 7, 8), seed: int = 0) -> list[dict]:
    """How much of the variance in FUTURE behaviour the segments explain.

    The only criterion here tied to what segments are for. Reported as eta^2 --
    between-group sum of squares over total -- which is bounded in [0,1] and
    rises mechanically with k, so the comparison that matters is against the
    ADJUSTED version below rather than the raw one.
    """
    out = []
    n = len(future)
    grand = float(np.mean(future))
    sst = float(np.sum((future - grand) ** 2))
    for k in ks:
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        ssb = 0.0
        for g in np.unique(lab):
            m = lab == g
            ssb += m.sum() * (float(np.mean(future[m])) - grand) ** 2
        eta2 = ssb / sst if sst > 0 else 0.0
        # Adjusted for the free parameters: eta^2 rises with k even on noise, so
        # an unadjusted table always recommends the largest k on offer.
        adj = 1 - (1 - eta2) * (n - 1) / max(n - k, 1)
        out.append(dict(k=k, eta_squared=float(eta2), adjusted_eta_squared=float(adj),
                        spread=float(np.max([np.mean(future[lab == g])
                                             for g in np.unique(lab)]) -
                                     np.min([np.mean(future[lab == g])
                                             for g in np.unique(lab)]))))
    return out


def hdbscan_segments(X: np.ndarray, min_cluster_size: int = 150) -> dict:
    """Density clustering: no k, and a NOISE label for whatever is not dense."""
    h = HDBSCAN(min_cluster_size=min_cluster_size)
    lab = h.fit_predict(X)
    n_noise = int((lab == -1).sum())
    clusters = [c for c in np.unique(lab) if c != -1]
    sizes = {int(c): int((lab == c).sum()) for c in clusters}
    return {"labels": lab, "n_clusters": len(clusters),
            "noise_share": n_noise / len(lab), "sizes": sizes}
