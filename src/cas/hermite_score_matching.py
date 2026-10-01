"""Stage-1 Hermite copula score matching and the regularized solves.

Builds the multi-index set, the score-matching system (A, b), the regularized
solves (solve_ridge_theoretical, solve_ridge_hermitescaled and alternatives)
and the score second-moment eigenbasis (Part I, Section 3).
"""

import numpy as np
from scipy import stats
from scipy.linalg import cho_factor, cho_solve, qr, eigh
from itertools import combinations, product
import math


# ---------------------------------------------------------------
# Hermite polynomials: normalized probabilists'
# ---------------------------------------------------------------

def hermite_norm(z, max_degree):
    """
    Evaluate normalized probabilists' Hermite polynomials h_0, ..., h_{max_degree}
    at points z (array of any shape). Returns array with trailing axis of size
    max_degree + 1 indexed by degree.

    Recurrence (unnormalized He): He_0=1, He_1=z, He_{n+1} = z He_n - n He_{n-1}.
    Then h_n = He_n / sqrt(n!).
    """
    z = np.asarray(z, dtype=float)
    out = np.empty(z.shape + (max_degree + 1,), dtype=float)
    out[..., 0] = 1.0
    if max_degree >= 1:
        out[..., 1] = z
        # Unnormalized, then normalize at the end
        for n in range(1, max_degree):
            out[..., n + 1] = z * out[..., n] - n * out[..., n - 1]
    # Normalize: divide by sqrt(n!)
    norms = np.array([math.sqrt(math.factorial(n)) for n in range(max_degree + 1)])
    return out / norms


# ---------------------------------------------------------------
# Dictionary enumeration
# ---------------------------------------------------------------

def enumerate_dictionary(d, K, q):
    """
    Enumerate multi-indices alpha in N_0^d with:
      - 1 <= |alpha| <= K  (total degree cap)
      - 1 <= |alpha|_0 <= q  (interaction-order cap: number of nonzero entries)
      - exclude pure-axis degree-1 indices alpha = e_i (absorbed by -z baseline)

    Returns list of alpha as tuples. Stored sparsely: (support_indices, support_degrees).
    """
    A = []
    # Iterate over number of active coordinates m = 1, ..., q
    for m in range(1, q + 1):
        # Iterate over which coordinate subset of size m
        for coord_subset in combinations(range(d), m):
            # Iterate over degree tuples on these m coordinates with sum in [1, K],
            # each component >= 1 (since the coord is in the support)
            # and excluding (m=1, deg=1) which is pure-axis linear
            for total in range(1, K + 1):
                if total < m:
                    continue  # can't have m positive entries summing to < m
                # Enumerate compositions of `total` into m parts, each >= 1
                for deg_tuple in _compositions(total, m):
                    # Exclude pure-axis linear: m=1 and deg_tuple=(1,)
                    if m == 1 and deg_tuple == (1,):
                        continue
                    A.append((coord_subset, deg_tuple))
    return A


def _compositions(total, parts):
    """Yield all compositions of `total` into `parts` positive integers."""
    if parts == 1:
        yield (total,)
        return
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


# Also: "extended" dictionary including the pure-axis linear e_i terms.
# These are NEVER learned (absorbed by -z baseline) but their Hermite evaluations
# may appear as shifts of degree-2 features. We need to evaluate them but not
# include them as columns in A, b.
#
# Actually: we never need them as entries in theta. What we need is: for each
# alpha in A, evaluate sqrt(alpha_i) H_{alpha - e_i} for each i in support.
# If alpha - e_i has a pure-axis support of size 1 and remaining degree d, that's
# fine - we just evaluate h_d at that coordinate and 1 elsewhere. Nothing special.


# ---------------------------------------------------------------
# Feature matrices (key subroutine)
# ---------------------------------------------------------------


def _build_support_groups(A):
    """Group dictionary entries by their support pattern. Returns a list of
    (supp, deg_arr, idx_arr) where idx_arr indexes into A. Used by the dense
    per-axis Phi builder to batch the elementwise products.
    """
    groups = {}
    for a, (supp, deg) in enumerate(A):
        groups.setdefault(supp, ([], []))
        groups[supp][0].append(deg)
        groups[supp][1].append(a)
    out = []
    for supp, (degs, idxs) in groups.items():
        deg_arr = np.array(degs, dtype=np.int64)
        idx_arr = np.array(idxs, dtype=np.int64)
        out.append((supp, deg_arr, idx_arr))
    return out


def _build_axis_meta(A, d):
    """For each axis i in [0, d), pre-compute:
       - local_idxs[i]: int array of dictionary indices touching axis i (sorted)
       - groups_for_axis[i]: list of (supp, k_in_supp, deg_arr, local_pos)
         where local_pos[m] is the column position in the dense (N, |S_i|)
         block for that group entry.
    """
    from collections import defaultdict
    groups_global = _build_support_groups(A)
    raw_local = [[] for _ in range(d)]
    for (supp, deg_arr, idx_arr) in groups_global:
        for k, axis_i in enumerate(supp):
            for m, gidx in enumerate(idx_arr):
                raw_local[axis_i].append((int(gidx), supp, k, deg_arr, m))

    local_idxs_arr = []
    groups_for_axis = []
    for axis_i in range(d):
        items = raw_local[axis_i]
        items.sort(key=lambda t: t[0])
        gidxs = np.array([t[0] for t in items], dtype=np.int64)
        local_idxs_arr.append(gidxs)
        per_supp = defaultdict(list)
        for local_pos, (gidx, supp, k, deg_arr, m) in enumerate(items):
            per_supp[(supp, k, id(deg_arr))].append((local_pos, m, supp, k, deg_arr))
        groups = []
        for key, entries in per_supp.items():
            local_pos_arr = np.array([e[0] for e in entries], dtype=np.int64)
            m_arr = np.array([e[1] for e in entries], dtype=np.int64)
            supp = entries[0][2]
            k = entries[0][3]
            deg_arr_full = entries[0][4]
            deg_arr_sub = deg_arr_full[m_arr]
            groups.append((supp, k, deg_arr_sub, local_pos_arr))
        groups_for_axis.append(groups)

    return local_idxs_arr, groups_for_axis


def _fill_phi_dense_for_axis(Phi_dense, Phi2_dense, H_chunk,
                             groups_for_axis_i):
    """Fill dense (Nc, |S_i|) Phi and Phi2 blocks for one axis. Caller
    pre-zeros the buffers. Pass Phi2_dense=None to skip second-derivative.
    """
    for (supp, k, deg_arr, local_pos) in groups_for_axis_i:
        s = len(supp)
        # First-derivative
        deg_lookup = deg_arr.copy()
        deg_lookup[:, k] -= 1
        if s == 1:
            block = H_chunk[supp[0]][:, deg_lookup[:, 0]]
        elif s == 2:
            block = (H_chunk[supp[0]][:, deg_lookup[:, 0]]
                     * H_chunk[supp[1]][:, deg_lookup[:, 1]])
        elif s == 3:
            block = (H_chunk[supp[0]][:, deg_lookup[:, 0]]
                     * H_chunk[supp[1]][:, deg_lookup[:, 1]]
                     * H_chunk[supp[2]][:, deg_lookup[:, 2]])
        else:
            block = H_chunk[supp[0]][:, deg_lookup[:, 0]].copy()
            for l in range(1, s):
                block *= H_chunk[supp[l]][:, deg_lookup[:, l]]
        factor1 = np.sqrt(deg_arr[:, k].astype(np.float32))
        Phi_dense[:, local_pos] = factor1 * block

        if Phi2_dense is None:
            continue
        # Second-derivative
        mask2 = deg_arr[:, k] >= 2
        if mask2.any():
            deg_lookup2 = deg_arr[mask2].copy()
            deg_lookup2[:, k] -= 2
            if s == 1:
                block2 = H_chunk[supp[0]][:, deg_lookup2[:, 0]]
            elif s == 2:
                block2 = (H_chunk[supp[0]][:, deg_lookup2[:, 0]]
                          * H_chunk[supp[1]][:, deg_lookup2[:, 1]])
            elif s == 3:
                block2 = (H_chunk[supp[0]][:, deg_lookup2[:, 0]]
                          * H_chunk[supp[1]][:, deg_lookup2[:, 1]]
                          * H_chunk[supp[2]][:, deg_lookup2[:, 2]])
            else:
                block2 = H_chunk[supp[0]][:, deg_lookup2[:, 0]].copy()
                for l in range(1, s):
                    block2 *= H_chunk[supp[l]][:, deg_lookup2[:, l]]
            a_k = deg_arr[mask2, k].astype(np.float32)
            factor2 = np.sqrt(a_k * (a_k - 1.0))
            Phi2_dense[:, local_pos[mask2]] = factor2 * block2


def evaluate_features(Z, A, K):
    """
    Given sample matrix Z in R^{N x d} and dictionary A (as (supp, deg) tuples),
    evaluate the feature matrix Psi in R^{N x |A|} with Psi[k, a] = H_{A[a]}(Z[k]).

    Also returns:
      Phi_list: list of d arrays of shape (N, |A|), where Phi_list[i][k, a] =
                sqrt(alpha_i) * H_{alpha - e_i}(Z[k]) = partial_i H_alpha at Z[k].
      Phi2_list: list of d arrays, Phi2_list[i][k, a] = sqrt(alpha_i (alpha_i - 1))
                 * H_{alpha - 2 e_i}(Z[k]) = partial_i^2 H_alpha at Z[k].

    Uses grouped-by-support batched products (Psi) and dense per-axis blocks
    scattered into wide Phi/Phi2 (for caller API compatibility).
    """
    N, d = Z.shape
    nA = len(A)
    H = np.empty((d, N, K + 1), dtype=float)
    for j in range(d):
        H[j] = hermite_norm(Z[:, j], K)

    Psi = np.empty((N, nA), dtype=float)
    groups = _build_support_groups(A)
    for (supp, deg_arr, idx_arr) in groups:
        s = len(supp)
        if s == 1:
            block = H[supp[0]][:, deg_arr[:, 0]]
        elif s == 2:
            block = (H[supp[0]][:, deg_arr[:, 0]]
                     * H[supp[1]][:, deg_arr[:, 1]])
        elif s == 3:
            block = (H[supp[0]][:, deg_arr[:, 0]]
                     * H[supp[1]][:, deg_arr[:, 1]]
                     * H[supp[2]][:, deg_arr[:, 2]])
        else:
            block = H[supp[0]][:, deg_arr[:, 0]].copy()
            for l in range(1, s):
                block *= H[supp[l]][:, deg_arr[:, l]]
        Psi[:, idx_arr] = block

    # Phi_list / Phi2_list via dense per-axis, scattered to wide.
    # Match the float64 dtype that callers (build_A_b, assemble_T) expect.
    Phi_list = [np.zeros((N, nA), dtype=float) for _ in range(d)]
    Phi2_list = [np.zeros((N, nA), dtype=float) for _ in range(d)]
    local_idxs, groups_for_axis = _build_axis_meta(A, d)
    for i in range(d):
        li = local_idxs[i]
        n_local = len(li)
        if n_local == 0:
            continue
        # Compute the dense block in float64 directly using the same product
        # structure as _fill_phi_dense_for_axis but without the float32 cast.
        for (supp, k, deg_arr, local_pos) in groups_for_axis[i]:
            s = len(supp)
            # First-derivative block
            deg_lookup = deg_arr.copy()
            deg_lookup[:, k] -= 1
            if s == 1:
                block = H[supp[0]][:, deg_lookup[:, 0]]
            elif s == 2:
                block = (H[supp[0]][:, deg_lookup[:, 0]]
                         * H[supp[1]][:, deg_lookup[:, 1]])
            elif s == 3:
                block = (H[supp[0]][:, deg_lookup[:, 0]]
                         * H[supp[1]][:, deg_lookup[:, 1]]
                         * H[supp[2]][:, deg_lookup[:, 2]])
            else:
                block = H[supp[0]][:, deg_lookup[:, 0]].copy()
                for l in range(1, s):
                    block *= H[supp[l]][:, deg_lookup[:, l]]
            factor1 = np.sqrt(deg_arr[:, k].astype(float))
            Phi_list[i][:, li[local_pos]] = factor1 * block

            # Second-derivative block
            mask2 = deg_arr[:, k] >= 2
            if mask2.any():
                deg_lookup2 = deg_arr[mask2].copy()
                deg_lookup2[:, k] -= 2
                if s == 1:
                    block2 = H[supp[0]][:, deg_lookup2[:, 0]]
                elif s == 2:
                    block2 = (H[supp[0]][:, deg_lookup2[:, 0]]
                              * H[supp[1]][:, deg_lookup2[:, 1]])
                elif s == 3:
                    block2 = (H[supp[0]][:, deg_lookup2[:, 0]]
                              * H[supp[1]][:, deg_lookup2[:, 1]]
                              * H[supp[2]][:, deg_lookup2[:, 2]])
                else:
                    block2 = H[supp[0]][:, deg_lookup2[:, 0]].copy()
                    for l in range(1, s):
                        block2 *= H[supp[l]][:, deg_lookup2[:, l]]
                a_k = deg_arr[mask2, k].astype(float)
                factor2 = np.sqrt(a_k * (a_k - 1.0))
                Phi2_list[i][:, li[local_pos[mask2]]] = factor2 * block2

    return Psi, Phi_list, Phi2_list


def evaluate_features_psi_only(Z, A, K):
    """
    Slim variant of evaluate_features that returns only Psi (the H_alpha values),
    not the gradient/Hessian Phi/Phi2 lists. At large (N, |A|, d) this saves
    ~2*d times the memory of the full version, which matters for large M_norm
    QMC normalizer evaluations.

    Uses grouped-by-support batched products instead of an entry-by-entry
    Python loop over |A|.

    Returns Psi: (N, |A|).
    """
    N, d = Z.shape
    nA = len(A)
    H = np.empty((d, N, K + 1), dtype=float)
    for j in range(d):
        H[j] = hermite_norm(Z[:, j], K)

    Psi = np.empty((N, nA), dtype=float)
    groups = _build_support_groups(A)
    for (supp, deg_arr, idx_arr) in groups:
        s = len(supp)
        if s == 1:
            block = H[supp[0]][:, deg_arr[:, 0]]
        elif s == 2:
            block = (H[supp[0]][:, deg_arr[:, 0]]
                     * H[supp[1]][:, deg_arr[:, 1]])
        elif s == 3:
            block = (H[supp[0]][:, deg_arr[:, 0]]
                     * H[supp[1]][:, deg_arr[:, 1]]
                     * H[supp[2]][:, deg_arr[:, 2]])
        else:
            block = H[supp[0]][:, deg_arr[:, 0]].copy()
            for l in range(1, s):
                block *= H[supp[l]][:, deg_arr[:, l]]
        Psi[:, idx_arr] = block
    return Psi


# ---------------------------------------------------------------
# Rank-Gaussianization
# ---------------------------------------------------------------

def rank_gaussianize(X):
    """
    Componentwise rank-Gaussianization:
      Z[k, i] = Phi^{-1}(R[k, i] / (N + 1))
    where R[k, i] is the average rank of X[k, i] among X[:, i].
    """
    N, d = X.shape
    Z = np.empty_like(X, dtype=float)
    for i in range(d):
        # stats.rankdata uses average rank for ties by default
        R = stats.rankdata(X[:, i], method='average')
        Z[:, i] = stats.norm.ppf(R / (N + 1))
    return Z


# ---------------------------------------------------------------
# Convex Hermite score matching
# ---------------------------------------------------------------

def build_A_b(Z, A, K, Phi_list=None, Phi2_list=None):
    """
    Build Gram matrix A and right-hand side b for the convex quadratic
        J(theta) = (1/2) theta^T A theta - b^T theta + const.

    The Hyvarinen objective is
        J = (1/N) sum_k [ (1/2) ||s_theta(Z[k])||^2 + div(s_theta)(Z[k]) ]
    with s_theta(z) = -z + sum_alpha theta_alpha * grad H_alpha(z).

    Expanding, dropping theta-independent terms:
        A[a, b] = (1/N) sum_i sum_k Phi_i[k, a] * Phi_i[k, b]
        b[a]    = (1/N) sum_i sum_k Z[k, i] * Phi_i[k, a]      (from cross term)
                  - (1/N) sum_i sum_k Phi2_i[k, a]              (from div term)
    """
    N, d = Z.shape
    nA = len(A)
    if Phi_list is None or Phi2_list is None:
        _, Phi_list, Phi2_list = evaluate_features(Z, A, K)

    Amat = np.zeros((nA, nA), dtype=float)
    bvec = np.zeros(nA, dtype=float)
    for i in range(d):
        Phi_i = Phi_list[i]
        Amat += Phi_i.T @ Phi_i / N
        bvec += (Z[:, i:i+1].T @ Phi_i).ravel() / N
        bvec -= Phi2_list[i].sum(axis=0) / N
    return Amat, bvec, Phi_list, Phi2_list


def build_A_b_streaming(Z, A, K, keep_Phi=True):
    """
    Memory-efficient assembly of (A, b). Uses dense per-axis Phi blocks: for
    each axis i, only the |S_i| ~ |Lambda|*q/d columns whose support touches
    axis i are materialized. The (|S_i|, |S_i|) Gram contribution is computed
    via gemm and scatter-added into the full (|Lambda|, |Lambda|) Gram.
    Compared to the old entry-by-entry loop this gives ~20-30x speedup at
    realistic basis sizes; the savings grow with |Lambda|.

    If keep_Phi=True, returns Phi_list and Phi2_list as full sparse (N, |A|)
    matrices for compatibility with downstream callers like assemble_T(Phi_list,
    theta). The dense block is scattered back into a sparse Phi_i in this case,
    losing the dense memory win but keeping the FLOP win.
    """
    N, d = Z.shape
    nA = len(A)
    H = np.empty((d, N, K + 1), dtype=np.float32)
    for j in range(d):
        H[j] = hermite_norm(Z[:, j], K).astype(np.float32)

    Amat = np.zeros((nA, nA), dtype=float)
    bvec = np.zeros(nA, dtype=float)
    Phi_list = [] if keep_Phi else None
    Phi2_list = [] if keep_Phi else None

    local_idxs, groups_for_axis = _build_axis_meta(A, d)

    for i in range(d):
        li = local_idxs[i]
        n_local = len(li)
        if n_local == 0:
            if keep_Phi:
                Phi_list.append(np.zeros((N, nA), dtype=np.float32))
                Phi2_list.append(np.zeros((N, nA), dtype=np.float32))
            continue
        Phi_dense = np.zeros((N, n_local), dtype=np.float32)
        Phi2_dense = np.zeros((N, n_local), dtype=np.float32)
        _fill_phi_dense_for_axis(Phi_dense, Phi2_dense, H, groups_for_axis[i])

        Phi_dense_64 = Phi_dense.astype(np.float64)
        G_local = Phi_dense_64.T @ Phi_dense_64
        Amat[np.ix_(li, li)] += G_local / N
        bvec[li] += (Z[:, i].astype(np.float64) @ Phi_dense_64) / N
        bvec[li] -= Phi2_dense.astype(np.float64).sum(axis=0) / N

        if keep_Phi:
            # Reconstruct full sparse Phi_i for caller compatibility.
            Phi_i_full = np.zeros((N, nA), dtype=np.float32)
            Phi_i_full[:, li] = Phi_dense
            Phi_list.append(Phi_i_full)
            Phi2_i_full = np.zeros((N, nA), dtype=np.float32)
            Phi2_i_full[:, li] = Phi2_dense
            Phi2_list.append(Phi2_i_full)

    return Amat, bvec, Phi_list, Phi2_list, H


def build_weighted_A_b_streaming(Z, A, K, weights, keep_Phi=False):
    """
    Weighted-Fisher version of build_A_b_streaming.

    Implements the empirical RHS/LHS of the weighted Hyvarinen identity from
    Sec. 3.2 of the paper:

        Aw[alpha, alpha'] = sum_j w_j * (1/N) sum_n Phi_{j,alpha}(z_n) Phi_{j,alpha'}(z_n)
        bw[alpha]         = sum_j w_j * (1/N) sum_n [ z_{n,j} Phi_{j,alpha}(z_n)
                                                       - Phi^{(2)}_{j,alpha}(z_n) ]

    where Phi_{j,alpha}(z) = partial_j H_alpha(z) and
    Phi^{(2)}_{j,alpha}(z) = partial_j^2 H_alpha(z).

    Setting weights = np.ones(d) recovers build_A_b_streaming exactly, modulo
    accumulation order (numerical equality up to ~1e-13 relative).

    Parameters
    ----------
    Z : (N, d) array of samples (in the coordinate system the score is
        being matched against — e.g. the projected u-coordinates for Stage-2).
    A : Hermite multi-index dictionary (output of enumerate_dictionary).
    K : max per-coord degree (matches A's enumeration).
    weights : (d,) array of positive weights w_j (one per coordinate j).
    keep_Phi : if True, also returns Phi_list and Phi2_list. Defaults to
        False because Stage-2 has no need to assemble T after the inner solve.

    Returns
    -------
    Aw, bw, Phi_list_or_None, Phi2_list_or_None, H_precomp
    """
    weights = np.asarray(weights, dtype=float).ravel()
    N, d = Z.shape
    assert weights.shape == (d,), \
        f"weights must be shape (d,)=({d},), got {weights.shape}"
    assert np.all(weights > 0), "weights must be strictly positive"
    nA = len(A)

    H = np.empty((d, N, K + 1), dtype=np.float32)
    for j in range(d):
        H[j] = hermite_norm(Z[:, j], K).astype(np.float32)

    Aw = np.zeros((nA, nA), dtype=float)
    bw = np.zeros(nA, dtype=float)
    Phi_list = [] if keep_Phi else None
    Phi2_list = [] if keep_Phi else None

    local_idxs, groups_for_axis = _build_axis_meta(A, d)

    for i in range(d):
        w_i = float(weights[i])
        li = local_idxs[i]
        n_local = len(li)
        if n_local == 0:
            if keep_Phi:
                Phi_list.append(np.zeros((N, nA), dtype=np.float32))
                Phi2_list.append(np.zeros((N, nA), dtype=np.float32))
            continue
        Phi_dense = np.zeros((N, n_local), dtype=np.float32)
        Phi2_dense = np.zeros((N, n_local), dtype=np.float32)
        _fill_phi_dense_for_axis(Phi_dense, Phi2_dense, H, groups_for_axis[i])

        Phi_dense_64 = Phi_dense.astype(np.float64)
        G_local = Phi_dense_64.T @ Phi_dense_64
        Aw[np.ix_(li, li)] += w_i * G_local / N
        bw[li] += w_i * (Z[:, i].astype(np.float64) @ Phi_dense_64) / N
        bw[li] -= w_i * Phi2_dense.astype(np.float64).sum(axis=0) / N

        if keep_Phi:
            Phi_i_full = np.zeros((N, nA), dtype=np.float32)
            Phi_i_full[:, li] = Phi_dense
            Phi_list.append(Phi_i_full)
            Phi2_i_full = np.zeros((N, nA), dtype=np.float32)
            Phi2_i_full[:, li] = Phi2_dense
            Phi2_list.append(Phi2_i_full)

    return Aw, bw, Phi_list, Phi2_list, H


def build_psr_b_streaming(U, A, K, weights, y_sc):
    """
    Approach-2 (CAS-specific projected-score regression) RHS, eq. (3.7) /
    eq:psr-rhs of the paper.

    Computes
        b^{w,psr}[alpha] = sum_j w_j * (1/N) sum_n y_sc[j, n] * Phi_{j,alpha}(u^{(n)})

    where Phi_{j,alpha}(u) = partial_j H_alpha(u) is the same first-derivative
    feature used in build_weighted_A_b_streaming, and y_sc[j, n] is the
    Stage-1 score estimate at training point n, projected onto the recovered
    subspace coordinate j:
        y_sc[j, n] = (V_r^T T_train)[j, n].

    This shares the Hessian A^w with build_weighted_A_b_streaming(U, A, K,
    weights), so for the full PSR linear system one calls

        Aw, _, _, _, _ = build_weighted_A_b_streaming(U, A, K, weights)
        bw_psr        = build_psr_b_streaming(U, A, K, weights, y_sc)
        # then solve Aw beta = bw_psr.

    Parameters
    ----------
    U : (N, r) — projected (and possibly secondarily Gaussianized) samples,
        in the same coordinate system as the inner Hermite dictionary A.
    A : inner-r Hermite multi-index dictionary.
    K : max per-coord degree of A.
    weights : (r,) positive weights w_j (typically Stage-1 spectrum-derived).
    y_sc : (r, N) array of projected Stage-1 score labels.

    Returns
    -------
    bw_psr : (|A|,) array.
    """
    weights = np.asarray(weights, dtype=float).ravel()
    y_sc    = np.asarray(y_sc, dtype=float)
    N, r = U.shape
    assert weights.shape == (r,), \
        f"weights must be shape (r,)=({r},), got {weights.shape}"
    assert y_sc.shape == (r, N), \
        f"y_sc must be shape (r, N)=({r}, {N}), got {y_sc.shape}"
    assert np.all(weights > 0), "weights must be strictly positive"
    nA = len(A)

    # Precompute normalized Hermite evaluations once per coord
    H = np.empty((r, N, K + 1), dtype=np.float32)
    for j in range(r):
        H[j] = hermite_norm(U[:, j], K).astype(np.float32)

    bw_psr = np.zeros(nA, dtype=float)
    for i in range(r):
        w_i = float(weights[i])
        y_i = y_sc[i, :]                                # (N,)
        # Build Phi_i: partial_i H_alpha at each sample
        Phi_i = np.zeros((N, nA), dtype=np.float32)
        for a, (supp, deg) in enumerate(A):
            if i not in supp:
                continue
            j_idx = supp.index(i)
            alpha_i = deg[j_idx]
            factor = math.sqrt(alpha_i)
            vals1 = np.ones(N, dtype=np.float32)
            for j2_idx, j2 in enumerate(supp):
                d2 = deg[j2_idx] - 1 if j2_idx == j_idx else deg[j2_idx]
                vals1 *= H[j2, :, d2]
            Phi_i[:, a] = factor * vals1
        # Accumulate w_i * y_i^T @ Phi_i / N
        Phi_i_64 = Phi_i.astype(np.float64)
        bw_psr += w_i * (y_i @ Phi_i_64) / N

    return bw_psr


def build_psr_A_b_streaming(Z, A, K, weights, Y_sc, keep_Phi=False):
    """
    Projected-Score Regression (PSR) variant of the weighted Stage-2 solve.

    Same Hessian as build_weighted_A_b_streaming:
        Aw[alpha, alpha'] = sum_j w_j * (1/N) sum_n Phi_{j,alpha}(z_n) Phi_{j,alpha'}(z_n)

    Different RHS: replaces the Hyvärinen integration-by-parts cross-term
        sum_j w_j * (1/N) sum_n [ z_{n,j} Phi_{j,alpha}(z_n) - Phi^{(2)}_{j,alpha}(z_n) ]
    with a plug-in from Stage-1 score labels (paper eq. 3.7 = eq:psr-rhs):
        b^psr[alpha] = sum_j w_j * (1/N) sum_n y_{sc,j}^{(n)} Phi_{j,alpha}(z_n)

    where y_{sc}^{(n)} = V_r^T T(z^{(n)}) is the projected Stage-1 score
    evaluated on training samples (Y_sc shape (N, d) — one component per
    coordinate of the *Stage-2* coordinate system Z).

    The two variants share the same target population minimizer of J_W under
    the conditional-score identity (paper Theorem reduced-fisher-budget),
    but differ empirically:
      - direct: avoids Stage-1 score-error dependence; uses second derivatives.
      - PSR: inherits Stage-1 score errors; no second derivatives needed,
        which can be useful when the inner Hermite basis has high
        condition number.

    Parameters
    ----------
    Z : (N, d) array of samples in the Stage-2 coordinate system.
    A : Hermite multi-index dictionary.
    K : max per-coord degree.
    weights : (d,) positive weights w_j.
    Y_sc : (N, d) projected Stage-1 score labels in the same coordinate
        system as Z.
    keep_Phi : if True, also returns Phi_list. Phi2_list is always None
        (PSR does not need second derivatives).

    Returns
    -------
    Aw, b_psr, Phi_list_or_None, None, H_precomp
    """
    weights = np.asarray(weights, dtype=float).ravel()
    Y_sc = np.asarray(Y_sc, dtype=float)
    N, d = Z.shape
    assert weights.shape == (d,), \
        f"weights must be shape (d,)=({d},), got {weights.shape}"
    assert Y_sc.shape == (N, d), \
        f"Y_sc must be shape (N, d)=({N}, {d}), got {Y_sc.shape}"
    assert np.all(weights > 0), "weights must be strictly positive"
    nA = len(A)

    H = np.empty((d, N, K + 1), dtype=np.float32)
    for j in range(d):
        H[j] = hermite_norm(Z[:, j], K).astype(np.float32)

    Aw = np.zeros((nA, nA), dtype=float)
    b_psr = np.zeros(nA, dtype=float)
    Phi_list = [] if keep_Phi else None

    local_idxs, groups_for_axis = _build_axis_meta(A, d)

    for i in range(d):
        w_i = float(weights[i])
        li = local_idxs[i]
        n_local = len(li)
        if n_local == 0:
            if keep_Phi:
                Phi_list.append(np.zeros((N, nA), dtype=np.float32))
            continue
        Phi_dense = np.zeros((N, n_local), dtype=np.float32)
        # PSR doesn't need Phi2 — pass None.
        _fill_phi_dense_for_axis(Phi_dense, None, H, groups_for_axis[i])

        Phi_dense_64 = Phi_dense.astype(np.float64)
        G_local = Phi_dense_64.T @ Phi_dense_64
        Aw[np.ix_(li, li)] += w_i * G_local / N
        # Cross-term: Phi_dense^T @ Y_sc[:, i] -> (n_local,)
        b_psr[li] += w_i * (Y_sc[:, i].astype(np.float64) @ Phi_dense_64) / N

        if keep_Phi:
            Phi_i_full = np.zeros((N, nA), dtype=np.float32)
            Phi_i_full[:, li] = Phi_dense
            Phi_list.append(Phi_i_full)

    return Aw, b_psr, Phi_list, None, H


def assemble_T_from_H(H_precomp, A, theta, d, keep_phi_free=False):
    """
    Assemble T directly from the Hermite evaluations without storing Phi_i.
    H_precomp: (d, N, K+1) array from build_A_b_streaming.
    T[i, k] = (Phi_i)[k, :] @ theta
            = sum_{alpha with i in supp} sqrt(alpha_i) H_{alpha - e_i}(Z[k]) * theta[alpha]

    Uses the same dense per-axis builder as build_A_b_streaming for speed.
    """
    N = H_precomp.shape[1]
    T = np.zeros((d, N), dtype=float)

    local_idxs, groups_for_axis = _build_axis_meta(A, d)

    for i in range(d):
        li = local_idxs[i]
        n_local = len(li)
        if n_local == 0:
            continue
        Phi_dense = np.zeros((N, n_local), dtype=np.float32)
        _fill_phi_dense_for_axis(Phi_dense, None, H_precomp, groups_for_axis[i])
        theta_local = theta[li].astype(np.float64)
        T[i, :] = Phi_dense.astype(np.float64) @ theta_local

    return T


# Default Tikhonov scale: lambda = LAMBDA0_FACTOR_DEFAULT * ||A||_op (the
# kappa of eq:tikhonov). Must stay in sync with
# cas.stage1_reference.Config.lambda0_factor_emp.
LAMBDA0_FACTOR_DEFAULT = 1e-4


def solve_ridge(Amat, bvec, A, lambda0=None, degree_weighted=True):
    """Solve (A + Lambda) theta = b by Cholesky. Lambda = lambda_0 * diag(|alpha|^2)."""
    nA = len(A)
    total_deg = np.array([sum(deg) for (_, deg) in A], dtype=float)
    if lambda0 is None:
        lambda0 = LAMBDA0_FACTOR_DEFAULT * np.linalg.norm(Amat, ord=2)
    Lambda_diag = lambda0 * (total_deg ** 2) if degree_weighted else lambda0 * np.ones(nA)
    M = Amat + np.diag(Lambda_diag)
    c, low = cho_factor(M, lower=True)
    theta = cho_solve((c, low), bvec)
    return theta, lambda0


def solve_ridge_chaos_block(Amat, bvec, A, lambda0=None):
    """
    (d-light) Per-chaos-block scalar Tikhonov.

    For each chaos level k, add a constant shrinkage tau_k to the k-th diagonal
    block of A. tau_k = lambda0 * ||A^{(k,k)}||_op by default (lambda0 default
    1e-3). Off-diagonal blocks unchanged.

    Returns (theta, dict {k: tau_k}).
    """
    nA = len(A)
    total_deg = np.array([sum(deg) for (_, deg) in A], dtype=int)
    levels = sorted(set(total_deg.tolist()))
    if lambda0 is None:
        lambda0 = 1e-3
    Lambda_full = np.zeros_like(Amat)
    taus_per_block = {}
    for k in levels:
        if k == 0:
            continue
        idx = np.where(total_deg == k)[0]
        block = Amat[np.ix_(idx, idx)]
        tau_k = float(lambda0) * np.linalg.norm(block, ord=2)
        taus_per_block[k] = tau_k
        for i in idx:
            Lambda_full[i, i] = tau_k
    M = Amat + Lambda_full
    c, low = cho_factor(M, lower=True)
    theta = cho_solve((c, low), bvec)
    return theta, taus_per_block


def solve_ridge_chaos_block_mom(Amat, bvec, A, U_train=None, K_inner=None,
                                 lambda0=None, n_blocks=24):
    """
    Median-of-means chaos-block Tikhonov ($\\Lambda_{\\rm rcb}$, MoM variant).

    Robust to heavy-tailed Hermite-feature norms by replacing
        ||A^{(k,k)}||_op
    with the median across n_blocks per-sample-batch operator-norm estimates.

    Requires U_train (the projected training samples used to build Amat) and
    K_inner (the inner total-degree truncation), to rebuild per-block A
    matrices on data subsets.

    For each chaos level k:
      1. Partition U_train into n_blocks roughly equal sample blocks.
      2. For each block, compute A_j^{(k,k)} (the (k,k) sub-block of the
         per-block sample covariance of Hermite gradients).
      3. Take median_j ||A_j^{(k,k)}||_op as a robust block-norm estimate.

    The median commutes with orthogonal conjugation (each per-block norm
    is invariant) and with scalar inflation by c_k^{-2} (each per-block
    norm scales the same), so the chaos-block group action commutes with
    the median, preserving Prop 3.1 basis-invariance.

    Returns (theta, dict {k: tau_k}, dict {k: rho_k}). rho is 0 in MoM.
    """
    nA = len(A)
    total_deg = np.array([sum(deg) for (_, deg) in A], dtype=int)
    levels = sorted(set(total_deg.tolist()))
    if lambda0 is None:
        lambda0 = 1e-3

    # Build per-block A matrices via repeated streaming on row-subsets.
    if U_train is None or K_inner is None:
        # Fallback to plain CB
        return solve_ridge_chaos_block(Amat, bvec, A, lambda0=lambda0)

    N_tr = U_train.shape[0]
    block_size = max(1, N_tr // n_blocks)
    actual_n_blocks = min(n_blocks, N_tr // block_size)
    # Pre-build per-block A matrices (only diagonal blocks needed)
    per_block_op_norms = {k: [] for k in levels if k > 0}
    for j in range(actual_n_blocks):
        lo = j * block_size
        hi = (j + 1) * block_size if j < actual_n_blocks - 1 else N_tr
        U_j = U_train[lo:hi]
        # Build A on this sub-batch (we don't need bvec, just A diagonal blocks)
        A_j_full, _, _, _, _ = build_A_b_streaming(U_j, A, K_inner, keep_Phi=False)
        for k in levels:
            if k == 0:
                continue
            idx = np.where(total_deg == k)[0]
            block_jk = A_j_full[np.ix_(idx, idx)]
            per_block_op_norms[k].append(float(np.linalg.norm(block_jk, ord=2)))

    Lambda_full = np.zeros_like(Amat)
    taus_per_block = {}
    rhos_per_block = {}
    for k in levels:
        if k == 0:
            continue
        idx = np.where(total_deg == k)[0]
        # Median-of-means estimate
        tau_k = float(lambda0) * float(np.median(per_block_op_norms[k]))
        taus_per_block[k] = tau_k
        rhos_per_block[k] = 0.0  # not used; for return-shape consistency
        for i in idx:
            Lambda_full[i, i] = tau_k
    M = Amat + Lambda_full
    c, low = cho_factor(M, lower=True)
    theta = cho_solve((c, low), bvec)
    return theta, taus_per_block, rhos_per_block


def solve_ridge_chaos_block_quantile(Amat, bvec, A, lambda0=None, quantile=0.5):
    """
    Chaos-block Tikhonov using a quantile of per-block eigenvalues as the scale.

    tau_k = lambda0 * Q(lambda_i^{(k,k)}; quantile)
    where Q is the empirical quantile.

    quantile=1.0 recovers plain Lambda_cb (uses lam_max).
    quantile=0.5 uses the median: robust to operator-norm inflation by
                  outlier eigenvalues.

    Basis invariance: per-block scalar action c_k^{-2} scales every eigenvalue
    identically, hence the quantile by c_k^{-2}; orthogonal action leaves
    eigenvalues unchanged. Prop 3.1 extends.

    Returns (theta, dict {k: tau_k, scale_k}).
    """
    nA = len(A)
    total_deg = np.array([sum(deg) for (_, deg) in A], dtype=int)
    levels = sorted(set(total_deg.tolist()))
    if lambda0 is None:
        lambda0 = 1e-3

    Lambda_full = np.zeros_like(Amat)
    diag = {}
    for k in levels:
        if k == 0:
            continue
        idx = np.where(total_deg == k)[0]
        block = Amat[np.ix_(idx, idx)]
        eigs = np.linalg.eigvalsh(block)  # ascending
        scale_k = float(np.quantile(eigs, quantile))
        tau_k = float(lambda0) * scale_k
        diag[k] = {'tau_k': tau_k, 'scale_k': scale_k,
                   'eig_min': float(eigs.min()), 'eig_max': float(eigs.max())}
        for i in idx:
            Lambda_full[i, i] = tau_k
    M = Amat + Lambda_full
    c, low = cho_factor(M, lower=True)
    theta = cho_solve((c, low), bvec)
    return theta, diag


def solve_ridge_chaos_block_sigmoid(Amat, bvec, A, lambda0=None, N=None,
                                      p=2.0, return_diag=False):
    """
    Sigmoid-weighted per-direction blend of CB and whiten ridges ($\\Lambda_{\\rm rcb}$).

    For each chaos block k, compute eigendecomposition $\\hmA^{(k,k)} = U_k \\Lambda_k U_k^T$,
    then assign per-eigenvalue ridge

        tau_i = lam0 * lam_max + w_i * (lam_max - lam_i),

    where the per-direction weight w_i is sigmoidal in log(lam_i / lam_noise):

        w_i = 1 / (1 + (lam_i / lam_noise)^p),
        lam_noise = tr(A^{(k,k)}) / N.

    Interpretation:
      - lam_noise is the Marchenko-Pastur scale: a direction with population
        eigenvalue at this scale is at the noise edge.
      - w_i ≈ 1 for noise-floor directions (lam_i << lam_noise): full whitening
        lifts them up to lam_max.
      - w_i ≈ 0 for above-noise directions (lam_i >> lam_noise): only CB
        applied. The directions that genuinely carry signal are not whitened.

    Basis invariance: lam_i scales as c_k^{-2}, lam_noise = tr/N also scales
    as c_k^{-2}, so the ratio lam_i/lam_noise is invariant under the chaos-block
    group. Hence w_i is invariant, and the entire ridge transforms as c_k^{-2}.
    Prop 3.1 extends.

    Special cases:
      - p -> infinity: hard cutoff (full whitening below lam_noise, none above).
      - p = 0: w_i = 1/2 everywhere (uniform half-blend).
      - lam_noise -> 0 (large N): all w_i -> 0, reduces to plain CB.
      - lam_noise -> infinity (catastrophic small N): all w_i -> 1, reduces
        to whiten.

    Returns (theta, diag) where diag holds per-block w_i, lam_i, lam_noise
    for diagnostics; if return_diag=False returns just theta.
    """
    nA = len(A)
    total_deg = np.array([sum(deg) for (_, deg) in A], dtype=int)
    levels = sorted(set(total_deg.tolist()))
    if lambda0 is None:
        lambda0 = 1e-3
    if N is None or N <= 0:
        # Without N, can't compute noise scale — fall back to plain CB.
        return solve_ridge_chaos_block(Amat, bvec, A, lambda0=lambda0)

    Lambda_full = np.zeros_like(Amat)
    diag = {}
    for k in levels:
        if k == 0:
            continue
        idx = np.where(total_deg == k)[0]
        block = Amat[np.ix_(idx, idx)]
        eigvals_k, eigvecs_k = np.linalg.eigh(block)  # ascending
        lam_max = float(eigvals_k.max())
        # Marchenko-Pastur scale per block:
        lam_noise = float(np.trace(block)) / N
        # Sigmoid weights per eigenvalue
        # w_i = 1 / (1 + (lam_i / lam_noise)^p)
        # Use log-space for numerical stability when p is large.
        if lam_noise <= 0 or lam_max <= 0:
            w = np.zeros_like(eigvals_k)
        else:
            ratio = np.maximum(eigvals_k, 1e-30) / lam_noise
            w = 1.0 / (1.0 + ratio ** p)
        # Per-eigenvalue ridge: CB minimum + whitening admixture weighted by w
        tau_per = lambda0 * lam_max + w * (lam_max - eigvals_k)
        block_reg = eigvecs_k @ np.diag(tau_per) @ eigvecs_k.T
        Lambda_full[np.ix_(idx, idx)] += block_reg
        diag[k] = {
            'eigvals': eigvals_k,
            'lam_max': lam_max,
            'lam_noise': lam_noise,
            'w': w,
            'tau_per': tau_per,
            'dim_k': len(idx),
        }
    M = Amat + Lambda_full
    c, low = cho_factor(M, lower=True)
    theta = cho_solve((c, low), bvec)
    if return_diag:
        return theta, diag
    return theta, diag


def solve_ridge_trace(Amat, bvec, A, c=3.0, p=4.0, N=None, return_diag=False):
    """
    Trace ridge $\\Lambda_{\\rm tr}$ — the deployed Stage-2 regularizer per the paper.

    Paper formula (eq:trace-ridge):

        tau_i^{(k)} = c * sigma_k^2 + w_i^{(k)} * (lam_max^{(k)} - lam_i^{(k)}),
        sigma_k^2 = tr(A^{(k,k)}) / N,
        w_i^{(k)} = 1 / (1 + (lam_i^{(k)} / sigma_k^2) ** p).

    Differs from solve_ridge_chaos_block_sigmoid: that function uses
    `lam0 * lam_max` for the first term (an older parameterization);
    here we use `c * sigma_k^2`, which is the basis-invariant Stein-optimal
    uniform shrinkage at the per-class noise scale.

    Defaults: c=3 (nominal), p=4. See §3.3 of the main paper.
    """
    import numpy as np
    from scipy.linalg import cho_factor, cho_solve

    if N is None or N <= 0:
        raise ValueError("solve_ridge_trace requires N (training sample size)")

    nA = len(A)
    total_deg = np.array([sum(deg) for (_, deg) in A], dtype=int)
    levels = sorted(set(total_deg.tolist()))

    Lambda_full = np.zeros_like(Amat)
    diag = {}
    for k in levels:
        if k == 0:
            continue
        idx = np.where(total_deg == k)[0]
        block = Amat[np.ix_(idx, idx)]
        eigvals_k, eigvecs_k = np.linalg.eigh(block)  # ascending
        lam_max = float(eigvals_k.max())
        sigma_k_sq = float(np.trace(block)) / N    # per-class noise variance estimate
        if sigma_k_sq <= 0 or lam_max <= 0:
            w = np.zeros_like(eigvals_k)
        else:
            ratio = np.maximum(eigvals_k, 1e-30) / sigma_k_sq
            w = 1.0 / (1.0 + ratio ** p)
        # Paper formula: c * sigma_k^2 + w_i * (lam_max - lam_i)
        tau_per = c * sigma_k_sq + w * (lam_max - eigvals_k)
        block_reg = eigvecs_k @ np.diag(tau_per) @ eigvecs_k.T
        Lambda_full[np.ix_(idx, idx)] += block_reg
        diag[k] = {
            'eigvals': eigvals_k,
            'lam_max': lam_max,
            'sigma_k_sq': sigma_k_sq,
            'w': w,
            'tau_per': tau_per,
            'dim_k': len(idx),
            'c': c, 'p': p,
        }
    M = Amat + Lambda_full
    cf, low = cho_factor(M, lower=True)
    theta = cho_solve((cf, low), bvec)
    if return_diag:
        return theta, diag
    return theta, diag



def solve_ridge_trace_hybrid(Amat, bvec, A, c=3.0, p=4.0, delta=1e-5, N=None, return_diag=False):
    """
    Hybrid trace ridge + Sobolev floor: $\\Lambda_{\\rm tr+sob}$.

    Combines the basis-invariant trace ridge with a Sobolev-style floor that
    addresses misspecification bias at the boundary degree class.

    Per-eigenvalue formula:
      tau_i^(k) = c * sigma_hat_k^2 + w_i^(k) * (lam_max^(k) - lam_i^(k)) + delta * ||A||_op * k^2

    The first two terms are the standard trace ridge (Stein-optimal noise calibration
    + finite-sample whitening admixture). The third term is a misspecification
    floor that does NOT vanish with N — it tracks the cross-coupling between
    in-dictionary degree classes and the missing higher-degree truncation residual.

    Crossover analysis: Stein dominates when c * tr(A_kk)/N >> delta * ||A||_op * k^2,
    i.e., for N << c * tr(A_kk) / (delta * ||A||_op * k^2). With c=3, delta=1e-5,
    typical scales give crossover N around 50,000 for k=5: Stein for moderate N,
    Sobolev floor activates at very large N.

    Theoretical motivation:
      - bias-variance-misspecification decomposition (standard M-estimation)
      - statistical noise scales as 1/N (handled by trace ridge)
      - truncation/misspecification bias scales as constant (handled by Sobolev floor)
      - boundary degree class concentrates the misspecification (cf. Sobolev k^2 weight)

    Defaults: c=3 (paper-deployed nominal), p=4 (sigmoid steepness), delta=1e-5
    (Sobolev floor; chosen so crossover at N~50k for our banana grid).

    Reduces to:
      - Λ_tr (paper's trace ridge) when delta=0
      - Λ_dw-like when c=0 and only the third term survives (modulo block structure)
    """
    import numpy as np
    from scipy.linalg import cho_factor, cho_solve

    if N is None or N <= 0:
        raise ValueError("solve_ridge_trace_hybrid requires N (training sample size)")

    nA = len(A)
    total_deg = np.array([sum(deg) for (_, deg) in A], dtype=int)
    levels = sorted(set(total_deg.tolist()))

    # Compute ||A||_op once for the Sobolev-floor prefactor
    A_op = float(np.linalg.norm(Amat, ord=2))

    Lambda_full = np.zeros_like(Amat)
    diag = {'A_op': A_op, 'c': c, 'delta': delta, 'p': p}
    for k in levels:
        if k == 0:
            continue
        idx = np.where(total_deg == k)[0]
        block = Amat[np.ix_(idx, idx)]
        eigvals_k, eigvecs_k = np.linalg.eigh(block)  # ascending
        lam_max = float(eigvals_k.max())
        sigma_k_sq = float(np.trace(block)) / N
        if sigma_k_sq <= 0 or lam_max <= 0:
            w = np.zeros_like(eigvals_k)
        else:
            ratio = np.maximum(eigvals_k, 1e-30) / sigma_k_sq
            w = 1.0 / (1.0 + ratio ** p)
        # Sobolev floor: delta * ||A||_op * k^2
        sobolev_floor = delta * A_op * (k ** 2)
        # Combined per-eigenvalue ridge
        tau_per = c * sigma_k_sq + w * (lam_max - eigvals_k) + sobolev_floor
        block_reg = eigvecs_k @ np.diag(tau_per) @ eigvecs_k.T
        Lambda_full[np.ix_(idx, idx)] += block_reg
        diag[k] = {
            'eigvals': eigvals_k,
            'lam_max': lam_max,
            'sigma_k_sq': sigma_k_sq,
            'w': w,
            'tau_per': tau_per,
            'sobolev_floor': sobolev_floor,
            'dim_k': len(idx),
        }
    M = Amat + Lambda_full
    cf, low = cho_factor(M, lower=True)
    theta = cho_solve((cf, low), bvec)
    if return_diag:
        return theta, diag
    return theta, diag



def solve_ridge_chaos_block_blend(Amat, bvec, A, lambda0=None,
                                    N=None, alpha=4.0, mode='sample_size'):
    """
    Robust chaos-block Tikhonov ($\\Lambda_{\\rm rcb}$, blend variant).

    Per chaos level k, blend between:
      - the eigenvalue-whitened ridge: tau_i^whiten = (lam_max - lam_i) + lam0 lam_max
      - the plain chaos-block ridge: tau^cb = lam0 lam_max
    by a per-block coefficient beta_k:
        tau_i = lam0 lam_max + beta_k * (lam_max - lam_i)

    Two modes for choosing beta_k:

    'deficiency' (default, data-adaptive):
        beta_k = 1 - r_stable / |A^(k)|,
        where r_stable := tr(A^(k,k)) / lam_max(A^(k,k)) is the stable rank.
        At well-conditioned blocks (r_stable ~ |A^(k)|), beta_k ~ 0.
        At rank-deficient blocks (r_stable << |A^(k)|), beta_k ~ 1.
        Both r_stable and |A^(k)| are basis-invariant.

    'sample_size' (legacy):
        beta_k = min(1, alpha * |A^(k)| / N).
        Pure sample-size criterion. Can over-regularize at moderate N when
        population eigenvalues are genuinely spread.

    At any beta_k = 0 we recover plain Lambda_cb; at beta_k = 1 we recover
    the eigenvalue-whitened ridge.

    Basis invariance under the chaos-block group G is preserved in both modes:
    the regularizer tau_i scales by c_k^{-2} under per-block scalar action and
    eigenvectors transform by the orthogonal action; beta_k is basis-invariant
    (depends on tr/lam_max ratio in 'deficiency' mode, or on N/|A^(k)| both
    untouched by G in 'sample_size' mode).

    Conditioning bound: at beta_k = 1, the regularized block has condition
    number 1 + lam0; at beta_k < 1, smallest regularized eigenvalue is
        lam_min (1 - beta_k) + lam_max (beta_k + lam0).

    Returns (theta, dict {k: blend diagnostics}).
    """
    nA = len(A)
    total_deg = np.array([sum(deg) for (_, deg) in A], dtype=int)
    levels = sorted(set(total_deg.tolist()))
    if lambda0 is None:
        lambda0 = 1e-3

    Lambda_full = np.zeros_like(Amat)
    diag = {}
    for k in levels:
        if k == 0:
            continue
        idx = np.where(total_deg == k)[0]
        block = Amat[np.ix_(idx, idx)]
        eigvals_k, eigvecs_k = np.linalg.eigh(block)  # ascending
        lam_max = float(eigvals_k.max())
        dim_k = len(idx)
        if mode == 'deficiency':
            if lam_max <= 0:
                beta_k = 0.0
                r_stable = 0.0
            else:
                trace = float(np.trace(block))
                r_stable = trace / lam_max
                beta_k = max(0.0, 1.0 - r_stable / max(dim_k, 1))
        elif mode == 'sample_size':
            r_stable = float('nan')
            if N is None:
                beta_k = 0.0
            else:
                beta_k = min(1.0, alpha * dim_k / N)
        else:
            raise ValueError(f'unknown mode: {mode}')
        # Blended per-eigenvalue tau:
        # tau_i = lam0 * lam_max + beta_k * (lam_max - lam_i)
        tau_per = float(lambda0) * lam_max + beta_k * (lam_max - eigvals_k)
        block_reg = eigvecs_k @ np.diag(tau_per) @ eigvecs_k.T
        Lambda_full[np.ix_(idx, idx)] += block_reg
        diag[k] = {
            'eig_min': float(eigvals_k.min()),
            'eig_max': lam_max,
            'beta_k': beta_k,
            'r_stable': r_stable,
            'dim_k': dim_k,
            'tau_min': float(tau_per.min()),
            'tau_max': float(tau_per.max()),
        }
    M = Amat + Lambda_full
    c, low = cho_factor(M, lower=True)
    theta = cho_solve((c, low), bvec)
    return theta, diag


def solve_ridge_chaos_eig(Amat, bvec, A, lambda0=None, mode='whiten'):
    """
    (d-aggressive) Per-chaos-block eigendecomposition + per-eigenvalue Tikhonov.

    Within each chaos block k, eigendecompose A^{(k,k)} = U_k Lambda_k U_k^T.
    Apply per-eigenvalue shrinkage in eigenbasis. Several modes:

    - mode='whiten': tau_i = (lambda_max - lambda_i) + lambda0*lambda_max
        Adds a non-uniform shift so that lambda_i + tau_i = lambda_max + lambda0*lambda_max
        for ALL i in the block. The inverse acts uniformly across eigendirections
        within the block — true spectral whitening per block.

    - mode='proportional': tau_i = lambda0 * lambda_max
        Constant shrinkage per block (degenerates to chaos_block).

    - mode='inverse': tau_i = lambda0 * lambda_max + lambda0 * lambda_max^2 / lambda_i
        Adds a 1/lambda_i term to lift small eigenvalues. Compromise between
        constant and full whitening.

    Returns (theta, dict per-block diagnostics).
    """
    nA = len(A)
    total_deg = np.array([sum(deg) for (_, deg) in A], dtype=int)
    levels = sorted(set(total_deg.tolist()))
    if lambda0 is None:
        lambda0 = 1e-3
    Lambda_full = np.zeros_like(Amat)
    diag = {}
    for k in levels:
        if k == 0:
            continue
        idx = np.where(total_deg == k)[0]
        block = Amat[np.ix_(idx, idx)]
        eigvals_k, eigvecs_k = np.linalg.eigh(block)  # ascending
        lam_max = eigvals_k.max()
        # Compute per-eigenvalue tau according to mode
        if mode == 'whiten':
            # Pull every eigenvalue up to lam_max, plus a small uniform shrinkage.
            tau_per = (lam_max - eigvals_k) + lambda0 * lam_max
        elif mode == 'proportional':
            tau_per = np.full_like(eigvals_k, lambda0 * lam_max)
        elif mode == 'inverse':
            # Floor the eigenvalues at lam_max/1e6 for numerical safety
            eigs_safe = np.maximum(eigvals_k, lam_max / 1e6)
            tau_per = lambda0 * lam_max * (1.0 + lam_max / eigs_safe)
        else:
            raise ValueError(f"unknown mode: {mode}")
        block_reg = eigvecs_k @ np.diag(tau_per) @ eigvecs_k.T
        Lambda_full[np.ix_(idx, idx)] += block_reg
        diag[k] = {'eig_min': float(eigvals_k.min()), 'eig_max': float(eigvals_k.max()),
                   'tau_min': float(tau_per.min()), 'tau_max': float(tau_per.max())}
    M = Amat + Lambda_full
    c, low = cho_factor(M, lower=True)
    theta = cho_solve((c, low), bvec)
    return theta, diag


def solve_ridge_full(Amat, bvec, A, lambda0=None, mode='whiten'):
    """
    (d-full) Global eigendecomposition + per-eigenvalue Tikhonov.

    Same modes as solve_ridge_chaos_eig but operating on the full Hessian
    rather than per-chaos-block.
    """
    if lambda0 is None:
        lambda0 = 1e-3
    eigvals, eigvecs = np.linalg.eigh(Amat)
    lam_max = eigvals.max()
    if mode == 'whiten':
        tau_per = (lam_max - eigvals) + lambda0 * lam_max
    elif mode == 'proportional':
        tau_per = np.full_like(eigvals, lambda0 * lam_max)
    elif mode == 'inverse':
        eigs_safe = np.maximum(eigvals, lam_max / 1e6)
        tau_per = lambda0 * lam_max * (1.0 + lam_max / eigs_safe)
    else:
        raise ValueError(f"unknown mode: {mode}")
    inv_diag = 1.0 / (eigvals + tau_per)
    theta = eigvecs @ (inv_diag * (eigvecs.T @ bvec))
    diag = {'eig_min': float(eigvals.min()), 'eig_max': float(eigvals.max()),
            'tau_min': float(tau_per.min()), 'tau_max': float(tau_per.max())}
    return theta, diag


# ---------------------------------------------------------------
# Score samples and randomized eigendecomposition
# ---------------------------------------------------------------

def assemble_T(Phi_list, theta):
    """T[i, k] = sum_a Phi_list[i][k, a] * theta[a]. Shape (d, N)."""
    d = len(Phi_list)
    N = Phi_list[0].shape[0]
    T = np.empty((d, N), dtype=float)
    for i in range(d):
        T[i, :] = Phi_list[i] @ theta
    return T


def randomized_eig(T, r, p=10, q_pwr=2, seed=0):
    """
    Randomized eigendecomposition of C_hat = T T^T / N for top r eigenpairs.
    Halko-Martinsson-Tropp style. Returns (V_r, eigvals) with eigvals descending.
    V_r has shape (d, r).
    """
    rng = np.random.default_rng(seed)
    d, N = T.shape
    Omega = rng.standard_normal((d, r + p))
    # Y = T (T^T Omega) / N
    Y = T @ (T.T @ Omega) / N
    for _ in range(q_pwr):
        Q, _ = qr(Y, mode='economic')
        Y = T @ (T.T @ Q) / N
    Q, _ = qr(Y, mode='economic')
    # B = Q^T T T^T Q / N
    QtT = Q.T @ T
    B = QtT @ QtT.T / N
    # Symmetrize and eigendecompose
    B = 0.5 * (B + B.T)
    w, W = eigh(B)
    # Descending
    idx = np.argsort(-w)
    w = w[idx]
    W = W[:, idx]
    V_full = Q @ W  # shape (d, r+p)
    return V_full[:, :r], w[:r], w  # also return full eigvals of B for diagnostics


# ---------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------

def stein_residual(T, Z):
    """
    Compute ||M - S||_F / ||S||_F where
      M = T Z / N        (should equal Sigma^Z - I under the Stein identity)
      S = Z^T Z / N - I
    """
    N, d = Z.shape
    M = T @ Z / N           # shape (d, d)
    S = Z.T @ Z / N - np.eye(d)
    num = np.linalg.norm(M - S, ord='fro')
    den = np.linalg.norm(S, ord='fro')
    if den < 1e-12:
        return num  # absolute scale if S tiny
    return num / den


def hermite_degree_decomp(Phi_list, theta, A):
    """Return array tr_C_k for k=1..K_max, where tr_C_k = (1/N)||T_k||_F^2
    with T_k formed from theta restricted to multi-indices of total degree k."""
    N = Phi_list[0].shape[0]
    total_deg = np.array([sum(deg) for (_, deg) in A])
    K_max = int(total_deg.max())
    traces = np.zeros(K_max, dtype=float)
    for k in range(1, K_max + 1):
        mask = (total_deg == k)
        if not mask.any():
            continue
        theta_k = np.where(mask, theta, 0.0)
        T_k = assemble_T(Phi_list, theta_k)
        traces[k - 1] = np.sum(T_k * T_k) / N
    return traces


# ---------------------------------------------------------------
# Top-level CAS routine
# ---------------------------------------------------------------

def cas_fit(X, K=3, q=3, r=None, lambda0=None, p=10, q_pwr=2, seed=0,
            return_all=False):
    """
    Full CAS pipeline. Returns dict with V_r, eigvals, theta, Stein residual,
    Hermite-degree traces, residual trace, and the feature matrices if requested.

    If r is None, returns the top (r+p)-spectrum and user can select r post hoc.
    """
    N, d = X.shape
    Z = rank_gaussianize(X)
    A = enumerate_dictionary(d, K, q)
    Amat, bvec, Phi_list, Phi2_list = build_A_b(Z, A, K)
    theta, lambda0_used = solve_ridge(Amat, bvec, A, lambda0=lambda0)
    T = assemble_T(Phi_list, theta)
    r_eff = r if r is not None else min(d, 10)
    V_r, top_eigvals, all_eigvals = randomized_eig(T, r_eff, p=p, q_pwr=q_pwr, seed=seed)
    stein = stein_residual(T, Z)
    trC_k = hermite_degree_decomp(Phi_list, theta, A)
    trC_total = float(np.sum(T * T) / N)
    E_r = trC_total - float(np.sum(top_eigvals))

    result = {
        'Z': Z,
        'V_r': V_r,
        'eigvals_top': top_eigvals,
        'eigvals_all': all_eigvals,
        'theta': theta,
        'stein_residual': float(stein),
        'trC_k': trC_k,
        'trC_total': trC_total,
        'E_r': float(E_r),
        'lambda0': float(lambda0_used),
        'dict_size': len(A),
        'K': K, 'q': q, 'd': d, 'N': N,
    }
    if return_all:
        result['A'] = A
        result['Amat'] = Amat
        result['bvec'] = bvec
        result['Phi_list'] = Phi_list
        result['T'] = T
    return result


def cas_fit_streaming(X, K=3, q=3, r=None, lambda0=None, p=10, q_pwr=2, seed=0,
                       keep_phi=False, even_degree=False):
    """
    Memory-efficient variant of cas_fit for large d or |A|.

    If keep_phi=False, never stores the Phi_i matrices; feature evaluations
    happen twice (once for A,b, once for T). Safe for d*N*|A| regimes where
    full Phi storage would exceed memory.

    If even_degree=True, restricts the Hermite dictionary to even total degree
    |alpha|_1. For a globally sign-symmetric copula (c^Z(-Z) = c^Z(Z), e.g. the
    conformal z^3 law) the odd-degree coefficients are exactly zero in the
    population, so dropping them removes null modes at no bias cost and lowers
    the small-N variance of the subspace estimate.
    """
    N, d = X.shape
    Z = rank_gaussianize(X)
    A = enumerate_dictionary(d, K, q)
    if even_degree:
        A = [a for a in A if sum(a[1]) % 2 == 0]
    Amat, bvec, Phi_list, Phi2_list, H_precomp = build_A_b_streaming(
        Z, A, K, keep_Phi=False)
    theta, lambda0_used = solve_ridge(Amat, bvec, A, lambda0=lambda0)

    # Build the dense per-axis Phi blocks ONCE; reuse for the main T and for
    # every Hermite-degree restriction T_k. Avoids K_max+1 redundant rebuilds.
    local_idxs, groups_for_axis = _build_axis_meta(A, d)
    Phi_dense_list = [None] * d
    for i in range(d):
        li = local_idxs[i]
        n_local = len(li)
        if n_local == 0:
            Phi_dense_list[i] = (li, None)
            continue
        Phi_dense = np.zeros((N, n_local), dtype=np.float32)
        _fill_phi_dense_for_axis(Phi_dense, None, H_precomp, groups_for_axis[i])
        Phi_dense_list[i] = (li, Phi_dense)

    def _T_from_dense(theta_vec):
        T = np.zeros((d, N), dtype=float)
        for i in range(d):
            li, Phi_dense = Phi_dense_list[i]
            if Phi_dense is None:
                continue
            theta_local = theta_vec[li].astype(np.float64)
            T[i, :] = Phi_dense.astype(np.float64) @ theta_local
        return T

    T = _T_from_dense(theta)
    r_eff = r if r is not None else min(d, 10)
    V_r, top_eigvals, all_eigvals = randomized_eig(T, r_eff, p=p, q_pwr=q_pwr, seed=seed)
    stein = stein_residual(T, Z)

    # Hermite-degree decomposition via theta restriction. Each T_k is just a
    # different linear combination of the SAME Phi blocks; reuse them.
    total_deg = np.array([sum(deg) for (_, deg) in A])
    K_max = int(total_deg.max())
    trC_k = np.zeros(K_max, dtype=float)
    for k in range(1, K_max + 1):
        mask = (total_deg == k)
        if not mask.any():
            continue
        theta_k = np.where(mask, theta, 0.0)
        T_k = _T_from_dense(theta_k)
        trC_k[k - 1] = np.sum(T_k * T_k) / N
    trC_total = float(np.sum(T * T) / N)
    E_r = trC_total - float(np.sum(top_eigvals))
    return {
        'Z': Z,
        'V_r': V_r,
        'eigvals_top': top_eigvals,
        'eigvals_all': all_eigvals,
        'theta': theta,
        'T_train': T,
        'stein_residual': float(stein),
        'trC_k': trC_k,
        'trC_total': trC_total,
        'E_r': float(E_r),
        'lambda0': float(lambda0_used),
        'dict_size': len(A),
        'K': K, 'q': q, 'd': d, 'N': N,
    }




# ---------------------------------------------------------------
# Utility: principal angle between two subspaces
# ---------------------------------------------------------------

def principal_angle(V1, V2):
    """
    Maximum principal angle in degrees between column spans of V1, V2.
    V1, V2 have orthonormal columns (or we orthonormalize first).
    """
    Q1, _ = qr(V1, mode='economic')
    Q2, _ = qr(V2, mode='economic')
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    # Principal angles are arccos(s); return the max (worst-aligned direction)
    angles_rad = np.arccos(np.minimum(1.0, np.abs(s)))
    return float(np.degrees(angles_rad.max()))


def sin_theta_F(V1, V2):
    """Frobenius-norm sine of principal angles between col spans of V1, V2."""
    Q1, _ = qr(V1, mode='economic')
    Q2, _ = qr(V2, mode='economic')
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    sin_sq = np.maximum(0.0, 1.0 - s ** 2)
    return float(np.sqrt(np.sum(sin_sq)))


# ---------------------------------------------------------------------------
# Stage-2 estimator (Part I, Section 3)
# ---------------------------------------------------------------------------

def solve_ridge_theoretical(Amat, bvec, A, U_train=None,
                              c_cov=None, c_sob=None,
                              N=None, return_diag=False,
                              c=None, delta=None):
    """
    Theoretically derived regularizer with centering constraint (paper §3.3).

    Implements the regularizer from Eq. (R-combined) of the paper:

      R(Â_r) = c_cov · diag_k( σ̂²_k I_{p_k} )                       (R_cov)
             + c_sob · ‖Â_r‖_op · diag(|α|₁²)                          (R_sob)

    where each summand is a *separate* block-diagonal structure:
      - R_cov has per-degree-class identities I_{p_k} of size p_k = |Λ^{(k)}|,
        scaled by the empirical per-class trace σ̂²_k := tr(Â_r^{(k,k)}) / N.
      - R_sob has a global diagonal with weight |α|₁² (which equals k² on
        class Λ^{(k)}), scaled by ‖Â_r‖_op.

    These add on each class to a single per-class identity multiplied by
        [ c_cov · σ̂²_k  +  c_sob · ‖Â_r‖_op · k² ]
    but the two coefficients have distinct theoretical roles:
      - c_cov: per-class variance shrinkage (Stein, vanishes as N → ∞)
      - c_sob: Sobolev floor (asymptotically N-independent)

    Plus the centering constraint  Ĉθ = 0,  with
        Ĉ[j, α] = (1/N) Σ_n  ∂_j H_α(u_n)
    enforced exactly via Lagrangian.

    Published defaults: (c_cov, c_sob) = (3, 1e-5).

    Keyword aliases: ``c`` and ``delta``
    are accepted and map to ``c_cov`` and ``c_sob`` respectively.
    """
    import numpy as np
    from scipy.linalg import cho_factor, cho_solve, solve

    # Resolve aliases: explicit new-style wins, then old-style, then default.
    if c_cov is None:
        c_cov = c if c is not None else 3.0
    if c_sob is None:
        c_sob = delta if delta is not None else 1e-5

    if N is None or N <= 0:
        raise ValueError("requires N (training sample size)")
    if U_train is None:
        raise ValueError("requires U_train (N x r training projections)")

    nA = len(A)
    r = U_train.shape[1]
    A_op = float(np.linalg.norm(Amat, ord=2))
    total_deg = np.array([sum(deg) for (_, deg) in A], dtype=int)

    # ----- R_cov: per-class Stein term  c_cov · σ̂²_k · I_{p_k}  on each class -----
    R_cov = np.zeros(nA)
    by_deg = {}
    for a, td in enumerate(total_deg):
        by_deg.setdefault(int(td), []).append(a)
    for k, idx_list in by_deg.items():
        idx = np.array(idx_list)
        A_kk = Amat[np.ix_(idx, idx)]
        sigma_k_sq = max(float(np.trace(A_kk)) / N, 0.0)
        R_cov[idx] = c_cov * sigma_k_sq

    # ----- R_sob: Sobolev floor  c_sob · ‖Â_r‖_op · |α|₁²  -----
    R_sob = c_sob * A_op * (total_deg ** 2)

    R = np.diag(R_cov + R_sob)
    M = Amat + R

    # ----- Build centering constraint Ĉ[j, α] = (1/N) Σ_n  ∂_j H_α(u_n) -----
    K_max = int(total_deg.max())
    H_vals = np.empty((N, r, K_max + 1))
    for j in range(r):
        H_vals[:, j, :] = hermite_norm(U_train[:, j], K_max)

    C = np.zeros((r, nA))
    for a, (supp, deg) in enumerate(A):
        for j_idx, j in enumerate(supp):
            alpha_j = deg[j_idx]
            if alpha_j == 0:
                continue
            factor = math.sqrt(alpha_j)
            vals = np.full(N, factor, dtype=np.float64)
            for j2_idx, j2 in enumerate(supp):
                d2 = deg[j2_idx] - 1 if j2_idx == j_idx else deg[j2_idx]
                vals = vals * H_vals[:, j2, d2]
            C[j, a] = float(np.mean(vals))

    # ----- Solve via Lagrangian:  θ = M^{-1}b - M^{-1}C^T (C M^{-1} C^T)^{-1} C M^{-1} b
    cf, low = cho_factor(M, lower=True)
    M_inv_b = cho_solve((cf, low), bvec)
    M_inv_CT = cho_solve((cf, low), C.T)
    schur = C @ M_inv_CT
    rhs = C @ M_inv_b
    try:
        lam = solve(schur, rhs, assume_a='sym')
    except np.linalg.LinAlgError:
        lam = np.linalg.lstsq(schur, rhs, rcond=None)[0]
    theta = M_inv_b - M_inv_CT @ lam

    diag = {
        'A_op': A_op,
        'c_cov': c_cov, 'c_sob': c_sob,
        'c': c_cov, 'delta': c_sob,   # legacy aliases for downstream caches
        'R_cov_diag_max': float(R_cov.max()),
        'R_sob_diag_max': float(R_sob.max()),
        'theta_norm': float(np.linalg.norm(theta)),
        'centering_residual': float(np.linalg.norm(C @ theta)),
    }
    return (theta, diag) if return_diag else (theta, diag)


def solve_ridge_hermitescaled(Amat, bvec, A, c=3.0, alpha=0.0, delta=1e-5,
                                N=None, return_diag=False):
    """
    Hermite-scaled BISC: tau_k = c * k^alpha * sigma_hat_k^2 + delta*||A||_op*k^2.

    With c=0,alpha=0,delta=κ: pure Sobolev-Tikhonov  R = κ ||A||_op |α|².
    With c=0,alpha=0,delta=1e-12: essentially unregularized solve.

    No centering constraint (so this is the ablation against R_cov + R_sob + centering).
    """
    import numpy as np
    from scipy.linalg import cho_factor, cho_solve

    if N is None or N <= 0:
        raise ValueError("solve_ridge_hermitescaled requires N")

    nA = len(A)
    total_deg = np.array([sum(deg) for (_, deg) in A], dtype=int)
    levels = sorted(set(total_deg.tolist()))
    A_op = float(np.linalg.norm(Amat, ord=2))

    Lambda_full = np.zeros_like(Amat)
    diag = {'A_op': A_op, 'c': c, 'alpha': alpha, 'delta': delta}

    for k in levels:
        if k == 0:
            continue
        idx = np.where(total_deg == k)[0]
        block = Amat[np.ix_(idx, idx)]
        sigma_k_sq = float(np.trace(block)) / N
        tau_eb = c * (k ** alpha) * sigma_k_sq
        sobolev_floor = delta * A_op * (k ** 2)
        tau = tau_eb + sobolev_floor
        for ii in idx:
            Lambda_full[ii, ii] += tau

    M = Amat + Lambda_full
    cf, low = cho_factor(M, lower=True)
    theta = cho_solve((cf, low), bvec)
    return (theta, diag) if return_diag else (theta, diag)
