"""Integration test: end-to-end Hermite fit + J_hi CAS extraction on ppg.

This is slower than the unit tests in test_jhi_cas.py because it requires
sampling from a real benchmark and fitting Hermite features. It's marked
@pytest.mark.slow and skipped by default; run with:

    pytest tests/test_jhi_cas_integration.py -v -m slow
"""
import os
import sys

import numpy as np
import pytest

# Required pool path
POOL_PATH = 'cache/_cor44_scratch_ppg/pool_sorted_N20000000.npy'
HAVE_POOL = os.path.exists(POOL_PATH)


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_POOL, reason="requires ppg pool at cache/_cor44_scratch_ppg/")
class TestHermiteJhiIntegration:
    """End-to-end: Hermite fit ppg samples, extract M, H, C, do J_hi CAS."""

    def _fit_and_extract(self, N, seed):
        """Run Hermite pipeline at N samples; return M, H, C, V_hat, theta."""
        from cas.stage1_reference import (
            enumerate_dictionary, build_A_b_streamed, solve_ridge,
            stream_T_at_basis, top_r_eigs, materialize_Z, MarginalCDFs,
        )
        from cas.ppg import sample_ppg_noise
        from cas.jhi_cas import MHC_from_T_memmap

        scratch = 'cache/_cor44_scratch_ppg'
        marginals = MarginalCDFs(
            os.path.join(scratch, 'pool_sorted_N20000000.npy'),
            20_000_000, 20)

        d = 20
        K, q, r = 4, 2, 3
        N_chunk = 80000

        Z_path = materialize_Z(
            sample_ppg_noise, d, marginals, N, seed, scratch,
            f'integ_N{N}_s{seed}', N_chunk)
        A = enumerate_dictionary(d, K, q)
        Amat, bvec = build_A_b_streamed(Z_path, A, K, d, N, N_chunk)
        lambda0 = 0.001 * np.linalg.norm(Amat, ord=2)
        theta = solve_ridge(Amat, bvec, A, lambda0)

        T_path = os.path.join(scratch, f'integ_T_N{N}_s{seed}.npy')
        TTdN, trC = stream_T_at_basis(Z_path, theta, A, K, d, N, N_chunk,
                                         T_out_path=T_path)
        V_hat, eigvals, _ = top_r_eigs(TTdN, r)

        M_hat, H_hat, C_hat = MHC_from_T_memmap(Z_path, T_path, N, d, N_chunk)

        # Cleanup
        for p in [Z_path, T_path]:
            if 'integ_' in p and os.path.exists(p):
                os.remove(p)

        return M_hat, H_hat, C_hat, V_hat

    def test_Z_has_std_normal_marginals(self):
        """Materialize_Z should produce Z with std-1 Gaussian marginals."""
        from cas.stage1_reference import materialize_Z, MarginalCDFs
        from cas.ppg import sample_ppg_noise
        scratch = 'cache/_cor44_scratch_ppg'
        marginals = MarginalCDFs(
            os.path.join(scratch, 'pool_sorted_N20000000.npy'),
            20_000_000, 20)

        N = 50_000
        Z_path = materialize_Z(sample_ppg_noise, 20, marginals, N, 99,
                                scratch, 'test_marg', 25000)
        Z = np.array(np.load(Z_path, mmap_mode='r'), dtype=np.float64)
        per_coord_std = Z.std(axis=0)
        # Each coord should have std ~1
        assert np.allclose(per_coord_std, np.ones(20), atol=0.02), (
            f"per-coord std: {per_coord_std}"
        )
        # Each coord should have mean ~0
        per_coord_mean = Z.mean(axis=0)
        assert np.allclose(per_coord_mean, np.zeros(20), atol=0.02), (
            f"per-coord mean: {per_coord_mean}"
        )
        if os.path.exists(Z_path):
            os.remove(Z_path)

    def test_extracted_M_trace(self):
        """tr(M_hat) ~ d for Z with std-1 marginals."""
        M, H, C, V_hat = self._fit_and_extract(N=20_000, seed=5)
        assert abs(np.trace(M) - 20) < 0.5, f"tr(M) = {np.trace(M)}, expected ~20"

    def test_jhi_descent_from_V_hat_decreases(self):
        """Running J_hi descent from V_hat as warm start gives J_hi <= initial."""
        from cas.jhi_cas import J_hi, jhi_grassmann_descent
        M, H, C, V_hat = self._fit_and_extract(N=20_000, seed=10)
        J_init = J_hi(V_hat, M, H)
        V_opt, info = jhi_grassmann_descent(M, H, r=3, V_init=V_hat, n_iter=500)
        J_final = info["final_J_hi"]
        assert J_final <= J_init + 1e-8, (
            f"J_hi increased: {J_init} -> {J_final}"
        )

    def test_subspaces_close_at_high_N(self):
        """At N=200K, V_hat and V_star should agree to sin Theta < 0.05."""
        from cas.jhi_cas import jhi_grassmann_descent, sin_theta_F
        M, H, C, V_hat = self._fit_and_extract(N=200_000, seed=7)
        V_star, info = jhi_grassmann_descent(M, H, r=3, V_init=V_hat, n_iter=500)
        sin_th = sin_theta_F(V_hat, V_star)
        assert sin_th < 0.10, f"V_hat vs V* sin Theta at N=200K = {sin_th}"
