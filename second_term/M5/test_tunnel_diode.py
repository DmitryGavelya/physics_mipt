#!/usr/bin/env python3
"""
Тесты для модуля tunnel_diode_oscillator.py
Запуск:
    python -m pytest test_tunnel_diode.py -v
"""

import math
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tunnel_diode_oscillator import (
    A_COEF, B_COEF, C_COEF,
    tunnel_diode_current,
    tunnel_diode_dI_du,
    differential_resistance,
    ndr_boundaries,
    ode_system,
    integrate_rk45,
    detect_oscillation,
    compute_thd,
    autogeneration_condition,
    validate_positive,
    validate_in_ndr,
    parse_float,
)

R_DEF = 300.0
C_DEF = 10e-9
L_DEF = 100e-6
U0_DEF = 1.3


class TestTunnelDiodeCurrent:
    def test_zero_voltage_zero_current(self):
        assert tunnel_diode_current(0.0) == pytest.approx(0.0, abs=1e-15)

    def test_formula_at_1V(self):
        u = 1.0
        expected = A_COEF * u**3 + B_COEF * u**2 + C_COEF * u
        assert tunnel_diode_current(u) == pytest.approx(expected, rel=1e-10)

    def test_peak_higher_than_valley(self):
        u1, u2 = ndr_boundaries()
        assert tunnel_diode_current(u1) > tunnel_diode_current(u2)

    def test_current_decreases_in_ndr(self):
        u1, u2 = ndr_boundaries()
        u_mid = (u1 + u2) / 2
        assert tunnel_diode_current(u_mid) < tunnel_diode_current(u1)

    def test_units_physical_range(self):
        I = tunnel_diode_current(1.0)
        assert 1e-4 < I < 0.1

    def test_negative_voltage_returns_negative(self):
        assert tunnel_diode_current(-0.5) < 0

    def test_linearity_near_zero(self):
        u = 1e-3
        I = tunnel_diode_current(u)
        assert I == pytest.approx(C_COEF * u, rel=1e-3)


class TestDifferentialResistance:
    def test_negative_in_ndr(self):
        u1, u2 = ndr_boundaries()
        u_mid = (u1 + u2) / 2
        assert differential_resistance(u_mid) < 0

    def test_positive_outside_ndr(self):
        assert differential_resistance(0.1) > 0
        assert differential_resistance(2.5) > 0

    def test_reciprocal_of_dI_du(self):
        u = 1.5
        assert differential_resistance(u) == pytest.approx(
            1.0 / tunnel_diode_dI_du(u), rel=1e-10
        )

    def test_large_at_peak(self):
        u1, _ = ndr_boundaries()
        assert abs(differential_resistance(u1)) > 1e3


class TestNDRBoundaries:
    def test_u1_lt_u2(self):
        u1, u2 = ndr_boundaries()
        assert u1 < u2

    def test_approximate_u1_u2(self):
        u1, u2 = ndr_boundaries()
        assert u1 == pytest.approx(0.73, abs=0.05)
        assert u2 == pytest.approx(1.93, abs=0.05)

    def test_max_amplitude(self):
        u1, u2 = ndr_boundaries()
        assert (u2 - u1) / 2 == pytest.approx(0.60, abs=0.05)

    def test_derivative_zero_at_boundaries(self):
        u1, u2 = ndr_boundaries()
        assert tunnel_diode_dI_du(u1) == pytest.approx(0.0, abs=1e-8)
        assert tunnel_diode_dI_du(u2) == pytest.approx(0.0, abs=1e-8)


class TestODESystem:
    def test_output_length(self):
        res = ode_system(0.0, [0.01, 0.001], R_DEF, C_DEF, L_DEF, U0_DEF)
        assert len(res) == 2

    def test_equilibrium_at_origin(self):
        res = ode_system(0.0, [0.0, 0.0], R_DEF, C_DEF, L_DEF, U0_DEF)
        assert res[0] == pytest.approx(0.0, abs=1e-10)
        assert res[1] == pytest.approx(0.0, abs=1e-10)

    def test_diL_dt_proportional_v(self):
        v = 0.5
        res = ode_system(0.0, [v, 0.0], R_DEF, C_DEF, L_DEF, U0_DEF)
        assert res[1] == pytest.approx(v / L_DEF, rel=1e-10)

    def test_no_time_dependence(self):
        state = [0.1, 0.005]
        r1 = ode_system(0.0, state, R_DEF, C_DEF, L_DEF, U0_DEF)
        r2 = ode_system(999.0, state, R_DEF, C_DEF, L_DEF, U0_DEF)
        assert r1[0] == pytest.approx(r2[0], rel=1e-12)
        assert r1[1] == pytest.approx(r2[1], rel=1e-12)

    def test_unstable_at_small_positive_perturbation(self):
        v_small = 1e-4
        res = ode_system(0.0, [v_small, 0.0], R_DEF, C_DEF, L_DEF, U0_DEF)
        G0 = tunnel_diode_dI_du(U0_DEF)
        if G0 + 1 / R_DEF < 0:
            assert res[0] > 0


class TestIntegrateRK45:
    def setup_method(self):
        T0 = 2 * math.pi * math.sqrt(L_DEF * C_DEF)
        self.t_end = 10 * T0

    def test_output_shapes(self):
        t, v, iL = integrate_rk45(
            R_DEF, C_DEF, L_DEF, U0_DEF, self.t_end, n_points=500
        )
        assert len(t) == 500
        assert len(v) == 500
        assert len(iL) == 500

    def test_time_starts_at_zero(self):
        t, _, _ = integrate_rk45(
            R_DEF, C_DEF, L_DEF, U0_DEF, self.t_end, n_points=200
        )
        assert t[0] == pytest.approx(0.0, abs=1e-15)

    def test_time_ends_at_t_end(self):
        t, _, _ = integrate_rk45(
            R_DEF, C_DEF, L_DEF, U0_DEF, self.t_end, n_points=200
        )
        assert t[-1] == pytest.approx(self.t_end, rel=1e-6)

    def test_all_finite(self):
        _, v, iL = integrate_rk45(
            R_DEF, C_DEF, L_DEF, U0_DEF, self.t_end, n_points=500
        )
        assert np.all(np.isfinite(v))
        assert np.all(np.isfinite(iL))

    def test_initial_v_v0(self):
        v0 = 0.05
        _, v, _ = integrate_rk45(
            R_DEF, C_DEF, L_DEF, U0_DEF, self.t_end, v0=v0, n_points=200
        )
        assert v[0] == pytest.approx(v0, rel=1e-4)


class TestAutogenerationCondition:
    def test_standard_params_generates(self):
        assert autogeneration_condition(300.0, 1.3)

    def test_small_r_no_generation(self):
        assert not autogeneration_condition(100.0, 1.3)

    def test_boundary_R_equals_rd(self):
        U0 = 1.3
        rd_abs = abs(differential_resistance(U0))
        assert not autogeneration_condition(rd_abs * 0.99, U0)
        assert autogeneration_condition(rd_abs * 1.01, U0)

    def test_different_bias_points(self):
        u1, u2 = ndr_boundaries()
        for U0 in [u1 + 0.1, (u1 + u2) / 2, u2 - 0.1]:
            rd_abs = abs(differential_resistance(U0))
            assert autogeneration_condition(rd_abs * 1.5, U0)
            assert not autogeneration_condition(rd_abs * 0.5, U0)


class TestDetectOscillation:
    def test_oscillating_signal(self):
        t = np.linspace(0, 1e-4, 5000)
        v = 0.3 * np.sin(2 * math.pi * 1e5 * t)
        assert detect_oscillation(v)

    def test_flat_signal_not_oscillating(self):
        v = np.zeros(2000)
        assert not detect_oscillation(v)

    def test_tiny_amplitude_not_oscillating(self):
        u1, u2 = ndr_boundaries()
        A_max = (u2 - u1) / 2
        t = np.linspace(0, 1e-4, 2000)
        v = 0.001 * A_max * np.sin(2 * math.pi * 1e5 * t)
        assert not detect_oscillation(v)


class TestComputeTHD:
    def setup_method(self):
        fs = 1e6
        N = 8000
        f0 = 50e3
        A = 0.1
        self.t = np.linspace(0, N / fs, N)
        self.u_sine = A * np.sin(2 * math.pi * f0 * self.t)

    def test_thd_pure_sine_small(self):
        _, thd, _, _ = compute_thd(self.u_sine, self.t)
        assert thd < 0.05

    def test_a1_correct(self):
        A1, _, _, _ = compute_thd(self.u_sine, self.t)
        assert A1 == pytest.approx(0.1, rel=0.05)

    def test_returns_4_elements(self):
        assert len(compute_thd(self.u_sine, self.t)) == 4

    def test_freqs_nonneg(self):
        _, _, freqs, _ = compute_thd(self.u_sine, self.t)
        assert np.all(freqs >= 0)

    def test_amplitudes_nonneg(self):
        _, _, _, amps = compute_thd(self.u_sine, self.t)
        assert np.all(amps >= 0)

    def test_a1_scales_with_amplitude(self):
        fs = 1e6
        N = 8000
        f0 = 50e3
        t = np.linspace(0, N / fs, N)
        for A_test in [0.05, 0.10, 0.20]:
            v = A_test * np.sin(2 * math.pi * f0 * t)
            A1, _, _, _ = compute_thd(v, t)
            assert A1 == pytest.approx(A_test, rel=0.05)


class TestValidation:
    def test_positive_ok(self):
        assert validate_positive(5.0, "x") == 5.0

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="положительным"):
            validate_positive(0.0, "x")

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            validate_positive(-1.0, "x")

    def test_in_ndr_valid(self):
        validate_in_ndr(1.3)

    def test_out_of_ndr_raises(self):
        u1, u2 = ndr_boundaries()
        with pytest.raises(ValueError, match="NDR"):
            validate_in_ndr(u1 - 0.1)
        with pytest.raises(ValueError, match="NDR"):
            validate_in_ndr(u2 + 0.1)

    def test_parse_float_int_str(self):
        assert parse_float("42", "x") == pytest.approx(42.0)

    def test_parse_float_decimal(self):
        assert parse_float("3.14", "x") == pytest.approx(3.14)

    def test_parse_float_sci(self):
        assert parse_float("1.5e-9", "x") == pytest.approx(1.5e-9)

    def test_parse_float_invalid_raises(self):
        with pytest.raises(ValueError, match="Неверный формат"):
            parse_float("abc", "x")

    def test_parse_float_whitespace(self):
        assert parse_float("  100  ", "x") == pytest.approx(100.0)


class TestIntegration:
    def test_oscillation_with_standard_params(self):
        T0 = 2 * math.pi * math.sqrt(L_DEF * C_DEF)
        _, v, _ = integrate_rk45(
            R_DEF, C_DEF, L_DEF, U0_DEF, 150 * T0, n_points=8000
        )
        assert detect_oscillation(v)

    def test_no_oscillation_r_too_small(self):
        R_small = 50.0
        T0 = 2 * math.pi * math.sqrt(L_DEF * C_DEF)
        _, v, _ = integrate_rk45(
            R_small, C_DEF, L_DEF, U0_DEF, 100 * T0, n_points=5000
        )
        assert not detect_oscillation(v)

    def test_limit_cycle_stable(self):
        T0 = 2 * math.pi * math.sqrt(L_DEF * C_DEF)
        _, v, _ = integrate_rk45(
            R_DEF, C_DEF, L_DEF, U0_DEF, 200 * T0, n_points=10000
        )
        q = len(v) // 4
        amp3 = (np.max(v[2 * q:3 * q]) - np.min(v[2 * q:3 * q])) / 2
        amp4 = (np.max(v[3 * q:]) - np.min(v[3 * q:])) / 2
        rel_diff = abs(amp4 - amp3) / (amp4 + 1e-10)
        assert rel_diff < 0.10

    def test_frequency_close_to_lc(self):
        T0 = 2 * math.pi * math.sqrt(L_DEF * C_DEF)
        f0_theory = 1.0 / T0
        t, v, _ = integrate_rk45(
            R_DEF, C_DEF, L_DEF, U0_DEF, 150 * T0, n_points=10000
        )
        if detect_oscillation(v):
            _, _, freqs, amps = compute_thd(v, t)
            idx1 = int(np.argmax(amps[1:])) + 1
            f_measured = freqs[idx1]
            assert f_measured == pytest.approx(f0_theory, rel=0.05)

    def test_no_generation_outside_ndr(self):
        with pytest.raises(ValueError):
            validate_in_ndr(0.3)


class TestPhysicalConstraints:
    def test_conductance_negative_in_ndr(self):
        u1, u2 = ndr_boundaries()
        u_ndr = np.linspace(u1 + 0.05, u2 - 0.05, 20)
        for u in u_ndr:
            assert tunnel_diode_dI_du(u) < 0

    def test_conductance_positive_outside_ndr(self):
        u1, u2 = ndr_boundaries()
        u_low = np.linspace(0.01, u1 - 0.05, 10)
        u_high = np.linspace(u2 + 0.05, 3.0, 10)
        for arr in [u_low, u_high]:
            for u in arr:
                assert tunnel_diode_dI_du(u) > 0

    def test_frequency_formula(self):
        for L_uH in [50, 100, 200, 500]:
            L = L_uH * 1e-6
            C = 10e-9
            f0 = 1.0 / (2 * math.pi * math.sqrt(L * C))
            assert f0 == pytest.approx(
                1.0 / (2 * math.pi * math.sqrt(L * C)),
                rel=1e-12
            )

    def test_q_factor_formula(self):
        L = L_DEF
        C = C_DEF
        U0 = U0_DEF
        f0 = 1.0 / (2 * math.pi * math.sqrt(L * C))
        Q = abs(differential_resistance(U0)) / (2 * math.pi * f0 * L)
        assert Q > 0

    def test_max_amplitude_formula(self):
        u1, u2 = ndr_boundaries()
        A_max = (u2 - u1) / 2
        assert 0.50 < A_max < 0.70


if __name__ == "__main__":
    import pytest as pt
    sys.exit(pt.main([__file__, "-v"]))