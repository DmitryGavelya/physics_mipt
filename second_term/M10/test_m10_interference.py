#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тесты для программы M10. Интерференция.

Запуск:
    python3 -m unittest -v test_m10_interference.py

Тесты используют только стандартный модуль unittest, поэтому pytest не нужен.
Проверяются расчётные функции, а не интерактивное меню.
"""

import math
import unittest

import numpy as np

import m10_interference as m10


class TestYoungInterference(unittest.TestCase):
    def setUp(self):
        self.params = m10.YoungParams(
            wavelength=550e-9,
            slit_distance=0.5e-3,
            source_to_slits=1.0,
            slits_to_screen=2.0,
            screen_half_width=10e-3,
            screen_points=2001,
        )

    def test_fringe_period_default_parameters(self):
        """Период полос должен равняться lambda * L2 / d."""

        expected = 550e-9 * 2.0 / 0.5e-3
        actual = m10.fringe_period(self.params)
        self.assertAlmostEqual(actual, expected, delta=1e-15)
        self.assertAlmostEqual(actual, 2.2e-3, delta=1e-15)

    def test_screen_grid_is_symmetric(self):
        """Сетка экрана должна быть симметрична относительно нуля."""

        x = m10.screen_grid(self.params)
        self.assertEqual(len(x), self.params.screen_points)
        self.assertAlmostEqual(x[0], -self.params.screen_half_width)
        self.assertAlmostEqual(x[-1], self.params.screen_half_width)
        self.assertAlmostEqual(x[len(x) // 2], 0.0, delta=1e-18)
        np.testing.assert_allclose(x, -x[::-1], atol=1e-18)

    def test_path_difference_zero_at_center(self):
        """В центре экрана разность хода от двух симметричных щелей равна нулю."""

        x = np.array([0.0])
        exact = m10.path_difference_exact(x, self.params)[0]
        paraxial = m10.path_difference_paraxial(x, self.params)[0]
        self.assertAlmostEqual(exact, 0.0, delta=1e-18)
        self.assertAlmostEqual(paraxial, 0.0, delta=1e-18)

    def test_path_difference_is_odd_function(self):
        """Для симметричной схемы Delta(-x) = -Delta(x)."""

        x = np.linspace(-5e-3, 5e-3, 101)
        exact = m10.path_difference_exact(x, self.params)
        paraxial = m10.path_difference_paraxial(x, self.params)
        np.testing.assert_allclose(exact, -exact[::-1], atol=1e-18)
        np.testing.assert_allclose(paraxial, -paraxial[::-1], atol=1e-18)

    def test_analytic_point_intensity_has_unit_visibility(self):
        """Для двух равных когерентных источников видность должна быть близка к 1."""

        x = m10.screen_grid(self.params)
        intensity = m10.analytic_point_intensity(x, self.params, normalize=True)
        visibility = m10.michelson_visibility(intensity)
        self.assertAlmostEqual(float(np.max(intensity)), 1.0, delta=1e-12)
        self.assertAlmostEqual(visibility, 1.0, delta=1e-9)

    def test_numerical_point_intensity_matches_analytic(self):
        """Численное суммирование должно совпадать с аналитикой в параксиальном режиме."""

        x = m10.screen_grid(self.params)
        analytic = m10.analytic_point_intensity(x, self.params, normalize=True)
        numerical = m10.numerical_point_intensity(x, self.params, normalize=True)
        max_error = float(np.max(np.abs(analytic - numerical)))
        self.assertLess(max_error, 5e-4)

    def test_spatial_visibility_special_values(self):
        """Проверка V(b) в точках b=0, b=b0/2 и b=b0."""

        b0 = (
            self.params.wavelength
            * self.params.source_to_slits
            / self.params.slit_distance
        )

        self.assertAlmostEqual(
            float(m10.spatial_visibility_analytic(0.0, self.params)),
            1.0,
            delta=1e-15,
        )
        self.assertAlmostEqual(
            float(m10.spatial_visibility_analytic(b0 / 2.0, self.params)),
            2.0 / math.pi,
            delta=1e-12,
        )
        self.assertAlmostEqual(
            float(m10.spatial_visibility_analytic(b0, self.params)),
            0.0,
            delta=1e-15,
        )

    def test_extended_source_analytic_visibility_matches_formula(self):
        """Видность аналитической кривой должна совпадать с sinc-формулой."""

        # Берём очень широкий экран, чтобы в сетку попали максимум и минимум.
        params = m10.YoungParams(
            wavelength=550e-9,
            slit_distance=0.5e-3,
            source_to_slits=1.0,
            slits_to_screen=2.0,
            screen_half_width=22e-3,
            screen_points=20001,
        )
        x = m10.screen_grid(params)
        b0 = params.wavelength * params.source_to_slits / params.slit_distance
        b = b0 / 2.0

        intensity = m10.extended_source_intensity_analytic(x, params, b, normalize=False)
        expected = float(m10.spatial_visibility_analytic(b, params))
        actual = m10.michelson_visibility(intensity)
        self.assertAlmostEqual(actual, expected, delta=2e-3)

    def test_spectral_visibility_special_values(self):
        """Проверка спектральной видности: V=1 при delta_lambda=0 и нуль при аргументе 1."""

        lambda0 = self.params.wavelength
        optical_delta = 10.0 * lambda0

        self.assertAlmostEqual(
            float(m10.spectral_visibility_analytic(0.0, optical_delta, lambda0)),
            1.0,
            delta=1e-15,
        )

        delta_lambda_for_first_zero = lambda0**2 / optical_delta
        self.assertAlmostEqual(
            float(
                m10.spectral_visibility_analytic(
                    delta_lambda_for_first_zero,
                    optical_delta,
                    lambda0,
                )
            ),
            0.0,
            delta=1e-15,
        )

    def test_spectral_visibility_by_order_first_zero(self):
        """Функция V(m) должна обращаться в нуль при m * delta_lambda / lambda0 = 1."""

        lambda0 = self.params.wavelength
        delta_lambda = 5e-9
        first_zero_order = lambda0 / delta_lambda

        self.assertAlmostEqual(
            float(m10.spectral_visibility_by_order(0.0, delta_lambda, lambda0)),
            1.0,
            delta=1e-15,
        )
        self.assertAlmostEqual(
            float(
                m10.spectral_visibility_by_order(
                    first_zero_order,
                    delta_lambda,
                    lambda0,
                )
            ),
            0.0,
            delta=1e-15,
        )

    def test_quasimonochromatic_rejects_too_wide_spectrum(self):
        """Спектр не должен содержать неположительные длины волн."""

        x = m10.screen_grid(self.params)
        with self.assertRaises(ValueError):
            m10.quasimonochromatic_intensity(
                x,
                self.params,
                source_size=0.0,
                delta_lambda=2.0 * self.params.wavelength,
                source_points=1,
                wavelength_points=3,
            )

    def test_quasimonochromatic_delta_zero_equals_point_source(self):
        """При b=0 и delta_lambda=0 результат должен совпадать с точечным источником."""

        # Уменьшаем сетку, чтобы тест был быстрым.
        params = m10.YoungParams(
            wavelength=550e-9,
            slit_distance=0.5e-3,
            source_to_slits=1.0,
            slits_to_screen=2.0,
            screen_half_width=5e-3,
            screen_points=501,
        )
        x = m10.screen_grid(params)
        point = m10.numerical_point_intensity(x, params, normalize=True)
        quasi = m10.quasimonochromatic_intensity(
            x,
            params,
            source_size=0.0,
            delta_lambda=0.0,
            source_points=1,
            wavelength_points=1,
            normalize=True,
        )
        np.testing.assert_allclose(quasi, point, atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
