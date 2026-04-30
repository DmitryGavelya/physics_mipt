import unittest
import numpy as np
from scipy.constants import mu_0, e, m_p
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from magnetic_mirror import MagneticMirror, ParticleSimulation

class TestMagneticMirror(unittest.TestCase):

    def setUp(self):
        self.trap = MagneticMirror(R=1.0, d=2.0, I=1e6)
        self.sim = ParticleSimulation(self.trap)

    def test_magnetic_field_on_axis(self):
        z_test = 0.5
        _, _, Bz_calc = self.trap.get_field(0.0, 0.0, z_test)

        Bz_analytical = (mu_0 * self.trap.I * self.trap.R ** 2 / (
                    2 * (self.trap.R ** 2 + (z_test - self.trap.d / 2) ** 2) ** 1.5) +
                         mu_0 * self.trap.I * self.trap.R ** 2 / (
                                     2 * (self.trap.R ** 2 + (z_test + self.trap.d / 2) ** 2) ** 1.5))

        self.assertAlmostEqual(Bz_calc, Bz_analytical, places=10,
                               msg="Расчет поля на оси z не совпадает с аналитическим.")

    def test_energy_conservation(self):
        v0 = 1e6
        Y0 = np.array([0.0, 0.0, 0.0, v0, 0.0, v0])
        B0_mag = self.trap.get_field_magnitude(0, 0, 0)

        T_L = 2 * np.pi * m_p / (e * B0_mag)
        dt = T_L / 100.0

        Y1 = self.sim.rk4_step(0, Y0, dt=dt)

        E0 = 0.5 * m_p * np.sum(Y0[3:] ** 2)
        E1 = 0.5 * m_p * np.sum(Y1[3:] ** 2)

        self.assertTrue(np.isclose(E0, E1, rtol=1e-6),
                        msg="Кинетическая энергия не сохраняется на шаге интегрирования.")

    def test_zero_field_far_away(self):
        B_mag = self.trap.get_field_magnitude(100.0, 100.0, 100.0)
        self.assertAlmostEqual(B_mag, 0.0, places=5,
                               msg="Поле на большом расстоянии должно стремиться к нулю.")


if __name__ == '__main__':
    unittest.main()