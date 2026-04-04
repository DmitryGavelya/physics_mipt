import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe
from scipy.constants import mu_0, e, m_p


class MagneticMirror:
    def __init__(self, R=1.0, d=2.0, I=1e6):
        self.R = R
        self.d = d
        self.I = I

    def _ring_B(self, x, y, z):
        rho = np.hypot(x, y)
        mask = rho < 1e-10
        rho_safe = np.where(mask, 1e-10, rho)

        alpha2 = (self.R + rho_safe) ** 2 + z ** 2
        beta2 = (self.R - rho_safe) ** 2 + z ** 2
        k2 = 4 * self.R * rho_safe / alpha2

        K = ellipk(k2)
        E = ellipe(k2)

        factor = mu_0 * self.I / (2 * np.pi * np.sqrt(alpha2))
        B_z = factor * (K + (self.R ** 2 - rho_safe ** 2 - z ** 2) / beta2 * E)
        B_rho = factor * (z / rho_safe) * (((self.R ** 2 + rho_safe ** 2 + z ** 2) / beta2) * E - K)

        B_z_axis = mu_0 * self.I * self.R ** 2 / (2 * (self.R ** 2 + z ** 2) ** 1.5)
        B_z = np.where(mask, B_z_axis, B_z)
        B_rho = np.where(mask, 0.0, B_rho)

        B_x = B_rho * (x / rho_safe)
        B_y = B_rho * (y / rho_safe)

        return np.where(mask, 0.0, B_x), np.where(mask, 0.0, B_y), B_z

    def get_field(self, x, y, z):
        Bx1, By1, Bz1 = self._ring_B(x, y, z - self.d / 2)
        Bx2, By2, Bz2 = self._ring_B(x, y, z + self.d / 2)
        return Bx1 + Bx2, By1 + By2, Bz1 + Bz2

    def get_field_magnitude(self, x, y, z):
        Bx, By, Bz = self.get_field(x, y, z)
        return np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)


class ParticleSimulation:
    def __init__(self, trap, q=e, m=m_p):
        self.trap = trap
        self.q = q
        self.m = m

    def derivatives(self, t, Y):
        x, y, z, vx, vy, vz = Y
        Bx, By, Bz = self.trap.get_field(x, y, z)
        ax = (self.q / self.m) * (vy * Bz - vz * By)
        ay = (self.q / self.m) * (vz * Bx - vx * Bz)
        az = (self.q / self.m) * (vx * By - vy * Bx)
        return np.array([vx, vy, vz, ax, ay, az])

    def rk4_step(self, t, Y, dt):
        k1 = self.derivatives(t, Y)
        k2 = self.derivatives(t + dt / 2, Y + k1 * dt / 2)
        k3 = self.derivatives(t + dt / 2, Y + k2 * dt / 2)
        k4 = self.derivatives(t + dt, Y + k3 * dt)
        return Y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def run_scenarios():
    trap = MagneticMirror(R=1.0, d=2.0, I=5e6)
    sim = ParticleSimulation(trap)

    B_min = trap.get_field_magnitude(0, 0, 0)
    B_max = trap.get_field_magnitude(0, 0, trap.d / 2)
    alpha_loss = np.arcsin(np.sqrt(B_min / B_max))

    v_total = 1.5e6
    T_L = 2 * np.pi * m_p / (e * B_min)
    dt = T_L / 50.0
    t_max = 500 * T_L
    steps = int(t_max / dt)

    alpha_0 = alpha_loss + np.radians(10)
    v_perp0 = v_total * np.sin(alpha_0)
    v_z0 = v_total * np.cos(alpha_0)

    Y = np.array([0.0, 0.0, 0.0, v_perp0, 0.0, v_z0])
    history = np.zeros((steps, 6))
    times = np.linspace(0, t_max, steps)

    for i in range(steps):
        history[i] = Y
        Y = sim.rk4_step(times[i], Y, dt)

    x_h, y_h, z_h, vx_h, vy_h, vz_h = history.T
    v_perp_h = np.sqrt(vx_h ** 2 + vy_h ** 2)
    v_mag_h = np.sqrt(vx_h ** 2 + vy_h ** 2 + vz_h ** 2)
    B_mags = trap.get_field_magnitude(x_h, y_h, z_h)

    mu_h = 0.5 * m_p * v_perp_h ** 2 / B_mags
    E_h = 0.5 * m_p * v_mag_h ** 2

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(x_h, y_h, z_h, lw=0.8, color='b')

    theta = np.linspace(0, 2 * np.pi, 100)
    ring_x = trap.R * np.cos(theta)
    ring_y = trap.R * np.sin(theta)
    ax1.plot(ring_x, ring_y, trap.d / 2, color='r', lw=3)
    ax1.plot(ring_x, ring_y, -trap.d / 2, color='r', lw=3)
    ax1.set_zlim(-trap.d, trap.d)

    ax2 = fig.add_subplot(122)
    ax2.plot(times / T_L, mu_h / mu_h[0])
    ax2.plot(times / T_L, E_h / E_h[0])
    plt.tight_layout()
    plt.show()

    N = 500
    alphas = np.linspace(0, np.pi / 2, N)
    Y_batch = np.zeros((6, N))
    Y_batch[3] = v_total * np.sin(alphas)
    Y_batch[5] = v_total * np.cos(alphas)

    escaped = np.zeros(N, dtype=bool)
    z_max = trap.d / 2 + 0.5

    for _ in range(int(200 * T_L / dt)):
        Y_batch = sim.rk4_step(0, Y_batch, dt)
        escaped |= (np.abs(Y_batch[2]) > z_max)

    trapped = ~escaped

    plt.figure(figsize=(8, 4))
    plt.scatter(np.degrees(alphas[trapped]), np.ones(np.sum(trapped)), c='g', alpha=0.5, marker='|', s=200)
    plt.scatter(np.degrees(alphas[escaped]), np.zeros(np.sum(escaped)), c='r', alpha=0.5, marker='|', s=200)
    plt.axvline(np.degrees(alpha_loss), color='k', linestyle='--')
    plt.yticks([0, 1])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_scenarios()