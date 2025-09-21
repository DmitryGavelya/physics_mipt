import numpy as np
from scipy.integrate import solve_ivp

from src.phys_m1 import g, stone_equations, hit_ground


def test_no_resistance():

    m = 1.0
    v0 = 10.0
    angle = 45.0
    k = 0.0

    alpha_rad = np.radians(angle)
    v0x = v0 * np.cos(alpha_rad)
    v0y = v0 * np.sin(alpha_rad)

    T_analytical = 2 * v0y / g
    L_analytical = v0x * T_analytical
    H_analytical = v0y ** 2 / (2 * g)

    sol = solve_ivp(stone_equations, (0, 100), [0, 0, v0x, v0y],
                    args=('none', k, m), events=hit_ground,
                    rtol=1e-8, atol=1e-10, dense_output=True)

    T_numerical = sol.t_events[0][0]
    L_numerical = sol.sol(T_numerical)[0]
    H_numerical = np.max(sol.sol(np.linspace(0, T_numerical, 100))[1])

    tolerance = 0.01
    assert abs(T_numerical - T_analytical) / T_analytical < tolerance
    assert abs(L_numerical - L_analytical) / L_analytical < tolerance
    assert abs(H_numerical - H_analytical) / H_analytical < tolerance


def test_viscous_resistance():

    m = 1.0
    v0 = 10.0
    angle = 45.0
    k = 0.1

    alpha_rad = np.radians(angle)
    v0x = v0 * np.cos(alpha_rad)
    v0y = v0 * np.sin(alpha_rad)

    # Аналитическое решение для вязкого трения
    def viscous_solution(t):
        k_m = k / m
        vx = v0x * np.exp(-k_m * t)
        vy = (v0y + m * g / k) * np.exp(-k_m * t) - m * g / k
        x = (m / k) * v0x * (1 - np.exp(-k_m * t))
        y = (m / k) * (v0y + m * g / k) * (1 - np.exp(-k_m * t)) - (m * g / k) * t
        return x, y, vx, vy

    sol = solve_ivp(stone_equations, (0, 100), [0, 0, v0x, v0y],
                    args=('viscous', k, m), events=hit_ground,
                    rtol=1e-8, atol=1e-10, dense_output=True)

    T_numerical = sol.t_events[0][0]

    t_values = np.linspace(0, T_numerical * 0.9, 5)
    max_error = 0

    for t in t_values:
        state_num = sol.sol(t)
        x_num, y_num, vx_num, vy_num = state_num
        x_anal, y_anal, vx_anal, vy_anal = viscous_solution(t)

        error_vx = abs(vx_num - vx_anal) / (abs(vx_anal) + 1e-10)
        error_vy = abs(vy_num - vy_anal) / (abs(vy_anal) + 1e-10)
        error_x = abs(x_num - x_anal) / (abs(x_anal) + 1e-10)
        error_y = abs(y_num - y_anal) / (abs(y_anal) + 1e-10)

        max_error = max(max_error, error_vx, error_vy, error_x, error_y)
        tolerance = 1e-2
        assert max_error < tolerance, f"Слишком большая ошибка: {max_error:.4f}"


def main():
    test_viscous_resistance()
    test_no_resistance()
    print("Success")


