#!/usr/bin/env python3

import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A_COEF = 4e-3
B_COEF = -16e-3
C_COEF = 17e-3


def tunnel_diode_current(u: float) -> float:
    return A_COEF * u**3 + B_COEF * u**2 + C_COEF * u


def tunnel_diode_dI_du(u: float) -> float:
    return 3.0 * A_COEF * u**2 + 2.0 * B_COEF * u + C_COEF


def differential_resistance(u: float) -> float:
    g = tunnel_diode_dI_du(u)
    if abs(g) < 1e-30:
        return math.inf
    return 1.0 / g


def ndr_boundaries() -> tuple[float, float]:
    disc = (2.0 * B_COEF) ** 2 - 4.0 * (3.0 * A_COEF) * C_COEF
    if disc < 0:
        raise ValueError("NDR-участок отсутствует")
    root = math.sqrt(disc)
    u1 = (-2.0 * B_COEF - root) / (6.0 * A_COEF)
    u2 = (-2.0 * B_COEF + root) / (6.0 * A_COEF)
    return (u1, u2) if u1 < u2 else (u2, u1)


def ode_system(t, state, R, C, L, U0):
    v, i_l = state
    delta_i = tunnel_diode_current(v + U0) - tunnel_diode_current(U0)
    dv_dt = (1.0 / C) * (-v / R - i_l - delta_i)
    di_dt = v / L
    return [dv_dt, di_dt]


def integrate_rk45(R, C, L, U0, t_end, v0=None, iL0=0.0, n_points=10000):
    if v0 is None:
        v0 = 1e-3
    t_eval = np.linspace(0.0, t_end, n_points)
    T0 = 2.0 * math.pi * math.sqrt(L * C)
    sol = solve_ivp(
        ode_system,
        [0.0, t_end],
        [v0, iL0],
        args=(R, C, L, U0),
        method="RK45",
        t_eval=t_eval,
        max_step=T0 / 100.0,
        rtol=1e-8,
        atol=1e-10,
    )
    return sol.t, sol.y[0], sol.y[1]


def detect_oscillation(v_arr: np.ndarray, threshold_ratio: float = 0.05) -> bool:
    u1, u2 = ndr_boundaries()
    a_max = (u2 - u1) / 2.0
    half = len(v_arr) // 2
    amp = (np.max(v_arr[half:]) - np.min(v_arr[half:])) / 2.0
    return amp > threshold_ratio * a_max


def compute_thd(v_arr: np.ndarray, t_arr: np.ndarray):
    half = len(v_arr) // 2
    signal = v_arr[half:] - np.mean(v_arr[half:])
    n = len(signal)
    dt = (t_arr[-1] - t_arr[half]) / n
    spec = np.fft.rfft(signal) / n
    amps = 2.0 * np.abs(spec)
    if len(amps) > 0:
        amps[0] /= 2.0
    freqs = np.fft.rfftfreq(n, d=dt)
    idx1 = int(np.argmax(amps[1:]) + 1)
    a1 = amps[idx1]
    higher = amps[idx1 + 1: idx1 + 20]
    thd = math.sqrt(float(np.sum(higher ** 2))) / (a1 + 1e-30)
    return a1, thd, freqs, amps


def autogeneration_condition(R: float, U0: float) -> bool:
    rd = differential_resistance(U0)
    return (1.0 / R + 1.0 / rd) < 0.0


def validate_positive(value: float, name: str) -> float:
    if value <= 0:
        raise ValueError(f"{name} должно быть положительным, получено: {value}")
    return value


def validate_in_ndr(U0: float) -> float:
    u1, u2 = ndr_boundaries()
    if not (u1 < U0 < u2):
        raise ValueError(f"U0={U0:.3f} В вне NDR-участка [{u1:.3f}, {u2:.3f}] В")
    return U0


def parse_float(s: str, name: str) -> float:
    try:
        return float(s.strip())
    except ValueError:
        raise ValueError(f"Неверный формат числа для {name}: '{s}'")


def read_param(prompt: str, name: str, default: float, min_val: float = 1e-20) -> float:
    raw = input(prompt).strip()
    if raw == "":
        print(f"  -> используется значение по умолчанию: {default}")
        return default
    val = parse_float(raw, name)
    validate_positive(val, name)
    if val < min_val:
        raise ValueError(f"{name} слишком мало: {val}")
    return val


def scenario_single_trajectory():
    print("\n=== Сценарий 1: одиночная траектория ===")
    C_nF = read_param("C [нФ] (default 10): ", "C", 10.0)
    L_uH = read_param("L [мкГн] (default 100): ", "L", 100.0)
    R = read_param("R [Ом] (default 300): ", "R", 300.0)
    U0 = read_param("U0 [В] (default 1.3): ", "U0", 1.3)

    C = C_nF * 1e-9
    L = L_uH * 1e-6
    validate_in_ndr(U0)

    T0 = 2.0 * math.pi * math.sqrt(L * C)
    f0 = 1.0 / T0
    rd = differential_resistance(U0)
    Q = abs(rd) / (2.0 * math.pi * f0 * L)

    print(f"T0 = {T0 * 1e6:.3f} мкс")
    print(f"f0 = {f0 / 1e3:.3f} кГц")
    print(f"r_D(U0) = {rd:.2f} Ом")
    print(f"Q = {Q:.2f}")
    print("Условие автогенерации:", "выполнено" if autogeneration_condition(R, U0) else "не выполнено")

    t, v, i_l = integrate_rk45(R, C, L, U0, 150.0 * T0)
    u = v + U0
    osc = detect_oscillation(v)
    print("Автоколебания:", "да" if osc else "нет")
    if osc:
        a1, thd, _, _ = compute_thd(v, t)
        print(f"A1 = {a1 * 1e3:.3f} мВ")
        print(f"THD = {thd * 100:.2f}%")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    t_us = t * 1e6

    axes[0, 0].plot(t_us, u, lw=0.9)
    axes[0, 0].axhline(U0, color="r", ls="--", lw=1)
    axes[0, 0].set_title("Напряжение u(t)")
    axes[0, 0].set_xlabel("t, мкс")
    axes[0, 0].set_ylabel("u, В")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t_us, i_l * 1e3, lw=0.9, color="darkorange")
    axes[0, 1].set_title("Ток i_L(t)")
    axes[0, 1].set_xlabel("t, мкс")
    axes[0, 1].set_ylabel("i_L, мА")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(u, i_l * 1e3, lw=0.7, color="green")
    axes[1, 0].set_title("Фазовый портрет")
    axes[1, 0].set_xlabel("u, В")
    axes[1, 0].set_ylabel("i_L, мА")
    axes[1, 0].grid(True, alpha=0.3)

    u_grid = np.linspace(0.0, 2.5, 400)
    i_grid = np.array([tunnel_diode_current(x) for x in u_grid]) * 1e3
    u1, u2 = ndr_boundaries()
    axes[1, 1].plot(u_grid, i_grid, lw=2)
    axes[1, 1].axvspan(u1, u2, color="red", alpha=0.12)
    axes[1, 1].axvline(U0, color="green", ls="--")
    axes[1, 1].set_title("ВАХ диода")
    axes[1, 1].set_xlabel("U, В")
    axes[1, 1].set_ylabel("I, мА")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("scenario1_trajectory.png", dpi=140)
    plt.close()
    print("Сохранён график: scenario1_trajectory.png")


def scenario_autogeneration_map():
    print("\n=== Сценарий 2: карта режимов ===")
    C_nF = read_param("C [нФ] (default 10): ", "C", 10.0)
    L_uH = read_param("L [мкГн] (default 100): ", "L", 100.0)
    R_min = read_param("R_min [Ом] (default 50): ", "R_min", 50.0)
    R_max = read_param("R_max [Ом] (default 800): ", "R_max", 800.0)
    u1, u2 = ndr_boundaries()
    U0_min = read_param(f"U0_min [В] (default {u1 + 0.05:.2f}): ", "U0_min", u1 + 0.05)
    U0_max = read_param(f"U0_max [В] (default {u2 - 0.05:.2f}): ", "U0_max", u2 - 0.05)
    NR = int(read_param("Точек по R (default 15): ", "NR", 15, 3))
    NU = int(read_param("Точек по U0 (default 15): ", "NU", 15, 3))

    if R_max <= R_min:
        raise ValueError("R_max должно быть больше R_min")
    if U0_max <= U0_min:
        raise ValueError("U0_max должно быть больше U0_min")

    C = C_nF * 1e-9
    L = L_uH * 1e-6
    T0 = 2.0 * math.pi * math.sqrt(L * C)
    t_end = 80.0 * T0

    Rs = np.linspace(R_min, R_max, NR)
    U0s = np.linspace(U0_min, U0_max, NU)
    analytical = np.zeros((NU, NR), dtype=int)
    numerical = np.zeros((NU, NR), dtype=int)

    total = NR * NU
    done = 0
    for j, R in enumerate(Rs):
        for i, U0 in enumerate(U0s):
            analytical[i, j] = int(autogeneration_condition(R, U0))
            _, v, _ = integrate_rk45(R, C, L, U0, t_end, n_points=2500)
            numerical[i, j] = int(detect_oscillation(v))
            done += 1
            if done % 20 == 0:
                print(f"Обработано {done}/{total}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title in [
        (axes[0], analytical, "Аналитический критерий"),
        (axes[1], numerical, "Численное моделирование"),
    ]:
        im = ax.imshow(
            data,
            extent=[R_min, R_max, U0_min, U0_max],
            origin="lower",
            aspect="auto",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
        )
        ax.set_title(title)
        ax.set_xlabel("R, Ом")
        ax.set_ylabel("U0, В")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("scenario2_map.png", dpi=140)
    plt.close()
    print("Сохранён график: scenario2_map.png")


def scenario_spectrum_analysis():
    print("\n=== Сценарий 3: спектральный анализ ===")
    C_nF = read_param("C [нФ] (default 10): ", "C", 10.0)
    R = read_param("R [Ом] (default 300): ", "R", 300.0)
    U0 = read_param("U0 [В] (default 1.3): ", "U0", 1.3)
    NL = int(read_param("Число точек по L (default 10): ", "NL", 10, 3))
    validate_in_ndr(U0)

    C = C_nF * 1e-9
    Ls_uH = np.logspace(1, 4, NL)

    L_plot = []
    A1_plot = []
    THD_plot = []
    Q_plot = []

    for L_uH in Ls_uH:
        L = L_uH * 1e-6
        T0 = 2.0 * math.pi * math.sqrt(L * C)
        t, v, _ = integrate_rk45(R, C, L, U0, 100.0 * T0, n_points=8000)
        rd = differential_resistance(U0)
        f0 = 1.0 / T0
        Q = abs(rd) / (2.0 * math.pi * f0 * L)
        if detect_oscillation(v):
            A1, thd, _, _ = compute_thd(v, t)
        else:
            A1, thd = 0.0, 0.0
        L_plot.append(L_uH)
        A1_plot.append(A1 * 1e3)
        THD_plot.append(thd * 100.0)
        Q_plot.append(Q)
        print(f"L={L_uH:8.1f} мкГн | A1={A1*1e3:8.3f} мВ | THD={thd*100:7.2f}% | Q={Q:6.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].semilogx(L_plot, A1_plot, 'o-')
    axes[0].set_title("A1(L)")
    axes[0].set_xlabel("L, мкГн")
    axes[0].set_ylabel("A1, мВ")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(L_plot, THD_plot, 's-', color='crimson')
    axes[1].set_title("THD(L)")
    axes[1].set_xlabel("L, мкГн")
    axes[1].set_ylabel("THD, %")
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogx(L_plot, Q_plot, 'd-', color='green')
    axes[2].set_title("Q(L)")
    axes[2].set_xlabel("L, мкГн")
    axes[2].set_ylabel("Q")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("scenario3_spectrum.png", dpi=140)
    plt.close()
    print("Сохранён график: scenario3_spectrum.png")


MENU = """
================ Автогенератор на туннельном диоде ================
1 - Одиночная траектория и фазовый портрет
2 - Карта режимов автогенерации
3 - Спектральный анализ
0 - Выход
===================================================================
"""


def print_diode_info():
    u1, u2 = ndr_boundaries()
    print("Параметры модели туннельного диода:")
    print(f"I(u) = {A_COEF*1e3:.0f} мА/В^3 * u^3 + ({B_COEF*1e3:.0f}) мА/В^2 * u^2 + {C_COEF*1e3:.0f} мА/В * u")
    print(f"NDR-участок: U1 = {u1:.4f} В, U2 = {u2:.4f} В")
    print(f"Максимальная ориентировочная амплитуда: {(u2-u1)/2:.4f} В")
    print("Рекомендуемые параметры для генерации: R=300 Ом, C=10 нФ, L=100 мкГн, U0=1.3 В")


def main():
    print_diode_info()
    while True:
        print(MENU)
        choice = input("Выберите сценарий: ").strip()
        try:
            if choice == "0":
                print("Выход.")
                break
            elif choice == "1":
                scenario_single_trajectory()
            elif choice == "2":
                scenario_autogeneration_map()
            elif choice == "3":
                scenario_spectrum_analysis()
            else:
                print("Введите 0, 1, 2 или 3")
        except ValueError as e:
            print(f"Ошибка ввода: {e}")
        except KeyboardInterrupt:
            print("\nПрервано пользователем")
            break


if __name__ == "__main__":
    main()