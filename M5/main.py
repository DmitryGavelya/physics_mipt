from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional
import matplotlib.colors as mcolors


@dataclass
class Pendulum:
    """Физический маятник - твердое тело произвольной формы"""
    
    mass: float
    length: float
    inertia: float
    damping: float = 0.0

    @property
    def small_angle_period(self) -> float:
        """Период малых колебаний (теоретический)"""
        
        return 2 * np.pi * np.sqrt(self.inertia / (self.mass * 9.81 * self.length))


class PendulumSimulator:
    """Симулятор физического маятника"""

    def __init__(self, pendulum: Pendulum):
        self.pendulum = pendulum

    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """ Система дифференциальных уравнений маятника
            I * d²θ/dt² = -m*g*l*sin(θ) - b*dθ/dt """
        
        theta, omega = y
        g = 9.81

        alpha = (-self.pendulum.mass * g * self.pendulum.length * np.sin(theta)
                 - self.pendulum.damping * omega) / self.pendulum.inertia

        return np.array([omega, alpha])

    def energy(self, theta: float, omega: float) -> float:
        """Полная механическая энергия"""
        
        potential = self.pendulum.mass * 9.81 * self.pendulum.length * (1 - np.cos(theta))
        kinetic = 0.5 * self.pendulum.inertia * omega ** 2
        return potential + kinetic

    def simulate_euler(self, theta0: float, omega0: float,
                       t_span: Tuple[float, float], dt: float = 0.01) -> PendulumResult:
        """Моделирование методом Эйлера"""
        
        t0, t_end = t_span
        n_steps = int((t_end - t0) / dt) + 1
        t_values = np.linspace(t0, t_end, n_steps)

        theta = np.zeros(n_steps)
        omega = np.zeros(n_steps)
        energy_values = np.zeros(n_steps)

        theta[0] = theta0
        omega[0] = omega0
        energy_values[0] = self.energy(theta0, omega0)

        for i in range(1, n_steps):
            deriv = self.derivatives(t_values[i - 1], np.array([theta[i - 1], omega[i - 1]]))
            theta[i] = theta[i - 1] + dt * deriv[0]
            omega[i] = omega[i - 1] + dt * deriv[1]
            energy_values[i] = self.energy(theta[i], omega[i])

        return PendulumResult(t_values, theta, omega, energy_values, self.pendulum)


class PendulumResult:
    """Результаты моделирования маятника"""

    def __init__(self, t: np.ndarray, theta: np.ndarray, omega: np.ndarray,
                 energy: np.ndarray, pendulum: Pendulum):
        self.t = t
        self.theta = theta
        self.omega = omega
        self.energy = energy
        self.pendulum = pendulum

    def find_periods(self) -> List[float]:
        """Нахождение периодов колебаний по нулевым пересечениям"""
        
        periods = []
        zero_crossings = []

        for i in range(1, len(self.theta)):
            if self.theta[i - 1] <= 0 and self.theta[i] > 0:
                t_cross = self.t[i - 1] - self.theta[i - 1] * (self.t[i] - self.t[i - 1]) / (
                            self.theta[i] - self.theta[i - 1])
                zero_crossings.append(t_cross)

        for i in range(1, len(zero_crossings)):
            periods.append(zero_crossings[i] - zero_crossings[i - 1])

        return periods

    def analyze_period_vs_amplitude(self) -> Tuple[np.ndarray, np.ndarray]:
        """Анализ зависимости периода от амплитуды"""
        
        periods = self.find_periods()
        if not periods:
            return np.array([]), np.array([])

        amplitudes = []
        zero_indices = []

        for i in range(1, len(self.theta)):
            if self.theta[i - 1] <= 0 and self.theta[i] > 0:
                zero_indices.append(i)

        for i in range(1, len(zero_indices)):
            start_idx = zero_indices[i - 1]
            end_idx = zero_indices[i]
            half_oscillation = self.theta[start_idx:end_idx]
            amplitude = np.max(np.abs(half_oscillation))
            amplitudes.append(amplitude)

        min_len = min(len(periods), len(amplitudes))
        return np.array(periods[:min_len]), np.array(amplitudes[:min_len])


def run_autotests():
    """Автотесты физической корректности"""
    
    print("Запуск автотестов...")

    pendulum = Pendulum(mass=1.0, length=1.0, inertia=1.0, damping=0.0)
    simulator = PendulumSimulator(pendulum)
    result = simulator.simulate_euler(theta0=0.5, omega0=0.0, t_span=(0, 5), dt=0.01)

    energy_drift = (np.max(result.energy) - np.min(result.energy)) / np.mean(result.energy)
    if energy_drift < 0.01:
        print("Тест 1 пройден: энергия сохраняется без трения")
    else:
        print(f"Тест 1 не пройден: дрейф энергии {energy_drift:.4f}")

    small_angle_result = simulator.simulate_euler(theta0=0.1, omega0=0.0, t_span=(0, 10), dt=0.01)
    periods = small_angle_result.find_periods()
    if periods:
        avg_period = np.mean(periods[:3])
        theoretical_period = pendulum.small_angle_period
        period_error = abs(avg_period - theoretical_period) / theoretical_period

        if period_error < 0.05:
            print(f"Тест 2 пройден: период {avg_period:.3f} с близок к теоретическому {theoretical_period:.3f} с")
        else:
            print(f"Тест 2 не пройден: ошибка периода {period_error:.3f}")

    damped_pendulum = Pendulum(mass=1.0, length=1.0, inertia=1.0, damping=0.5)
    damped_simulator = PendulumSimulator(damped_pendulum)
    damped_result = damped_simulator.simulate_euler(theta0=0.5, omega0=0.0, t_span=(0, 10), dt=0.01)

    amplitude_decrease = np.max(np.abs(damped_result.theta[:100])) - np.max(np.abs(damped_result.theta[-100:]))
    if amplitude_decrease > 0.1:
        print("Тест 3 пройден: колебания затухают при трении")
    else:
        print("Тест 3 не пройден: недостаточное затухание")

    print("Автотесты завершены!\n")


def create_rod_pendulum(length: float = 1.0, mass: float = 1.0) -> Pendulum:
    inertia = (1 / 3) * mass * length ** 2
    return Pendulum(mass=mass, length=length / 2, inertia=inertia)


def create_disk_pendulum(radius: float = 0.1, mass: float = 1.0, pivot_distance: float = 0.5) -> Pendulum:
    inertia_center = 0.5 * mass * radius ** 2
    inertia = inertia_center + mass * pivot_distance ** 2
    return Pendulum(mass=mass, length=pivot_distance, inertia=inertia)


def study_free_oscillations():
    print("=== Свободные колебания без трения ===")

    pendulum = create_rod_pendulum(length=1.0, mass=1.0)
    simulator = PendulumSimulator(pendulum)

    amplitudes = [0.1, 0.5, 1.0, 1.5]
    periods_vs_amplitude = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = list(mcolors.TABLEAU_COLORS.values())

    for i, amplitude in enumerate(amplitudes):
        result = simulator.simulate_euler(theta0=amplitude, omega0=0.0, t_span=(0, 10), dt=0.01)

        ax1.plot(result.t, result.theta, color=colors[i],
                 label=f'θ₀ = {amplitude:.1f} рад', linewidth=2)

        period_data, amp_data = result.analyze_period_vs_amplitude()
        if len(period_data) > 0:
            avg_period = np.mean(period_data[:3])
            periods_vs_amplitude.append((amplitude, avg_period))
            print(f"Амплитуда: {amplitude:.2f} рад, Период: {avg_period:.3f} с")

    ax1.set_xlabel('Время (с)')
    ax1.set_ylabel('Угол θ (рад)')
    ax1.set_title('Свободные колебания без трения')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if periods_vs_amplitude:
        amps, periods = zip(*periods_vs_amplitude)
        theoretical_period = pendulum.small_angle_period
        ax2.axhline(y=theoretical_period, color='red', linestyle='--',
                    label=f'Теоретический период: {theoretical_period:.3f} с')
        ax2.plot(amps, periods, 'bo-', linewidth=2, markersize=8)
        ax2.set_xlabel('Начальная амплитуда (рад)')
        ax2.set_ylabel('Период (с)')
        ax2.set_title('Зависимость периода от амплитуды')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    plt.tight_layout()
    plt.show()

    result = simulator.simulate_euler(theta0=1.0, omega0=0.0, t_span=(0, 10), dt=0.01)
    energy_drift = (np.max(result.energy) - np.min(result.energy)) / np.mean(result.energy)
    print(f"Дрейф энергии за 10 секунд: {energy_drift:.6f}")


def study_damped_oscillations():
    print("\n=== Свободные колебания с трением ===")

    pendulum = create_rod_pendulum(length=1.0, mass=1.0)

    damping_coeffs = [0.0, 0.1, 0.3, 0.5]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = list(mcolors.TABLEAU_COLORS.values())

    periods_vs_damping = []

    for i, damping in enumerate(damping_coeffs):
        pendulum.damping = damping
        simulator = PendulumSimulator(pendulum)
        result = simulator.simulate_euler(theta0=1.0, omega0=0.0, t_span=(0, 15), dt=0.01)

        ax1.plot(result.t, result.theta, color=colors[i],
                 label=f'b = {damping}', linewidth=2)

        period_data, _ = result.analyze_period_vs_amplitude()
        if len(period_data) > 0:
            avg_period = np.mean(period_data[:2])  # Берем первые 2 периода
            periods_vs_damping.append((damping, avg_period))
            print(f"Коэффициент трения: {damping:.1f}, Период: {avg_period:.3f} с")

    ax1.set_xlabel('Время (с)')
    ax1.set_ylabel('Угол θ (рад)')
    ax1.set_title('Затухающие колебания')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if periods_vs_damping:
        dampings, periods = zip(*periods_vs_damping)
        theoretical_period = pendulum.small_angle_period
        ax2.axhline(y=theoretical_period, color='red', linestyle='--',
                    label=f'Без трения: {theoretical_period:.3f} с')
        ax2.plot(dampings, periods, 'bo-', linewidth=2, markersize=8)
        ax2.set_xlabel('Коэффициент трения b')
        ax2.set_ylabel('Период (с)')
        ax2.set_title('Зависимость периода от трения')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_phase_portrait():
    """Построение фазового портрета"""
    
    print("\n=== Фазовый портрет ===")

    pendulum = create_rod_pendulum(length=1.0, mass=1.0)
    simulator = PendulumSimulator(pendulum)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    result_no_damp = simulator.simulate_euler(theta0=1.5, omega0=0.0, t_span=(0, 10), dt=0.01)
    ax1.plot(result_no_damp.theta, result_no_damp.omega, 'b-', linewidth=2)
    ax1.set_xlabel('Угол θ (рад)')
    ax1.set_ylabel('Угловая скорость ω (рад/с)')
    ax1.set_title('Фазовый портрет (без трения)')
    ax1.grid(True, alpha=0.3)

    pendulum.damping = 0.3
    simulator_damped = PendulumSimulator(pendulum)
    result_damped = simulator_damped.simulate_euler(theta0=1.5, omega0=0.0, t_span=(0, 20), dt=0.01)
    ax2.plot(result_damped.theta, result_damped.omega, 'r-', linewidth=2)
    ax2.set_xlabel('Угол θ (рад)')
    ax2.set_ylabel('Угловая скорость ω (рад/с)')
    ax2.set_title('Фазовый портрет (с трением)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def validate_float_input(prompt: str, default: float, min_val: float = None, max_val: float = None) -> float:
    while True:
        try:
            value = input(prompt).strip()
            if not value:
                return default
            value = float(value)
            if min_val is not None and value < min_val:
                print(f"Значение должно быть не меньше {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Значение должно быть не больше {max_val}")
                continue
            return value
        except ValueError:
            print("Введите корректное число")


def interactive_simulation():
    
    print("\n=== Интерактивное моделирование ===")

    print("Выберите тип маятника:")
    print("1. Стержень")
    print("2. Диск")
    choice = input("Ваш выбор [1]: ").strip() or "1"

    if choice == "1":
        length = validate_float_input("Длина стержня (м) [1.0]: ", 1.0, 0.1, 5.0)
        mass = validate_float_input("Масса (кг) [1.0]: ", 1.0, 0.1, 10.0)
        pendulum = create_rod_pendulum(length, mass)
    else:
        radius = validate_float_input("Радиус диска (м) [0.1]: ", 0.1, 0.01, 1.0)
        distance = validate_float_input("Расстояние до точки подвеса (м) [0.5]: ", 0.5, 0.1, 2.0)
        mass = validate_float_input("Масса (кг) [1.0]: ", 1.0, 0.1, 10.0)
        pendulum = create_disk_pendulum(radius, mass, distance)

    pendulum.damping = validate_float_input("Коэффициент трения [0.0]: ", 0.0, 0.0, 2.0)
    theta0 = validate_float_input("Начальный угол (рад) [0.5]: ", 0.5, 0.0, 3.0)
    t_max = validate_float_input("Время моделирования (с) [10.0]: ", 10.0, 1.0, 50.0)

    simulator = PendulumSimulator(pendulum)
    result = simulator.simulate_euler(theta0=theta0, omega0=0.0, t_span=(0, t_max), dt=0.01)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    ax1.plot(result.t, result.theta, 'b-', linewidth=2)
    ax1.set_xlabel('Время (с)')
    ax1.set_ylabel('Угол θ (рад)')
    ax1.set_title('Колебания маятника')
    ax1.grid(True, alpha=0.3)

    ax2.plot(result.t, result.energy, 'r-', linewidth=2)
    ax2.set_xlabel('Время (с)')
    ax2.set_ylabel('Энергия (Дж)')
    ax2.set_title('Полная механическая энергия')
    ax2.grid(True, alpha=0.3)

    ax3.plot(result.theta, result.omega, 'g-', linewidth=2)
    ax3.set_xlabel('Угол θ (рад)')
    ax3.set_ylabel('Угловая скорость ω (рад/с)')
    ax3.set_title('Фазовый портрет')
    ax3.grid(True, alpha=0.3)

    theoretical_period = pendulum.small_angle_period
    periods = result.find_periods()
    measured_period = np.mean(periods[:3]) if periods else 0

    info_text = f"""Параметры:
Масса: {pendulum.mass:.2f} кг
Длина: {pendulum.length:.2f} м
Момент инерции: {pendulum.inertia:.3f} кг·м²
Трение: {pendulum.damping:.2f}

Теоретический период: {theoretical_period:.3f} с
Измеренный период: {measured_period:.3f} с
Дрейф энергии: {((np.max(result.energy) - np.min(result.energy)) / np.mean(result.energy)):.6f}"""

    ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax4.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    print("ФИЗИЧЕСКИЙ МАЯТНИК - МОДЕЛИРОВАНИЕ")
    print(" ")

    run_autotests()

    while True:
        print("\nВыберите режим:")
        print("1. Исследование свободных колебаний (без трения)")
        print("2. Исследование затухающих колебаний")
        print("3. Фазовые портреты")
        print("4. Интерактивное моделирование")
        print("5. Выход")

        choice = input("Ваш выбор [1]: ").strip() or "1"

        if choice == "1":
            study_free_oscillations()
        elif choice == "2":
            study_damped_oscillations()
        elif choice == "3":
            plot_phase_portrait()
        elif choice == "4":
            interactive_simulation()
        elif choice == "5":
            print("Выход из программы.")
            break
        else:
            print("Неверный выбор. Попробуйте еще раз.")


if __name__ == "__main__":
    main()
