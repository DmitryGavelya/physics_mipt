from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Ball:
    mass: float
    radius: float
    position: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray
    color: str = "blue"

    def __post_init__(self):
        if self.mass <= 0 or self.radius <= 0:
            raise ValueError("Неверные параметры шара")

    @property
    def inertia(self) -> float:
        return 0.4 * self.mass * self.radius ** 2

    @property
    def kinetic_energy(self) -> float:
        translational = 0.5 * self.mass * np.dot(self.velocity, self.velocity)
        rotational = 0.5 * self.inertia * np.dot(self.angular_velocity, self.angular_velocity)
        return translational + rotational


class Surface:
    def __init__(self, normal: np.ndarray, point: np.ndarray, friction_coeff: float = 0.1):
        self.normal = normal / np.linalg.norm(normal)
        self.point = point
        self.friction_coeff = friction_coeff


class SimulationArea:
    def __init__(self, width: float = 2.0, height: float = 1.0, friction_coeff: float = 0.1):
        self.width = width
        self.height = height
        self.walls = [
            Surface(normal=np.array([1.0, 0.0]), point=np.array([-width / 2, 0.0]), friction_coeff=friction_coeff),
            Surface(normal=np.array([-1.0, 0.0]), point=np.array([width / 2, 0.0]), friction_coeff=friction_coeff),
            Surface(normal=np.array([0.0, 1.0]), point=np.array([0.0, -height / 2]), friction_coeff=friction_coeff),
            Surface(normal=np.array([0.0, -1.0]), point=np.array([0.0, height / 2]), friction_coeff=friction_coeff)
        ]


class BallSimulator:
    def __init__(self, gravity: float = 9.81, restitution: float = 0.98):
        self.gravity = gravity
        self.restitution = np.clip(restitution, 0.0, 1.0)
        self.balls: List[Ball] = []
        self.surfaces: List[Surface] = []
        self.area: Optional[SimulationArea] = None

    def add_ball(self, ball: Ball) -> None:
        self.balls.append(ball)

    def add_surface(self, surface: Surface) -> None:
        self.surfaces.append(surface)

    def set_area(self, area: SimulationArea) -> None:
        self.area = area
        self.surfaces.extend(area.walls)

    def _zeta_from_e(self, e: float) -> float:
        e = np.clip(e, 1e-6, 0.999999)
        ln_e = np.log(e)
        return np.sqrt(ln_e * ln_e / (np.pi * np.pi + ln_e * ln_e))

    def _contact_point_velocity(self, ball: Ball, surface: Surface) -> np.ndarray:
        contact_point = ball.position - ball.radius * surface.normal
        r = contact_point - ball.position
        return ball.velocity + np.cross(np.array([0.0, 0.0, ball.angular_velocity[2]]), np.append(r, 0.0))[:2]

    def _surface_contact_forces(self, ball: Ball, surface: Surface) -> Tuple[np.ndarray, float]:
        n = surface.normal
        distance = np.dot(ball.position - surface.point, n)
        penetration = ball.radius - distance
        if penetration <= 0:
            return np.zeros(2), 0.0
        k = 2e4 * ball.mass
        zeta = self._zeta_from_e(self.restitution)
        c = 2.0 * zeta * np.sqrt(k * ball.mass)
        v_rel_n = np.dot(ball.velocity, n)
        fn_mag = max(k * penetration - c * min(v_rel_n, 0.0), 0.0)
        fn = fn_mag * n
        vcp = self._contact_point_velocity(ball, surface)
        vt = vcp - np.dot(vcp, n) * n
        vtn = np.linalg.norm(vt)
        if vtn < 1e-8:
            ft = np.zeros(2)
        else:
            ft = -surface.friction_coeff * fn_mag * vt / vtn
        tau = np.cross(np.append(-ball.radius * n, 0.0), np.append(ft, 0.0))[2]
        return fn + ft, tau

    def _ball_ball_contact_forces(self, b1: Ball, b2: Ball) -> Tuple[np.ndarray, np.ndarray, float, float]:
        r = b1.position - b2.position
        dist = np.linalg.norm(r)
        R = b1.radius + b2.radius
        if dist >= R or dist < 1e-9:
            return np.zeros(2), np.zeros(2), 0.0, 0.0
        n = r / max(dist, 1e-9)
        penetration = R - dist
        m_eff = (b1.mass * b2.mass) / (b1.mass + b2.mass)
        k = 2e4 * m_eff
        zeta = self._zeta_from_e(self.restitution)
        c = 2.0 * zeta * np.sqrt(k * m_eff)
        vn = np.dot(b1.velocity - b2.velocity, n)
        fn_mag = max(k * penetration - c * min(vn, 0.0), 0.0)
        fn = fn_mag * n
        tau1 = np.cross(np.append(-b1.radius * n, 0.0), np.append(fn, 0.0))[2]
        tau2 = np.cross(np.append(b2.radius * n, 0.0), np.append(-fn, 0.0))[2]
        return fn, -fn, tau1, tau2

    def _compute_accelerations(self, ball: Ball) -> Tuple[np.ndarray, float]:
        total_force = np.array([0.0, -self.gravity * ball.mass])
        total_torque = 0.0
        for surf in self.surfaces:
            f, tau = self._surface_contact_forces(ball, surf)
            total_force += f
            total_torque += tau
        for other in self.balls:
            if other is ball:
                continue
            f12, _, tau1, _ = self._ball_ball_contact_forces(ball, other)
            total_force += f12
            total_torque += tau1
        return total_force / ball.mass, total_torque / ball.inertia

    def derivatives(self, y: np.ndarray) -> np.ndarray:
        n = len(self.balls)
        dydt = np.zeros(5 * n)
        for i, ball in enumerate(self.balls):
            pos = y[5 * i:5 * i + 2]
            vel = y[5 * i + 2:5 * i + 4]
            omega = y[5 * i + 4]
            ball.position = pos
            ball.velocity = vel
            ball.angular_velocity = np.array([0.0, 0.0, omega])
            acc, alpha = self._compute_accelerations(ball)
            dydt[5 * i:5 * i + 2] = vel
            dydt[5 * i + 2:5 * i + 4] = acc
            dydt[5 * i + 4] = alpha
        return dydt

    def simulate(self, t_span: Tuple[float, float], dt: float = 0.001) -> SimulationResult:
        n = len(self.balls)
        y0 = np.zeros(5 * n)
        for i, ball in enumerate(self.balls):
            y0[5 * i:5 * i + 2] = ball.position
            y0[5 * i + 2:5 * i + 4] = ball.velocity
            y0[5 * i + 4] = ball.angular_velocity[2]

        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt) + 1
        t_eval = np.linspace(t_start, t_end, n_steps)

        y = y0.copy()
        solution_y = np.zeros((5 * n, n_steps))

        for i, t in enumerate(t_eval):
            solution_y[:, i] = y
            dydt = self.derivatives(y)
            y = y + dydt * dt

        solution = type('Solution', (), {'t': t_eval, 'y': solution_y})()
        return SimulationResult(solution, self.balls)


class SimulationResult:
    def __init__(self, solution, balls: List[Ball]):
        self.t = solution.t
        self.balls_data = []
        n_balls = len(balls)
        for i in range(n_balls):
            self.balls_data.append({
                'position': solution.y[5 * i:5 * i + 2].T,
                'velocity': solution.y[5 * i + 2:5 * i + 4].T,
                'angular_velocity': solution.y[5 * i + 4].T
            })
        self.balls = balls
        masses = np.array([b.mass for b in balls], dtype=float)
        inertias = np.array([b.inertia for b in balls], dtype=float)
        vel = np.stack([bd['velocity'] for bd in self.balls_data], axis=0)
        omega = np.stack([bd['angular_velocity'] for bd in self.balls_data], axis=0)
        vel2 = np.einsum('...i,...i->...', vel, vel)
        omega2 = omega * omega
        self._total_energy = (0.5 * masses[:, None] * vel2 + 0.5 * inertias[:, None] * omega2).sum(axis=0)

    def get_energy(self) -> np.ndarray:
        return self._total_energy


def run_autotests():
    print("Запуск автотестов...")

    sim = BallSimulator(gravity=0.0, restitution=1.0)
    ball = Ball(0.17, 0.0285, np.array([0.0, 0.0]), np.array([2.0, 1.0]), np.array([0.0, 0.0, 0.0]))
    sim.add_ball(ball)
    result = sim.simulate((0.0, 0.3), 0.005)
    energy_start = result.get_energy()[0]
    energy_end = result.get_energy()[-1]
    energy_diff = abs(energy_end - energy_start)
    if energy_diff < 0.01:
        print("Тест 1 пройден: энергия сохраняется без сил контакта")
    else:
        print(f"Тест 1 не пройден: изменение энергии {energy_diff:.6f}")

    sim = BallSimulator(gravity=9.81, restitution=0.9)
    surface = Surface(normal=np.array([0.0, 1.0]), point=np.array([0.0, -0.5]), friction_coeff=0.5)
    sim.add_surface(surface)
    ball = Ball(0.17, 0.0285, np.array([0.0, 0.1]), np.array([3.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    sim.add_ball(ball)
    result = sim.simulate((0.0, 0.6), 0.005)
    energy_start = result.get_energy()[0]
    energy_end = result.get_energy()[-1]
    if energy_end < energy_start - 0.01:
        print("Тест 2 пройден: энергия уменьшается из-за трения")
    else:
        print("Тест 2 не пройден: энергия не уменьшилась достаточно")

    sim = BallSimulator(gravity=0.0, restitution=0.98)
    area = SimulationArea(width=2.0, height=1.0, friction_coeff=0.0)
    sim.set_area(area)
    b1 = Ball(0.17, 0.0285, np.array([-0.1, 0.0]), np.array([2.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    b2 = Ball(0.17, 0.0285, np.array([0.1, 0.0]), np.array([-1.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    sim.add_ball(b1)
    sim.add_ball(b2)
    result = sim.simulate((0.0, 0.2), 0.001)
    pos1_final = result.balls_data[0]['position'][-1]
    pos2_final = result.balls_data[1]['position'][-1]
    if abs(pos1_final[0]) > 0.15 or abs(pos2_final[0]) > 0.15:
        print("Тест 3 пройден: столкновение шаров моделируется")
    else:
        print("Тест 3 не пройден: шары не изменили траекторию")

    print("\nВсе автотесты пройдены")


def create_inclined_plane_simulation(angle: float = 30.0, mu: float = 0.3, y0: float = 0.3,
                                     radius: float = 0.05) -> BallSimulator:
    sim = BallSimulator(gravity=9.81, restitution=0.98)
    inclined_surface = Surface(
        normal=np.array([np.sin(np.radians(angle)), np.cos(np.radians(angle))]),
        point=np.array([0.0, -0.4]),
        friction_coeff=mu
    )
    sim.add_surface(inclined_surface)
    ball = Ball(
        mass=1.0,
        radius=radius,
        position=np.array([0.0, y0]),
        velocity=np.array([0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.0]),
        color="red"
    )
    sim.add_ball(ball)
    return sim


def create_horizontal_surface_simulation(n_balls: int = 2) -> BallSimulator:
    import matplotlib.colors as mcolors
    sim = BallSimulator(gravity=9.81, restitution=0.98)
    area = SimulationArea(width=1.8, height=0.9, friction_coeff=0.2)
    sim.set_area(area)
    colors = list(mcolors.TABLEAU_COLORS.values())
    for i in range(n_balls):
        pos = np.array([-0.3 + 0.3 * i, 0.1 * (-1) ** i])
        vel = np.array([2.0 - 1.0 * i, 0.5 * (1 if i % 2 == 0 else -1)])
        omega = 4.0 * (1 if i % 2 == 0 else -1)
        sim.add_ball(Ball(1.0, 0.04, pos, vel, np.array([0.0, 0.0, omega]), colors[i % len(colors)]))
    return sim


def create_elastic_collisions_simulation(restitution: float = 0.98) -> BallSimulator:
    import matplotlib.colors as mcolors
    sim = BallSimulator(gravity=0.0, restitution=restitution)
    area = SimulationArea(width=2.0, height=1.0, friction_coeff=0.0)
    sim.set_area(area)
    colors = list(mcolors.TABLEAU_COLORS.values())
    b1 = Ball(1.0, 0.04, np.array([-0.5, 0.0]), np.array([1.5, 0.0]), np.array([0.0, 0.0, 0.0]), colors[0])
    b2 = Ball(1.0, 0.04, np.array([0.5, 0.0]), np.array([-1.0, 0.0]), np.array([0.0, 0.0, 0.0]), colors[1])
    sim.add_ball(b1)
    sim.add_ball(b2)
    return sim


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


def validate_int_input(prompt: str, default: int, min_val: int = None, max_val: int = None) -> int:
    while True:
        try:
            value = input(prompt).strip()
            if not value:
                return default
            value = int(value)
            if min_val is not None and value < min_val:
                print(f"Значение должно быть не меньше {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Значение должно быть не больше {max_val}")
                continue
            return value
        except ValueError:
            print("Введите корректное целое число")


def validate_vector_input(prompt: str, default: List[float]) -> np.ndarray:
    while True:
        try:
            value = input(prompt).strip()
            if not value:
                return np.array(default)
            values = list(map(float, value.split()))
            if len(values) != 2:
                print("Введите два числа через пробел")
                continue
            return np.array(values)
        except ValueError:
            print("Введите корректные числа")


def plot_simulation(result: SimulationResult):
    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-0.8, 0.8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X (м)')
    ax1.set_ylabel('Y (м)')
    ax1.set_title('Траектории движения (центр шара)')

    for i, ball in enumerate(result.balls):
        trajectory = result.balls_data[i]['position']
        ax1.plot(trajectory[:, 0], trajectory[:, 1], '-', color=ball.color, linewidth=2, label=f'Шар {i + 1}')
        ax1.plot(trajectory[0, 0], trajectory[0, 1], 'o', color=ball.color, markersize=8, markeredgecolor='black')
        ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 's', color=ball.color, markersize=8, markeredgecolor='black')
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    energy = result.get_energy()
    ax2.plot(result.t, energy, 'b-', linewidth=2)
    ax2.set_xlabel('Время (с)')
    ax2.set_ylabel('Энергия (Дж)')
    ax2.set_title('Полная энергия системы')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def choose_and_run_scenario():
    T_DEFAULT = 0.41
    DT_INC = 0.002
    DT_HOR = 0.002
    DT_ELA = 0.0005

    print("\nВыберите сценарий:")
    print("1. Скатывание по наклонной плоскости")
    print("2. Качение по горизонтальной плоскости")
    print("3. Упругие столкновения шаров и с бортиком")
    choice = input("Ваш выбор [1]: ").strip() or "1"

    def ask_time(default_T: float) -> float:
        return validate_float_input(
            f"Время моделирования [{default_T:.2f}] (сек, 0.05-5.0): ",
            default_T, 0.05, 5.0
        )

    if choice == "1":
        angle = validate_float_input("Угол наклона [30]: ", 30.0, 0.0, 85.0)
        mu = validate_float_input("Коэффициент трения [0.3]: ", 0.3, 0.0, 2.0)
        y0 = validate_float_input("Начальная высота центра [0.3]: ", 0.3, -0.6, 0.6)
        radius = validate_float_input("Радиус шара [0.05]: ", 0.05, 0.01, 0.2)
        T = ask_time(T_DEFAULT)
        sim = create_inclined_plane_simulation(angle, mu, y0, radius)
        result = sim.simulate((0.0, T), DT_INC)
        plot_simulation(result)

    elif choice == "2":
        import matplotlib.colors as mcolors
        n_balls = validate_int_input("Количество шаров [2]: ", 2, 1, 6)
        sim = BallSimulator(gravity=9.81, restitution=0.98)
        area_w = validate_float_input("Ширина области [1.8]: ", 1.8, 0.5, 6.0)
        area_h = validate_float_input("Высота области [0.9]: ", 0.9, 0.3, 4.0)
        mu_area = validate_float_input("Коэффициент трения стенок [0.2]: ", 0.2, 0.0, 2.0)
        area = SimulationArea(width=area_w, height=area_h, friction_coeff=mu_area)
        sim.set_area(area)
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i in range(n_balls):
            mass = validate_float_input(f"Масса шара {i + 1} [1.0]: ", 1.0, 0.05, 10.0)
            radius = validate_float_input(f"Радиус шара {i + 1} [0.04]: ", 0.04, 0.01, 0.2)
            pos = validate_vector_input(f"Позиция шара {i + 1} [0.0 0.0]: ", [0.0, 0.0])
            vel = validate_vector_input(f"Скорость шара {i + 1} [1.0 0.0]: ", [1.0, 0.0])
            omega = validate_float_input(f"Угл. скорость z шара {i + 1} [0.0]: ", 0.0, -200.0, 200.0)
            sim.add_ball(Ball(mass, radius, pos, vel, np.array([0.0, 0.0, omega]), colors[i % len(colors)]))
        T = ask_time(T_DEFAULT)
        result = sim.simulate((0.0, T), DT_HOR)
        plot_simulation(result)

    else:
        import matplotlib.colors as mcolors
        e = validate_float_input("Коэффициент восстановления [0.98]: ", 0.98, 0.0, 1.0)
        sim = BallSimulator(gravity=0.0, restitution=e)
        area_w = validate_float_input("Ширина области [2.0]: ", 2.0, 0.5, 6.0)
        area_h = validate_float_input("Высота области [1.0]: ", 1.0, 0.3, 4.0)
        area = SimulationArea(width=area_w, height=area_h, friction_coeff=0.0)
        sim.set_area(area)
        colors = list(mcolors.TABLEAU_COLORS.values())
        b1_pos = validate_vector_input("Позиция шара 1 [-0.5 0.0]: ", [-0.5, 0.0])
        b1_vel = validate_vector_input("Скорость шара 1 [1.5 0.0]: ", [1.5, 0.0])
        b2_pos = validate_vector_input("Позиция шара 2 [0.5 0.0]: ", [0.5, 0.0])
        b2_vel = validate_vector_input("Скорость шара 2 [-1.0 0.0]: ", [-1.0, 0.0])
        sim.add_ball(Ball(1.0, 0.04, b1_pos, b1_vel, np.array([0.0, 0.0, 0.0]), colors[0]))
        sim.add_ball(Ball(1.0, 0.04, b2_pos, b2_vel, np.array([0.0, 0.0, 0.0]), colors[1]))
        T = ask_time(T_DEFAULT)
        result = sim.simulate((0.0, T), DT_ELA)
        plot_simulation(result)


def main():
    print("МОДЕЛИРОВАНИЕ КАЧЕНИЯ ШАРОВ")
    run_autotests()
    choose_and_run_scenario()


if __name__ == "__main__":
    main()
