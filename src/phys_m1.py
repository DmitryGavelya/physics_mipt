import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

g = 9.81


def validate_inputs(m, v0, angle, k):
    if m <= 0:
        raise ValueError("Масса должна быть положительной (m > 0)")


    if m > 1000000000:
        raise ValueError("Масса должна быть меньше миллиарда")

    if v0 < 0:
        raise ValueError("Начальная скорость не может быть отрицательной (v0 ≥ 0)")

    if v0 > 1000:
        raise ValueError(f"Начальная скорость не может превышать 1000 м/с. Введенное значение: {v0} м/с")

    if angle < 0 or angle > 90:
        raise ValueError("Угол броска должен быть в диапазоне от 0 до 90 градусов")

    if k < 0:
        raise ValueError("Коэффициент сопротивления не может быть отрицательным (k ≥ 0)")

    if k > 10_000:
        raise ValueError("Коэффициент сопротивления не может быть больше 10_000")


def stone_equations(t, state, model_type, k, m):
    x, y, vx, vy = state

    if model_type == 'viscous':
        Fx = -k * vx
        Fy = -k * vy
    elif model_type == 'quadratic':
        v = np.sqrt(vx ** 2 + vy ** 2)
        if v > 1e-10:
            Fx = -k * v * vx
            Fy = -k * v * vy
        else:
            Fx, Fy = 0, 0
    else:
        Fx, Fy = 0, 0

    dxdt = vx
    dydt = vy
    dvxdt = Fx / m
    dvydt = -g + Fy / m

    return [dxdt, dydt, dvxdt, dvydt]


def hit_ground(t, state, *args):
    return state[1]


hit_ground.terminal = True
hit_ground.direction = -1


def main():
    m = float(input("Введите массу камня (кг): "))
    v0 = float(input("Введите начальную скорость (м/с): "))
    angle = float(input("Введите угол броска (градусы): "))
    k = float(input("Введите коэффициент сопротивления: "))

    try:
        validate_inputs(m, v0, angle, k)
    except ValueError as e:
        print(f"Ошибка ввода: {e}")
        print("Пожалуйста, введите корректные физические параметры")
        return

    alpha_rad = np.radians(angle)
    v0x = v0 * np.cos(alpha_rad)
    v0y = v0 * np.sin(alpha_rad)

    t_span = (0, 1000)

    method = 'RK45'
    rtol = 1e-8
    atol = 1e-10
    max_step = 0.1


    try:
        sol_viscous = solve_ivp(stone_equations, t_span, [0, 0, v0x, v0y],
                                method=method, rtol=rtol, atol=atol, max_step=max_step,
                                args=('viscous', k, m), events=hit_ground, dense_output=True)

        sol_quadratic = solve_ivp(stone_equations, t_span, [0, 0, v0x, v0y],
                                  method=method, rtol=rtol, atol=atol, max_step=max_step,
                                  args=('quadratic', k, m), events=hit_ground, dense_output=True)

        flight_time_viscous = sol_viscous.t_events[0][0] if sol_viscous.t_events[0].size > 0 else t_span[1]
        flight_time_quadratic = sol_quadratic.t_events[0][0] if sol_quadratic.t_events[0].size > 0 else t_span[1]

        print(f"Время полета (вязкое трение): {flight_time_viscous:.2f} с")
        print(f"Время полета (квадратичное сопротивление): {flight_time_quadratic:.2f} с")

        points_viscous = min(500, max(50, int(flight_time_viscous * 10)))
        points_quadratic = min(500, max(50, int(flight_time_quadratic * 100)))

        t_eval_viscous = np.linspace(0, flight_time_viscous, points_viscous)
        t_eval_quadratic = np.linspace(0, flight_time_quadratic, points_quadratic)

        x_viscous = sol_viscous.sol(t_eval_viscous)[0]
        y_viscous = sol_viscous.sol(t_eval_viscous)[1]
        x_quadratic = sol_quadratic.sol(t_eval_quadratic)[0]
        y_quadratic = sol_quadratic.sol(t_eval_quadratic)[1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.plot(x_viscous, y_viscous, 'b-', label='Вязкое трение', linewidth=2)
        ax1.plot(x_quadratic, y_quadratic, 'r--', label='Квадратичное сопротивление', linewidth=2)
        ax1.set_xlabel('Расстояние, м')
        ax1.set_ylabel('Высота, м')
        ax1.set_title('Сравнение траекторий')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)
        ax1.set_ylim(bottom=0)

        ax2.plot(x_quadratic, y_quadratic, 'r--', label='Квадратичное сопротивление', linewidth=2)
        ax2.set_xlabel('Расстояние, м')
        ax2.set_ylabel('Высота, м')
        ax2.set_title('Траектория с квадратичным сопротивлением')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(left=0)
        ax2.set_ylim(bottom=0)

        if len(x_quadratic) > 1 and len(y_quadratic) > 1:
            x_max_q = np.max(x_quadratic)
            y_max_q = np.max(y_quadratic)

            x_max_v = np.max(x_viscous) if len(x_viscous) > 0 else 0
            y_max_v = np.max(y_viscous) if len(y_viscous) > 0 else 0

            x_max = max(x_max_v, x_max_q)
            y_max = max(y_max_v, y_max_q)

            ax1.set_xlim(0, x_max * 1.05 if x_max > 0 else 100)
            ax1.set_ylim(0, y_max * 1.05 if y_max > 0 else 100)

            if x_max_q > 0 and y_max_q > 0:
                ax2.set_xlim(0, x_max_q * 1.1)
                ax2.set_ylim(0, y_max_q * 1.1)
            else:
                ax2.set_xlim(0, 10)
                ax2.set_ylim(0, 10)
        else:
            ax1.set_xlim(0, 100)
            ax1.set_ylim(0, 100)
            ax2.set_xlim(0, 10)
            ax2.set_ylim(0, 10)

        plt.tight_layout()
        plt.show()


    except Exception as e:
        print(f"Ошибка при расчете траектории: {e}")
        print("Попробуйте изменить параметры (уменьшить коэффициент сопротивления или увеличить массу)")


if __name__ == "__main__":
    main()