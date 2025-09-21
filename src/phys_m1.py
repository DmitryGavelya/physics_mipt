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


def stone_equations(t, state, model_type, k, m):
    x, y, vx, vy = state

    if model_type == 'viscous':
        # Вязкое трение: F = v
        Fx = -k * vx
        Fy = -k * vy
    elif model_type == 'quadratic':
        # Лобовое сопротивление: F = v²
        v = np.sqrt(vx ** 2 + vy ** 2)
        Fx = -k * v * vx
        Fy = -k * v * vy
    else:
        Fx = 0
        Fy = 0

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

    # Ввод параметров пользователем
    m = float(input("Введите массу: "))
    v0 = float(input("Введите начальную скорость: "))
    angle = float(input("Введите угол броска: "))
    k = float(input("Введите коэффициент сопротивления: "))

    try:
        validate_inputs(m, v0, angle, k)
    except ValueError as e:
        print(f"Ошибка ввода: {e}")
        raise print("Пожалуйста, введите корректные физические параметры")

    alpha_rad = np.radians(angle)
    v0x = v0 * np.cos(alpha_rad)
    v0y = v0 * np.sin(alpha_rad)




    t_span = (0, 10000)

    # Решение для вязкого трения
    sol_viscous = solve_ivp(stone_equations, t_span, [0, 0, v0x, v0y],
                            args=('viscous', k, m), events=hit_ground, dense_output=True)

    # Решение для квадратичного сопротивления
    sol_quadratic = solve_ivp(stone_equations, t_span, [0, 0, v0x, v0y],
                              args=('quadratic', k, m), events=hit_ground, dense_output=True)

    # Время полета для каждой модели
    flight_time_viscous = sol_viscous.t_events[0][0] if sol_viscous.t_events[0].size > 0 else t_span[1]
    flight_time_quadratic = sol_quadratic.t_events[0][0] if sol_quadratic.t_events[0].size > 0 else t_span[1]

    print(f"Время полета (вязкое трение): {flight_time_viscous:.2f} с")
    print(f"Время полета (квадратичное сопротивление): {flight_time_quadratic:.2f} с")

    t_eval_viscous = np.linspace(0, flight_time_viscous, 100)
    t_eval_quadratic = np.linspace(0, flight_time_quadratic, 100)

    plt.figure(figsize=(10, 6))
    plt.plot(sol_viscous.sol(t_eval_viscous)[0], sol_viscous.sol(t_eval_viscous)[1],
             label='Вязкое трение')
    plt.plot(sol_quadratic.sol(t_eval_quadratic)[0], sol_quadratic.sol(t_eval_quadratic)[1],
             label='Квадратичное сопротивление')
    plt.xlabel('x, м')
    plt.ylabel('y, м')
    plt.title('Траектории полета камня')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

