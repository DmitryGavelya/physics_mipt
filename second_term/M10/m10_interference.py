from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


EPS = 1e-15


@dataclass
class YoungParams:
    """Параметры установки Юнга."""

    wavelength: float = 550e-9       # lambda, м
    slit_distance: float = 0.5e-3    # d, м
    source_to_slits: float = 1.0     # L1, м
    slits_to_screen: float = 2.0     # L2, м
    screen_half_width: float = 10e-3 # половина ширины экрана, м
    screen_points: int = 2001


def normalized_sinc(x: np.ndarray | float) -> np.ndarray | float:
    """
    Нормированная sinc-функция:

        sinc(x) = sin(pi x) / (pi x)

    Именно такая форма используется в формулах из условия:

        V = |sin(pi b d / lambda L1) / (pi b d / lambda L1)|
    """

    return np.sinc(x)


def require_matplotlib() -> None:
    if plt is None:
        print("Ошибка: для построения графиков нужен matplotlib.")
        print("Установите его командой: pip install matplotlib")
        sys.exit(1)


def ask_float(
    prompt: str,
    default: Optional[float] = None,
    *,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_zero: bool = False,
) -> float:
    """Безопасный ввод вещественного числа с проверками."""

    while True:
        suffix = f" [{default:g}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip().replace(",", ".")

        if raw == "" and default is not None:
            value = default
        else:
            try:
                value = float(raw)
            except ValueError:
                print("  Некорректный ввод. Введите число, например 550e-9.")
                continue

        if not math.isfinite(value):
            print("  Значение должно быть конечным числом.")
            continue

        if min_value is not None:
            if allow_zero and value < min_value:
                print(f"  Значение должно быть не меньше {min_value:g}.")
                continue
            if not allow_zero and value <= min_value:
                print(f"  Значение должно быть больше {min_value:g}.")
                continue

        if max_value is not None and value > max_value:
            print(f"  Значение должно быть не больше {max_value:g}.")
            continue

        return value


def ask_int(
    prompt: str,
    default: Optional[int] = None,
    *,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    """Безопасный ввод целого числа с проверками."""

    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()

        if raw == "" and default is not None:
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                print("  Некорректный ввод. Введите целое число.")
                continue

        if min_value is not None and value < min_value:
            print(f"  Значение должно быть не меньше {min_value}.")
            continue

        if max_value is not None and value > max_value:
            print(f"  Значение должно быть не больше {max_value}.")
            continue

        return value


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Ввод ответа да/нет."""

    default_text = "Д/н" if default else "д/Н"
    while True:
        raw = input(f"{prompt} [{default_text}]: ").strip().lower()
        if raw == "":
            return default
        if raw in {"д", "да", "y", "yes"}:
            return True
        if raw in {"н", "нет", "n", "no"}:
            return False
        print("  Введите 'да' или 'нет'.")


def warn_if_non_paraxial(params: YoungParams) -> None:
    """Предупреждения о выходе из параксиального приближения."""

    max_angle_screen = params.screen_half_width / params.slits_to_screen
    slit_angle = params.slit_distance / params.slits_to_screen

    if max_angle_screen > 0.1:
        print(
            "Предупреждение: x_max / L2 > 0.1. "
            "Параксиальное приближение для краёв экрана может быть грубым."
        )

    if slit_angle > 0.1:
        print(
            "Предупреждение: d / L2 > 0.1. "
            "Параксиальное приближение для щелей может быть грубым."
        )


def input_common_params() -> YoungParams:
    """Ввод основных параметров установки."""

    print("\nВведите параметры установки в СИ. Можно нажимать Enter для значений по умолчанию.")
    defaults = YoungParams()

    wavelength = ask_float(
        "Длина волны lambda, м",
        defaults.wavelength,
        min_value=0.0,
    )
    slit_distance = ask_float(
        "Расстояние между щелями d, м",
        defaults.slit_distance,
        min_value=0.0,
    )
    source_to_slits = ask_float(
        "Расстояние источник — щели L1, м",
        defaults.source_to_slits,
        min_value=0.0,
    )
    slits_to_screen = ask_float(
        "Расстояние щели — экран L2, м",
        defaults.slits_to_screen,
        min_value=0.0,
    )
    screen_half_width = ask_float(
        "Половина ширины экрана x_max, м",
        defaults.screen_half_width,
        min_value=0.0,
    )
    screen_points = ask_int(
        "Число точек экрана",
        defaults.screen_points,
        min_value=101,
        max_value=200_001,
    )

    if screen_points % 2 == 0:
        screen_points += 1
        print(f"Число точек сделано нечётным для симметрии: {screen_points}")

    params = YoungParams(
        wavelength=wavelength,
        slit_distance=slit_distance,
        source_to_slits=source_to_slits,
        slits_to_screen=slits_to_screen,
        screen_half_width=screen_half_width,
        screen_points=screen_points,
    )

    warn_if_non_paraxial(params)
    return params


def screen_grid(params: YoungParams) -> np.ndarray:
    """Сетка координат экрана."""

    return np.linspace(
        -params.screen_half_width,
        params.screen_half_width,
        params.screen_points,
    )


def fringe_period(params: YoungParams) -> float:
    """Период интерференционных полос Lambda = lambda L2 / d."""

    return params.wavelength * params.slits_to_screen / params.slit_distance


def path_difference_exact(x: np.ndarray, params: YoungParams) -> np.ndarray:
    """
    Точная геометрическая разность хода от двух щелей до точки x экрана.

    Щели расположены в координатах -d/2 и +d/2.
    """

    d = params.slit_distance
    L2 = params.slits_to_screen
    r1 = np.sqrt(L2**2 + (x + d / 2) ** 2)
    r2 = np.sqrt(L2**2 + (x - d / 2) ** 2)
    return r2 - r1


def path_difference_paraxial(x: np.ndarray, params: YoungParams) -> np.ndarray:
    """
    Параксиальная разность хода.

    С точностью до знака Delta ≈ -x d / L2.
    Знак не влияет на интенсивность, потому что cos чётная.
    """

    return -x * params.slit_distance / params.slits_to_screen


def analytic_point_intensity(
    x: np.ndarray,
    params: YoungParams,
    *,
    exact_geometry: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    """
    Аналитическая интенсивность для точечного монохроматического источника.

    При равных амплитудах:
        I = 2 + 2 cos(2 pi Delta / lambda)
          = 4 cos^2(pi Delta / lambda)

    Если normalize=True, максимум нормируется к 1.
    """

    if exact_geometry:
        delta = path_difference_exact(x, params)
    else:
        delta = path_difference_paraxial(x, params)

    phase = 2.0 * np.pi * delta / params.wavelength
    intensity = 2.0 + 2.0 * np.cos(phase)

    if normalize:
        max_i = np.max(intensity)
        if max_i > EPS:
            intensity = intensity / max_i

    return intensity


def numerical_point_intensity(
    x: np.ndarray,
    params: YoungParams,
    *,
    source_x: float = 0.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Численное суммирование комплексных амплитуд от двух щелей.

    Источник S имеет координаты (source_x, -L1).
    Щели лежат в плоскости z = 0:
        S1 = (-d/2, 0), S2 = (+d/2, 0)
    Экран лежит в плоскости z = L2:
        P = (x, L2)

    Амплитуда:
        U(x) = sum_n exp(i k (r_source_to_slit + r_slit_to_screen))
               / (r_source_to_slit * r_slit_to_screen)
    """

    k = 2.0 * np.pi / params.wavelength
    d = params.slit_distance
    L1 = params.source_to_slits
    L2 = params.slits_to_screen

    slit_positions = np.array([-d / 2.0, d / 2.0])
    amplitude = np.zeros_like(x, dtype=np.complex128)

    for slit_x in slit_positions:
        r_source_slit = math.sqrt(L1**2 + (slit_x - source_x) ** 2)
        r_slit_screen = np.sqrt(L2**2 + (x - slit_x) ** 2)
        phase = k * (r_source_slit + r_slit_screen)
        amplitude += np.exp(1j * phase) / (r_source_slit * r_slit_screen)

    intensity = np.abs(amplitude) ** 2

    if normalize:
        max_i = np.max(intensity)
        if max_i > EPS:
            intensity = intensity / max_i

    return intensity


def michelson_visibility(intensity: np.ndarray) -> float:
    """
    Видность Майкельсона:
        V = (Imax - Imin) / (Imax + Imin)

    Для надёжности берём минимум и максимум по массиву.
    """

    i_max = float(np.max(intensity))
    i_min = float(np.min(intensity))
    denominator = i_max + i_min
    if denominator <= EPS:
        return 0.0
    return (i_max - i_min) / denominator


def spatial_visibility_analytic(
    b: np.ndarray | float,
    params: YoungParams,
) -> np.ndarray | float:
    """
    Видность от размера протяжённого источника:

        V = |sinc(b d / (lambda L1))|

    Здесь sinc(x) = sin(pi x) / (pi x).
    """

    argument = np.asarray(b) * params.slit_distance / (
        params.wavelength * params.source_to_slits
    )
    return np.abs(normalized_sinc(argument))


def spectral_visibility_analytic(
    delta_lambda: np.ndarray | float,
    optical_path_difference: np.ndarray | float,
    lambda0: float,
) -> np.ndarray | float:
    """
    Видность для равномерного спектра ширины delta_lambda около lambda0:

        V = |sinc((delta_lambda / lambda0^2) * Delta)|
    """

    argument = np.asarray(delta_lambda) * np.asarray(optical_path_difference) / (lambda0**2)
    return np.abs(normalized_sinc(argument))


def spectral_visibility_by_order(
    m: np.ndarray | float,
    delta_lambda: float,
    lambda0: float,
) -> np.ndarray | float:
    """
    Видность как функция порядка полосы m:

        V = |sinc(m delta_lambda / lambda0)|
    """

    argument = np.asarray(m) * delta_lambda / lambda0
    return np.abs(normalized_sinc(argument))


def extended_source_intensity_numerical(
    x: np.ndarray,
    params: YoungParams,
    source_size: float,
    source_points: int,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """
    Некогерентное суммирование интенсивностей от точек протяжённого источника.

    Разные точки протяжённого источника считаются взаимно некогерентными,
    поэтому суммируются интенсивности, а не комплексные амплитуды.
    """

    if source_size <= 0:
        return numerical_point_intensity(x, params, source_x=0.0, normalize=normalize)

    xis = np.linspace(-source_size / 2.0, source_size / 2.0, source_points)
    total = np.zeros_like(x, dtype=np.float64)

    for xi in xis:
        total += numerical_point_intensity(x, params, source_x=xi, normalize=False)

    total /= source_points

    if normalize:
        max_i = np.max(total)
        if max_i > EPS:
            total = total / max_i

    return total


def extended_source_intensity_analytic(
    x: np.ndarray,
    params: YoungParams,
    source_size: float,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """
    Аналитическая картина от протяжённого источника:

        I(x) = I0 [1 + V cos(2 pi d x / lambda L2)]
        V = sinc(b d / lambda L1)

    Берём знак sinc без модуля: отрицательный знак означает сдвиг полос на полпериода.
    Для самой видности используется модуль.
    """

    raw_v = float(
        normalized_sinc(
            source_size * params.slit_distance
            / (params.wavelength * params.source_to_slits)
        )
    )
    phase = 2.0 * np.pi * params.slit_distance * x / (
        params.wavelength * params.slits_to_screen
    )
    intensity = 1.0 + raw_v * np.cos(phase)

    if normalize:
        max_i = np.max(intensity)
        if max_i > EPS:
            intensity = intensity / max_i

    return intensity


def quasimonochromatic_intensity(
    x: np.ndarray,
    params: YoungParams,
    source_size: float,
    delta_lambda: float,
    source_points: int,
    wavelength_points: int,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """
    Численная картина для квазимонохроматического протяжённого источника.

    Алгоритм:
    1. Берём равномерную сетку источника xi.
    2. Берём равномерную сетку длин волн lambda вокруг lambda0.
    3. Для каждой пары (xi, lambda) считаем интерференционную картину.
    4. Складываем интенсивности некогерентно.

    Это медленнее аналитической формулы, но ближе к исходному принципу
    суммирования вкладов.
    """

    if delta_lambda < 0:
        raise ValueError("delta_lambda не может быть отрицательной")

    if delta_lambda >= 2.0 * params.wavelength:
        raise ValueError("delta_lambda слишком велика: спектр содержит неположительные длины волн")

    if source_size <= 0:
        xis = np.array([0.0])
    else:
        xis = np.linspace(-source_size / 2.0, source_size / 2.0, source_points)

    if delta_lambda == 0:
        wavelengths = np.array([params.wavelength])
    else:
        wavelengths = np.linspace(
            params.wavelength - delta_lambda / 2.0,
            params.wavelength + delta_lambda / 2.0,
            wavelength_points,
        )

    total = np.zeros_like(x, dtype=np.float64)
    count = 0

    original_wavelength = params.wavelength
    try:
        for lam in wavelengths:
            params.wavelength = float(lam)
            for xi in xis:
                total += numerical_point_intensity(x, params, source_x=float(xi), normalize=False)
                count += 1
    finally:
        params.wavelength = original_wavelength

    total /= max(count, 1)

    if normalize:
        max_i = np.max(total)
        if max_i > EPS:
            total = total / max_i

    return total


def print_basic_report(params: YoungParams) -> None:
    """Печать основных физических величин."""

    period = fringe_period(params)
    first_zero_b = params.wavelength * params.source_to_slits / params.slit_distance

    print("\nОсновные величины:")
    print(f"  lambda = {params.wavelength:.6e} м")
    print(f"  d      = {params.slit_distance:.6e} м")
    print(f"  L1     = {params.source_to_slits:.6e} м")
    print(f"  L2     = {params.slits_to_screen:.6e} м")
    print(f"  Период полос Lambda = lambda L2 / d = {period:.6e} м")
    print(f"  Первый нуль пространственной видности b0 = lambda L1 / d = {first_zero_b:.6e} м")


def plot_two_curves(
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    *,
    label1: str,
    label2: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    require_matplotlib()
    plt.figure(figsize=(10, 5))
    plt.plot(x, y1, label=label1, linewidth=2)
    plt.plot(x, y2, "--", label=label2, linewidth=1.5)
    plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_one_curve(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    good_level: Optional[float] = None,
) -> None:
    require_matplotlib()
    plt.figure(figsize=(9, 5))
    plt.plot(x, y, linewidth=2)
    if good_level is not None:
        plt.axhline(good_level, color="red", linestyle="--", label=f"V = {good_level:g}")
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def mode_point_source() -> None:
    """Режим 1: точечный монохроматический источник."""

    print("\n=== Режим 1. Монохроматический точечный источник ===")
    params = input_common_params()
    print_basic_report(params)

    x = screen_grid(params)
    analytic = analytic_point_intensity(x, params, exact_geometry=False, normalize=True)
    numerical = numerical_point_intensity(x, params, source_x=0.0, normalize=True)

    visibility_analytic = michelson_visibility(analytic)
    visibility_numerical = michelson_visibility(numerical)
    max_abs_error = float(np.max(np.abs(analytic - numerical)))

    print("\nРезультаты:")
    print(f"  Видность аналитически: {visibility_analytic:.6f}")
    print(f"  Видность численно:     {visibility_numerical:.6f}")
    print(f"  Max |I_analytic - I_numeric| после нормировки: {max_abs_error:.6e}")

    if ask_yes_no("Построить график интенсивности?", True):
        plot_two_curves(
            x * 1e3,
            analytic,
            numerical,
            label1="Аналитическая формула",
            label2="Численное суммирование",
            title="Интерференционная картина: точечный источник",
            xlabel="x, мм",
            ylabel="Нормированная интенсивность",
        )


def mode_geometry_visibility() -> None:
    """Режим 2: зависимости видности от геометрии."""

    print("\n=== Режим 2. Видность от геометрии установки ===")
    params = input_common_params()
    source_size = ask_float(
        "Размер протяжённого источника b, м",
        default=params.wavelength * params.source_to_slits / params.slit_distance / 4.0,
        min_value=0.0,
        allow_zero=True,
    )

    print_basic_report(params)
    v = float(spatial_visibility_analytic(source_size, params))
    print(f"\nДля b = {source_size:.6e} м аналитическая видность V = {v:.6f}")

    choice = ask_int(
        "\nЧто построить?\n"
        "  1 — V(d)\n"
        "  2 — V(L1)\n"
        "  3 — карта V(b, d)\n"
        "Ваш выбор",
        default=1,
        min_value=1,
        max_value=3,
    )

    if choice == 1:
        d_min = ask_float("Минимальное d, м", 0.1e-3, min_value=0.0)
        d_max = ask_float("Максимальное d, м", 2.0e-3, min_value=d_min)
        n = ask_int("Число точек", 500, min_value=10, max_value=100_000)

        d_values = np.linspace(d_min, d_max, n)
        old_d = params.slit_distance
        values = []
        for d in d_values:
            params.slit_distance = float(d)
            values.append(float(spatial_visibility_analytic(source_size, params)))
        params.slit_distance = old_d

        plot_one_curve(
            d_values * 1e3,
            np.array(values),
            title="Видность V(d) при фиксированном размере источника",
            xlabel="d, мм",
            ylabel="V",
            good_level=0.88,
        )

    elif choice == 2:
        l1_min = ask_float("Минимальное L1, м", 0.2, min_value=0.0)
        l1_max = ask_float("Максимальное L1, м", 5.0, min_value=l1_min)
        n = ask_int("Число точек", 500, min_value=10, max_value=100_000)

        l1_values = np.linspace(l1_min, l1_max, n)
        old_l1 = params.source_to_slits
        values = []
        for l1 in l1_values:
            params.source_to_slits = float(l1)
            values.append(float(spatial_visibility_analytic(source_size, params)))
        params.source_to_slits = old_l1

        plot_one_curve(
            l1_values,
            np.array(values),
            title="Видность V(L1) при фиксированном размере источника",
            xlabel="L1, м",
            ylabel="V",
            good_level=0.88,
        )

    else:
        require_matplotlib()
        d_min = ask_float("Минимальное d, м", 0.1e-3, min_value=0.0)
        d_max = ask_float("Максимальное d, м", 2.0e-3, min_value=d_min)
        b_max_default = 2.0 * params.wavelength * params.source_to_slits / params.slit_distance
        b_max = ask_float("Максимальное b, м", b_max_default, min_value=0.0)
        n_d = ask_int("Число точек по d", 300, min_value=20, max_value=2000)
        n_b = ask_int("Число точек по b", 300, min_value=20, max_value=2000)

        d_values = np.linspace(d_min, d_max, n_d)
        b_values = np.linspace(0.0, b_max, n_b)

        # V(b, d) = |sinc(b d / (lambda L1))|
        B, D = np.meshgrid(b_values, d_values, indexing="ij")
        V = np.abs(
            normalized_sinc(B * D / (params.wavelength * params.source_to_slits))
        )

        plt.figure(figsize=(9, 6))
        im = plt.imshow(
            V,
            extent=[d_min * 1e3, d_max * 1e3, b_values[0] * 1e3, b_values[-1] * 1e3],
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        plt.colorbar(im, label="V")
        plt.contour(
            d_values * 1e3,
            b_values * 1e3,
            V,
            levels=[0.88],
            colors="red",
            linewidths=1.5,
        )
        plt.xlabel("d, мм")
        plt.ylabel("b, мм")
        plt.title("Карта видности V(b, d); красная линия: V = 0.88")
        plt.tight_layout()
        plt.show()


def mode_quasimonochromatic() -> None:
    """Режим 3: квазимонохроматический протяжённый источник."""

    print("\n=== Режим 3. Квазимонохроматический протяжённый источник ===")
    params = input_common_params()
    source_size = ask_float(
        "Размер протяжённого источника b, м",
        default=params.wavelength * params.source_to_slits / params.slit_distance / 4.0,
        min_value=0.0,
        allow_zero=True,
    )
    delta_lambda = ask_float(
        "Ширина спектра delta_lambda, м",
        default=5e-9,
        min_value=0.0,
        allow_zero=True,
        max_value=2.0 * params.wavelength * (1.0 - 1e-6),
    )
    source_points = ask_int("Число точек источника", 51, min_value=1, max_value=5000)
    wavelength_points = ask_int("Число точек спектра", 41, min_value=1, max_value=5000)

    print_basic_report(params)
    if delta_lambda > 0:
        l_coh = params.wavelength**2 / delta_lambda
        m_max = params.wavelength / delta_lambda
        print(f"  Длина когерентности L_coh = lambda0^2 / delta_lambda = {l_coh:.6e} м")
        print(f"  Оценка числа наблюдаемых полос m_max = lambda0 / delta_lambda = {m_max:.3f}")
    else:
        print("  delta_lambda = 0: источник строго монохроматический, спектральная видность равна 1.")

    x = screen_grid(params)
    intensity = quasimonochromatic_intensity(
        x,
        params,
        source_size=source_size,
        delta_lambda=delta_lambda,
        source_points=source_points,
        wavelength_points=wavelength_points,
        normalize=True,
    )

    visibility_num = michelson_visibility(intensity)
    visibility_spatial = float(spatial_visibility_analytic(source_size, params))

    # Для оценки спектральной видности берём максимальную разность хода на краю экрана.
    delta_edge = abs(float(path_difference_paraxial(np.array([params.screen_half_width]), params)[0]))
    visibility_spectral_edge = float(
        spectral_visibility_analytic(delta_lambda, delta_edge, params.wavelength)
    )

    print("\nРезультаты:")
    print(f"  Численная видность по экрану: {visibility_num:.6f}")
    print(f"  Пространственный множитель видности V_b: {visibility_spatial:.6f}")
    print(f"  Спектральный множитель на краю экрана V_lambda(edge): {visibility_spectral_edge:.6f}")
    print(
        "  Замечание: при конечной ширине экрана спектральная видность зависит от x, "
        "поэтому одно число V_lambda(edge) является оценкой для края."
    )

    if ask_yes_no("Построить график интенсивности?", True):
        require_matplotlib()
        plt.figure(figsize=(10, 5))
        plt.plot(x * 1e3, intensity, linewidth=2)
        plt.grid(True, alpha=0.3)
        plt.xlabel("x, мм")
        plt.ylabel("Нормированная интенсивность")
        plt.title("Квазимонохроматический протяжённый источник")
        plt.tight_layout()
        plt.show()


def mode_visibility_dependences() -> None:
    """Режим 4: зависимости V(b), V(delta_lambda), V(m)."""

    print("\n=== Режим 4. Зависимости видности ===")
    params = input_common_params()
    print_basic_report(params)

    choice = ask_int(
        "\nЧто построить?\n"
        "  1 — V(b)\n"
        "  2 — V(delta_lambda)\n"
        "  3 — V(m)\n"
        "Ваш выбор",
        default=1,
        min_value=1,
        max_value=3,
    )

    if choice == 1:
        b0 = params.wavelength * params.source_to_slits / params.slit_distance
        b_max = ask_float("Максимальный размер источника b_max, м", 2.0 * b0, min_value=0.0)
        n = ask_int("Число точек", 1000, min_value=10, max_value=100_000)
        b_values = np.linspace(0.0, b_max, n)
        v_values = spatial_visibility_analytic(b_values, params)

        print(f"  Первый нуль ожидается при b0 = {b0:.6e} м")
        plot_one_curve(
            b_values * 1e3,
            v_values,
            title="Видность V(b) для протяжённого источника",
            xlabel="b, мм",
            ylabel="V",
            good_level=0.88,
        )

    elif choice == 2:
        delta_max = ask_float(
            "Максимальная ширина спектра delta_lambda_max, м",
            50e-9,
            min_value=0.0,
        )
        optical_delta = ask_float(
            "Оптическая разность хода Delta, м",
            fringe_period(params) * params.slit_distance / params.slits_to_screen * 10.0,
            min_value=0.0,
            allow_zero=True,
        )
        n = ask_int("Число точек", 1000, min_value=10, max_value=100_000)

        delta_values = np.linspace(0.0, delta_max, n)
        v_values = spectral_visibility_analytic(delta_values, optical_delta, params.wavelength)

        plot_one_curve(
            delta_values * 1e9,
            v_values,
            title="Видность V(delta_lambda) для равномерного спектра",
            xlabel="delta_lambda, нм",
            ylabel="V",
            good_level=0.88,
        )

    else:
        delta_lambda = ask_float(
            "Ширина спектра delta_lambda, м",
            5e-9,
            min_value=0.0,
            allow_zero=True,
            max_value=2.0 * params.wavelength * (1.0 - 1e-6),
        )
        if delta_lambda == 0:
            print("При delta_lambda = 0 видность V(m) = 1 для всех m.")
            return

        m_max_default = 2.0 * params.wavelength / delta_lambda
        m_max = ask_float("Максимальный порядок m", m_max_default, min_value=0.0)
        n = ask_int("Число точек", 1000, min_value=10, max_value=100_000)

        m_values = np.linspace(0.0, m_max, n)
        v_values = spectral_visibility_by_order(m_values, delta_lambda, params.wavelength)

        print(f"  Оценка максимального числа полос m_max = lambda0 / delta_lambda = {params.wavelength / delta_lambda:.3f}")
        plot_one_curve(
            m_values,
            v_values,
            title="Видность V(m) из-за конечной ширины спектра",
            xlabel="Порядок полосы m",
            ylabel="V",
            good_level=0.88,
        )


def mode_compare_extended_source() -> None:
    """Дополнительный режим: аналитика против численного расчёта для протяжённого источника."""

    print("\n=== Режим 5. Протяжённый источник: аналитика и численный расчёт ===")
    params = input_common_params()
    b0 = params.wavelength * params.source_to_slits / params.slit_distance
    source_size = ask_float(
        "Размер протяжённого источника b, м",
        default=b0 / 4.0,
        min_value=0.0,
        allow_zero=True,
    )
    source_points = ask_int("Число точек источника", 101, min_value=1, max_value=10_000)

    print_basic_report(params)
    x = screen_grid(params)
    analytic = extended_source_intensity_analytic(x, params, source_size, normalize=True)
    numerical = extended_source_intensity_numerical(
        x,
        params,
        source_size=source_size,
        source_points=source_points,
        normalize=True,
    )

    v_formula = float(spatial_visibility_analytic(source_size, params))
    v_analytic = michelson_visibility(analytic)
    v_numerical = michelson_visibility(numerical)

    print("\nРезультаты:")
    print(f"  Формула V = |sinc(b d / lambda L1)|: {v_formula:.6f}")
    print(f"  Видность аналитической кривой:          {v_analytic:.6f}")
    print(f"  Видность численной кривой:             {v_numerical:.6f}")

    if ask_yes_no("Построить график сравнения?", True):
        plot_two_curves(
            x * 1e3,
            analytic,
            numerical,
            label1="Аналитика для протяжённого источника",
            label2="Численное некогерентное суммирование",
            title="Протяжённый источник: сравнение",
            xlabel="x, мм",
            ylabel="Нормированная интенсивность",
        )


def print_menu() -> None:
    """Главное меню."""

    print("\n" + "=" * 72)
    print("M10. Интерференция в схеме Юнга")
    print("=" * 72)
    print("1 — Монохроматический точечный источник")
    print("2 — Видность от геометрии установки")
    print("3 — Квазимонохроматический протяжённый источник")
    print("4 — Зависимости V(b), V(delta_lambda), V(m)")
    print("5 — Протяжённый источник: аналитика vs численный расчёт")
    print("0 — Выход")


def main() -> None:
    """Точка входа."""

    while True:
        print_menu()
        choice = ask_int("Выберите режим", default=1, min_value=0, max_value=5)

        try:
            if choice == 0:
                print("Выход.")
                return
            if choice == 1:
                mode_point_source()
            elif choice == 2:
                mode_geometry_visibility()
            elif choice == 3:
                mode_quasimonochromatic()
            elif choice == 4:
                mode_visibility_dependences()
            elif choice == 5:
                mode_compare_extended_source()
        except KeyboardInterrupt:
            print("\nОперация прервана пользователем.")
        except Exception as exc:
            print(f"\nОшибка: {exc}")

        if not ask_yes_no("\nВернуться в главное меню?", True):
            print("Выход.")
            return


if __name__ == "__main__":
    main()
