import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import sys
import unittest

EPSILON_0 = 8.8541878128e-12

def get_float_input(prompt, min_val=None, max_val=None, nonzero=False):
    while True:
        try:
            value = float(input(prompt).strip().replace(',', '.'))
            if min_val is not None and value <= min_val:
                print(f"Ошибка: значение должно быть больше {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Ошибка: значение должно быть не больше {max_val}")
                continue
            if nonzero and abs(value) < 1e-15:
                print("Ошибка: значение не должно быть равно нулю")
                continue
            return value
        except ValueError:
            print("Ошибка: введите корректное число")

def get_int_input(prompt, min_val=None, max_val=None):
    while True:
        try:
            value = int(input(prompt).strip())
            if min_val is not None and value < min_val:
                print(f"Ошибка: значение должно быть не меньше {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Ошибка: значение должно быть не больше {max_val}")
                continue
            return value
        except ValueError:
            print("Ошибка: введите целое число")

def estimate_matrix_memory_mb(n):
    m = 2 * n * n
    return (m * m * 8) / (1024 ** 2)

def build_system(L, d, V, N):
    a = L / N
    s = a * a
    k = 1.0 / (4.0 * np.pi * EPSILON_0)

    x = np.linspace(-L / 2 + a / 2, L / 2 - a / 2, N)
    y = np.linspace(-L / 2 + a / 2, L / 2 - a / 2, N)
    X, Y = np.meshgrid(x, y)

    z1 = np.full_like(X, d / 2)
    z2 = np.full_like(X, -d / 2)

    centers1 = np.column_stack((X.ravel(), Y.ravel(), z1.ravel()))
    centers2 = np.column_stack((X.ravel(), Y.ravel(), z2.ravel()))
    centers = np.vstack((centers1, centers2))

    m = len(centers)
    phi = np.empty(m)
    phi[:N * N] = V / 2
    phi[N * N:] = -V / 2

    A = np.empty((m, m), dtype=np.float64)

    for i in range(m):
        for j in range(m):
            if i == j:
                A[i, i] = k / (0.4517 * np.sqrt(s))
            else:
                rij = np.linalg.norm(centers[i] - centers[j])
                A[i, j] = k / rij

    return A, phi, centers, s

def solve_problem(L, d, V, N):
    A, phi, centers, s = build_system(L, d, V, N)
    q = la.solve(A, phi)
    q1 = np.sum(q[:N * N])
    c_num = abs(q1 / V)
    c_theory = EPSILON_0 * L * L / d
    return q, centers, s, c_num, c_theory

def compute_field_slice(centers, q, L, d):
    k = 1.0 / (4.0 * np.pi * EPSILON_0)
    grid_res = 60
    x_eval = np.linspace(-1.2 * L, 1.2 * L, grid_res)
    z_lim = 1.2 * max(L, d)
    z_eval = np.linspace(-z_lim, z_lim, grid_res)
    X, Z = np.meshgrid(x_eval, z_eval)

    Ex = np.zeros_like(X)
    Ez = np.zeros_like(Z)

    for i in range(len(q)):
        dx = X - centers[i, 0]
        dy = -centers[i, 1]
        dz = Z - centers[i, 2]
        r3 = (dx * dx + dy * dy + dz * dz) ** 1.5 + 1e-18
        Ex += k * q[i] * dx / r3
        Ez += k * q[i] * dz / r3

    return x_eval, z_eval, Ex, Ez

def plot_results(q, centers, s, L, d, V, N):
    density = (q[:N * N] / s).reshape(N, N)
    x_eval, z_eval, Ex, Ez = compute_field_slice(centers, q, L, d)
    magnitude = np.sqrt(Ex ** 2 + Ez ** 2)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    im = plt.imshow(
        density,
        extent=[-L / 2, L / 2, -L / 2, L / 2],
        origin='lower',
        cmap='viridis'
    )
    plt.title('Плотность заряда на верхней пластине')
    plt.xlabel('x, м')
    plt.ylabel('y, м')
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.subplot(1, 2, 2)
    plt.plot([-L / 2, L / 2], [d / 2, d / 2], color='red', linewidth=3, label=f'+{V/2:.3g} В')
    plt.plot([-L / 2, L / 2], [-d / 2, -d / 2], color='blue', linewidth=3, label=f'{-V/2:.3g} В')
    plt.streamplot(x_eval, z_eval, Ex, Ez, color=magnitude, cmap='magma', density=1.4)
    plt.title('Силовые линии поля в сечении y = 0')
    plt.xlabel('x, м')
    plt.ylabel('z, м')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    print("Моделирование электростатического поля методом моментов")
    print("Конфигурация: две квадратные пластины конечного размера")

    V = get_float_input("Введите разность потенциалов V, В: ", nonzero=True)
    L = get_float_input("Введите длину стороны пластины L, м: ", min_val=0.0)
    d = get_float_input("Введите расстояние между пластинами d, м: ", min_val=0.0)
    N = get_int_input("Введите число разбиений N (2..50): ", min_val=2, max_val=50)

    if d >= L:
        print("Предупреждение: d >= L, модель остаётся вычислимой, но режим уже мало похож на обычный плоский конденсатор")

    mem_mb = estimate_matrix_memory_mb(N)
    print(f"Оценка памяти под матрицу: {mem_mb:.2f} МБ")

    try:
        q, centers, s, c_num, c_theory = solve_problem(L, d, V, N)
    except MemoryError:
        print("Ошибка: недостаточно памяти для построения или решения системы")
        return
    except la.LinAlgError as e:
        print(f"Ошибка линейной алгебры: {e}")
        return
    except Exception as e:
        print(f"Ошибка: {e}")
        return

    print()
    print("Результаты")
    print(f"Ёмкость по методу моментов: {c_num:.6e} Ф")
    print(f"Теоретическая оценка:      {c_theory:.6e} Ф")
    print(f"Отношение Cчисл / Cтеор:   {c_num / c_theory:.6f}")

    plot_results(q, centers, s, L, d, V, N)

class TestElectrostatics(unittest.TestCase):
    def test_estimate_matrix_memory_mb(self):
        self.assertAlmostEqual(estimate_matrix_memory_mb(1), 32 / (1024 ** 2), places=12)

    def test_build_system_shapes(self):
        L = 1.0
        d = 0.2
        V = 1.0
        N = 3
        A, phi, centers, s = build_system(L, d, V, N)
        m = 2 * N * N
        self.assertEqual(A.shape, (m, m))
        self.assertEqual(phi.shape, (m,))
        self.assertEqual(centers.shape, (m, 3))
        self.assertAlmostEqual(s, (L / N) ** 2)

    def test_phi_values(self):
        L = 1.0
        d = 0.2
        V = 2.0
        N = 2
        A, phi, centers, s = build_system(L, d, V, N)
        np.testing.assert_allclose(phi[:N * N], np.full(N * N, V / 2))
        np.testing.assert_allclose(phi[N * N:], np.full(N * N, -V / 2))

    def test_matrix_symmetry(self):
        L = 1.0
        d = 0.2
        V = 1.0
        N = 2
        A, phi, centers, s = build_system(L, d, V, N)
        np.testing.assert_allclose(A, A.T, rtol=1e-12, atol=1e-12)

    def test_diagonal_positive(self):
        L = 1.0
        d = 0.2
        V = 1.0
        N = 2
        A, phi, centers, s = build_system(L, d, V, N)
        self.assertTrue(np.all(np.diag(A) > 0))

    def test_total_charge_opposite_signs(self):
        L = 1.0
        d = 0.2
        V = 1.0
        N = 3
        q, centers, s, c_num, c_theory = solve_problem(L, d, V, N)
        q1 = np.sum(q[:N * N])
        q2 = np.sum(q[N * N:])
        self.assertGreater(q1, 0.0)
        self.assertLess(q2, 0.0)
        self.assertAlmostEqual(q1 + q2, 0.0, delta=abs(q1) * 1e-8 + 1e-20)

    def test_capacitance_positive(self):
        L = 1.0
        d = 0.2
        V = 1.0
        N = 3
        q, centers, s, c_num, c_theory = solve_problem(L, d, V, N)
        self.assertGreater(c_num, 0.0)
        self.assertGreater(c_theory, 0.0)

    def test_theory_formula(self):
        L = 2.0
        d = 0.5
        expected = EPSILON_0 * L * L / d
        _, _, _, _, c_theory = solve_problem(L, d, 1.0, 2)
        self.assertAlmostEqual(c_theory, expected, places=20)

    def test_field_shapes(self):
        L = 1.0
        d = 0.2
        V = 1.0
        N = 2
        q, centers, s, c_num, c_theory = solve_problem(L, d, V, N)
        x_eval, z_eval, Ex, Ez = compute_field_slice(centers, q, L, d)
        self.assertEqual(Ex.shape, Ez.shape)
        self.assertEqual(Ex.shape, (60, 60))
        self.assertEqual(len(x_eval), 60)
        self.assertEqual(len(z_eval), 60)

    def test_solution_contains_finite_values(self):
        L = 1.0
        d = 0.2
        V = 1.0
        N = 3
        q, centers, s, c_num, c_theory = solve_problem(L, d, V, N)
        self.assertTrue(np.all(np.isfinite(q)))
        self.assertTrue(np.isfinite(c_num))
        self.assertTrue(np.isfinite(c_theory))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == "test":
        unittest.main(argv=[sys.argv[0]])
    else:
        try:
            main()
        except KeyboardInterrupt:
            print("\nПрограмма остановлена пользователем")
            sys.exit(0)
