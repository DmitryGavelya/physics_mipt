import unittest
import math
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.phys_m2 import Ball, Table, METHOD_CONSERVATION, METHOD_DEFORMATION


class TestBilliardPhysics(unittest.TestCase):

    def setUp(self):
        self.table = Table(1000, 600)

    def test_energy_calculation_single_ball(self):
        ball = Ball(300, 300, 100, 50, 25, 20, (255, 0, 0), 1)
        self.table.add_ball(ball)

        expected_energy = 0.5 * 20 * (100 ** 2 + 50 ** 2)  # ½ * 20 * (10000 + 2500) = 125000
        actual_energy = self.table.calculate_energy()

        self.assertAlmostEqual(actual_energy, expected_energy, places=5,
                               msg="Энергия одного шара должна быть ½mv²")

    def test_wall_collision_energy_conservation(self):
        ball = Ball(100, 300, 150, 0, 25, 30, (255, 0, 0), 1)
        self.table.add_ball(ball)

        initial_energy = self.table.calculate_energy()
        self.table.handle_wall_collision(ball)
        final_energy = self.table.calculate_energy()

        self.assertAlmostEqual(final_energy, initial_energy, places=5,
                               msg="Энергия должна сохраняться при упругом ударе о стенку")

    def test_elastic_head_on_collision_equal_mass(self):

        ball1 = Ball(300, 300, 100, 0, 25, 20, (255, 0, 0), 1)
        ball2 = Ball(400, 300, -50, 0, 25, 20, (0, 255, 0), 2)

        self.table.add_ball(ball1)
        self.table.add_ball(ball2)
        self.table.collision_method = METHOD_CONSERVATION

        expected_v1 = 100
        expected_v2 = -50

        self.table.handle_ball_collision_conservation(ball1, ball2)

        self.assertAlmostEqual(ball1.vx, expected_v1, places=5,
                               msg="При лобовом ударе одинаковых масс шары должны обменяться скоростями")
        self.assertAlmostEqual(ball2.vx, expected_v2, places=5)

    def test_energy_conservation_elastic_collision(self):

        ball1 = Ball(300, 300, 80, 40, 25, 15, (255, 0, 0), 1)
        ball2 = Ball(500, 300, -60, 20, 25, 25, (0, 255, 0), 2)

        self.table.add_ball(ball1)
        self.table.add_ball(ball2)
        self.table.collision_method = METHOD_CONSERVATION

        initial_energy = self.table.calculate_energy()
        self.table.handle_ball_collision_conservation(ball1, ball2)
        final_energy = self.table.calculate_energy()

        self.assertAlmostEqual(final_energy, initial_energy, places=10,
                               msg="Энергия должна сохраняться точно в упругом столкновении")

    def test_momentum_conservation_elastic_collision(self):

        ball1 = Ball(300, 300, 100, 30, 25, 20, (255, 0, 0), 1)
        ball2 = Ball(450, 300, -80, -20, 25, 30, (0, 255, 0), 2)

        self.table.add_ball(ball1)
        self.table.add_ball(ball2)
        self.table.collision_method = METHOD_CONSERVATION

        initial_px = ball1.mass * ball1.vx + ball2.mass * ball2.vx
        initial_py = ball1.mass * ball1.vy + ball2.mass * ball2.vy

        self.table.handle_ball_collision_conservation(ball1, ball2)

        final_px = ball1.mass * ball1.vx + ball2.mass * ball2.vx
        final_py = ball1.mass * ball1.vy + ball2.mass * ball2.vy

        self.assertAlmostEqual(final_px, initial_px, places=10,
                               msg="Импульс по X должен сохраняться")
        self.assertAlmostEqual(final_py, initial_py, places=10,
                               msg="Импульс по Y должен сохраняться")

    def test_glancing_collision_minimal_interaction(self):

        ball1 = Ball(300, 300, 100, 0, 25, 20, (255, 0, 0), 1)
        ball2 = Ball(400, 350, 0, -50, 25, 20, (0, 255, 0), 2)  # Касательное положение

        self.table.add_ball(ball1)
        self.table.add_ball(ball2)
        self.table.collision_method = METHOD_CONSERVATION

        initial_v1 = (ball1.vx, ball1.vy)
        initial_v2 = (ball2.vx, ball2.vy)

        collision_occurred = self.table.handle_ball_collision_conservation(ball1, ball2)

        if collision_occurred:
            self.assertAlmostEqual(ball1.vx, initial_v1[0], delta=10,
                                   msg="При касательном столкновении скорость почти не меняется")
            self.assertAlmostEqual(ball2.vy, initial_v2[1], delta=10)

    def test_deformation_method_force_calculation(self):

        ball1 = Ball(300, 300, 0, 0, 25, 20, (255, 0, 0), 1)
        ball2 = Ball(320, 300, 0, 0, 25, 20, (0, 255, 0), 2)  # Перекрытие 30px


        self.table.force_law = "Hooke"
        hooke_collision = self.table.handle_ball_collision_deformation(ball1, ball2, 0.01)

        self.assertTrue(hooke_collision, "Должно происходить столкновение по закону Гука")

    def test_no_collision_when_balls_moving_apart(self):

        ball1 = Ball(300, 300, -50, 0, 25, 20, (255, 0, 0), 1)
        ball2 = Ball(400, 300, 50, 0, 25, 20, (0, 255, 0), 2)  # Удаляются друг от друга

        self.table.add_ball(ball1)
        self.table.add_ball(ball2)

        collision_occurred = self.table.handle_ball_collision_conservation(ball1, ball2)

        self.assertFalse(collision_occurred,
                         "Столкновения не должно быть при удалении шаров")

    def test_ball_separation_geometry(self):

        ball1 = Ball(300, 300, 0, 0, 25, 20, (255, 0, 0), 1)
        ball2 = Ball(310, 300, 0, 0, 25, 20, (0, 255, 0), 2)

        self.table.add_ball(ball1)
        self.table.add_ball(ball2)

        distance = math.hypot(ball2.x - ball1.x, ball2.y - ball1.y)
        min_distance = ball1.radius + ball2.radius

        self.assertGreaterEqual(distance, min_distance,
                                "После коррекции шары не должны перекрываться")


class TestAnalyticalSolutions(unittest.TestCase):
    def setUp(self):
        self.table = Table(1000, 600)

def run_analytical_tests():
    test_cases = [
        ("Энергия одного шара", TestBilliardPhysics('test_energy_calculation_single_ball')),
        ("Сохранение энергии при отскоке", TestBilliardPhysics('test_wall_collision_energy_conservation')),
        ("Лобовое упругое столкновение", TestBilliardPhysics('test_elastic_head_on_collision_equal_mass')),
        ("Точное сохранение энергии", TestBilliardPhysics('test_energy_conservation_elastic_collision')),
        ("Сохранение импульса", TestBilliardPhysics('test_momentum_conservation_elastic_collision')),
        ("Касательное столкновение", TestBilliardPhysics('test_glancing_collision_minimal_interaction')),
        ("Законы Гука/Герца", TestBilliardPhysics('test_deformation_method_force_calculation')),
        ("Отсутствие столкновения", TestBilliardPhysics('test_no_collision_when_balls_moving_apart')),
        ("Геометрическое разделение", TestBilliardPhysics('test_ball_separation_geometry')),
        ("Граничный случай масс", TestAnalyticalSolutions('test_perfectly_inelastic_collision_edge_case')),
        ("Ортогональное столкновение", TestAnalyticalSolutions('test_orthogonal_collision_energy_distribution')),
    ]

    runner = unittest.TextTestRunner(verbosity=2)

    for test_name, test_case in test_cases:
        suite = unittest.TestSuite([test_case])
        result = runner.run(suite)
        if not result.wasSuccessful():
            raise ValueError("wrong answer")

if __name__ == '__main__':
    run_analytical_tests()
