import pygame
import sys
import math
import numpy as np
from pygame.locals import *

pygame.init()

WIDTH, HEIGHT = 1000, 600
BACKGROUND_COLOR = (0, 100, 0)
BALL_COLORS = [(255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255)]
TABLE_MARGIN = 50
FPS = 120
FONT = pygame.font.SysFont('Arial', 16)

METHOD_CONSERVATION = 1
METHOD_DEFORMATION = 2


class Ball:
    def __init__(self, x, y, vx, vy, radius, mass, color, id):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.mass = mass
        self.color = color
        self.id = id
        self.trail = []
        self.initial_energy = 0.5 * mass * (vx ** 2 + vy ** 2)
        self.colliding = False
        self.collision_timer = 0

    def update_position(self, dt):
        if len(self.trail) == 0 or math.hypot(self.x - self.trail[-1][0], self.y - self.trail[-1][1]) > 5:
            self.trail.append((self.x, self.y))
        if len(self.trail) > 15:
            self.trail.pop(0)

        self.x += self.vx * dt
        self.y += self.vy * dt

        if self.colliding:
            self.collision_timer += dt
            if self.collision_timer > 0.1:
                self.colliding = False
                self.collision_timer = 0

    def draw(self, screen):
        for i, (trail_x, trail_y) in enumerate(self.trail):
            alpha = int(155 * (i / len(self.trail)) + 100)
            trail_radius = max(1, int(self.radius * (i + 1) / len(self.trail)))
            trail_surface = pygame.Surface((trail_radius * 2, trail_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(trail_surface, (*self.color, alpha),
                               (trail_radius, trail_radius), trail_radius)
            screen.blit(trail_surface, (trail_x - trail_radius, trail_y - trail_radius))

        color = self.color
        if self.colliding:
            color = (min(255, color[0] + 50), min(255, color[1] + 50), min(255, color[2] + 50))

        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, (255, 255, 255), (int(self.x), int(self.y)), self.radius, 2)

        text = FONT.render(str(self.id), True, (255, 255, 255))
        screen.blit(text, (self.x - text.get_width() // 2, self.y - text.get_height() // 2))


class Table:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.balls = []
        self.collision_method = METHOD_CONSERVATION
        self.force_law = "Hooke"
        self.energy_history = []
        self.initial_total_energy = 0

    def add_ball(self, ball):
        for existing_ball in self.balls:
            dx = ball.x - existing_ball.x
            dy = ball.y - existing_ball.y
            distance = math.hypot(dx, dy)
            min_distance = ball.radius + existing_ball.radius + 1

            if distance < min_distance:
                if distance == 0:
                    angle = np.random.random() * 2 * math.pi
                else:
                    angle = math.atan2(dy, dx)

                ball.x = existing_ball.x + math.cos(angle) * min_distance
                ball.y = existing_ball.y + math.sin(angle) * min_distance
                print(f"Adjusted position of ball {ball.id} to avoid overlap")

        self.balls.append(ball)

    def handle_wall_collision(self, ball):
        collision_occurred = False

        if ball.x - ball.radius < TABLE_MARGIN:
            ball.x = TABLE_MARGIN + ball.radius
            ball.vx = -ball.vx
            collision_occurred = True

        if ball.x + ball.radius > self.width - TABLE_MARGIN:
            ball.x = self.width - TABLE_MARGIN - ball.radius
            ball.vx = -ball.vx
            collision_occurred = True

        if ball.y - ball.radius < TABLE_MARGIN:
            ball.y = TABLE_MARGIN + ball.radius
            ball.vy = -ball.vy
            collision_occurred = True

        if ball.y + ball.radius > self.height - TABLE_MARGIN:
            ball.y = self.height - TABLE_MARGIN - ball.radius
            ball.vy = -ball.vy
            collision_occurred = True

        if collision_occurred:
            ball.colliding = True
            ball.collision_timer = 0

        return collision_occurred

    def handle_ball_collision_conservation(self, ball1, ball2):
        dx = ball2.x - ball1.x
        dy = ball2.y - ball1.y
        distance = math.hypot(dx, dy)

        if distance < ball1.radius + ball2.radius and distance > 0:
            nx = dx / distance
            ny = dy / distance

            dvx = ball2.vx - ball1.vx
            dvy = ball2.vy - ball1.vy
            velocity_along_normal = dvx * nx + dvy * ny

            if velocity_along_normal > 0:
                return False

            reduced_mass = (ball1.mass * ball2.mass) / (ball1.mass + ball2.mass)
            impulse = 2 * reduced_mass * velocity_along_normal

            ball1.vx += impulse * nx / ball1.mass
            ball1.vy += impulse * ny / ball1.mass
            ball2.vx -= impulse * nx / ball2.mass
            ball2.vy -= impulse * ny / ball2.mass

            overlap = (ball1.radius + ball2.radius - distance) / 2.0
            if overlap > 0:
                ball1.x -= overlap * nx
                ball1.y -= overlap * ny
                ball2.x += overlap * nx
                ball2.y += overlap * ny

            ball1.colliding = True
            ball2.colliding = True
            ball1.collision_timer = 0
            ball2.collision_timer = 0
            return True

        return False

    def handle_ball_collision_deformation(self, ball1, ball2, dt):
        dx = ball2.x - ball1.x
        dy = ball2.y - ball1.y
        distance = math.hypot(dx, dy)

        if distance < ball1.radius + ball2.radius and distance > 0:
            nx = dx / distance
            ny = dy / distance
            penetration = ball1.radius + ball2.radius - distance

            k = 3000

            if self.force_law == "Hooke":
                force_magnitude = k * penetration
            elif self.force_law == "Hertz":
                force_magnitude = k * (penetration ** 1.5)
            else:
                force_magnitude = k * penetration

            damping = 0.3 * math.sqrt(k * min(ball1.mass, ball2.mass))

            dvx = ball2.vx - ball1.vx
            dvy = ball2.vy - ball1.vy
            relative_velocity = dvx * nx + dvy * ny
            damping_force = damping * relative_velocity

            total_force = force_magnitude - damping_force
            impulse = total_force * dt

            max_impulse = 5 * min(ball1.mass, ball2.mass) * abs(relative_velocity)
            if abs(impulse) > max_impulse:
                impulse = math.copysign(max_impulse, impulse)

            ball1.vx -= impulse * nx / ball1.mass
            ball1.vy -= impulse * ny / ball1.mass
            ball2.vx += impulse * nx / ball2.mass
            ball2.vy += impulse * ny / ball2.mass

            separation = penetration * 0.5
            ball1.x -= separation * nx
            ball1.y -= separation * ny
            ball2.x += separation * nx
            ball2.y += separation * ny

            ball1.colliding = True
            ball2.colliding = True
            ball1.collision_timer = 0
            ball2.collision_timer = 0
            return True

        return False

    def update_physics(self, dt):
        for ball in self.balls:
            ball.update_position(dt)

        for ball in self.balls:
            self.handle_wall_collision(ball)

        for _ in range(3):
            collision_occurred = False
            for i in range(len(self.balls)):
                for j in range(i + 1, len(self.balls)):
                    if self.collision_method == METHOD_CONSERVATION:
                        if self.handle_ball_collision_conservation(self.balls[i], self.balls[j]):
                            collision_occurred = True
                    else:
                        if self.handle_ball_collision_deformation(self.balls[i], self.balls[j], dt):
                            collision_occurred = True

            if not collision_occurred:
                break

    def calculate_energy(self):
        kinetic_energy = 0
        for ball in self.balls:
            speed_sq = ball.vx ** 2 + ball.vy ** 2
            kinetic_energy += 0.5 * ball.mass * speed_sq
        return kinetic_energy

    def calculate_energy_conservation(self):
        current_energy = self.calculate_energy()
        if self.initial_total_energy > 0:
            return (current_energy / self.initial_total_energy) * 100
        return 100

    def set_initial_values(self):
        self.initial_total_energy = self.calculate_energy()

    def draw(self, screen):
        pygame.draw.rect(screen, (139, 69, 19), (TABLE_MARGIN - 20, TABLE_MARGIN - 20,
                                                 self.width - 2 * TABLE_MARGIN + 40,
                                                 self.height - 2 * TABLE_MARGIN + 40))
        pygame.draw.rect(screen, BACKGROUND_COLOR, (TABLE_MARGIN, TABLE_MARGIN,
                                                    self.width - 2 * TABLE_MARGIN, self.height - 2 * TABLE_MARGIN))

        corner_hole_radius = 20
        side_hole_radius = 15
        holes = [
            (TABLE_MARGIN, TABLE_MARGIN),
            (self.width - TABLE_MARGIN, TABLE_MARGIN),
            (TABLE_MARGIN, self.height - TABLE_MARGIN),
            (self.width - TABLE_MARGIN, self.height - TABLE_MARGIN),
            (self.width // 2, TABLE_MARGIN),
            (self.width // 2, self.height - TABLE_MARGIN)
        ]
        for i, (hx, hy) in enumerate(holes):
            r = corner_hole_radius if i < 4 else side_hole_radius
            pygame.draw.circle(screen, (0, 0, 0), (hx, hy), r)

        for ball in self.balls:
            ball.draw(screen)

        energy = self.calculate_energy()
        energy_conservation = self.calculate_energy_conservation()

        self.energy_history.append(energy)
        if len(self.energy_history) > 200:
            self.energy_history.pop(0)

        method_text = FONT.render(
            f"Method: {'Conservation Laws' if self.collision_method == METHOD_CONSERVATION else 'Deformation (' + self.force_law + ')'}",
            True, (255, 255, 255))
        energy_text = FONT.render(f"Kinetic Energy: {energy:.2f} J", True, (255, 255, 255))
        energy_cons_text = FONT.render(f"Energy Conservation: {energy_conservation:.1f}%", True, (255, 255, 255))

        screen.blit(method_text, (10, 10))
        screen.blit(energy_text, (10, 30))
        screen.blit(energy_cons_text, (10, 50))

        self.draw_energy_graph(screen, 10, 70)

    def draw_energy_graph(self, screen, x, y):
        width, height = 200, 40
        if len(self.energy_history) < 2:
            return

        max_energy = max(self.energy_history) if self.energy_history else 1
        min_energy = min(self.energy_history) if self.energy_history else 0

        if max_energy == min_energy:
            max_energy = min_energy + 0.1

        pygame.draw.rect(screen, (40, 40, 40), (x, y, width, height))
        pygame.draw.rect(screen, (100, 100, 100), (x, y, width, height), 1)

        points = []
        for i, energy in enumerate(self.energy_history):
            x_pos = x + width * i / len(self.energy_history)
            y_pos = y + height - (energy - min_energy) / (max_energy - min_energy) * height * 0.9
            points.append((x_pos, y_pos))

        if len(points) > 1:
            pygame.draw.lines(screen, (255, 255, 0), False, points, 2)

        label = FONT.render("Energy History", True, (255, 255, 0))
        screen.blit(label, (x, y - 15))


def get_user_input():
    print("=== Billiard Simulation ===")
    try:
        n_balls = int(input("Enter the number of balls (1 or 2): "))
        if n_balls != 1 and n_balls != 2:
            raise ValueError("wrong value")

        print("Select collision method:")
        print("1. Conservation Laws (instantaneous)")
        print("2. Deformation (continuous)")
        method = int(input())
        if method != 1 and method != 2:
            raise ValueError("wrong value")
        method = METHOD_CONSERVATION if method == 1 else METHOD_DEFORMATION

        force_law = "Hooke"
        if method == METHOD_DEFORMATION:
            print("Select force law:")
            print("1. Hooke (F ~ -Δx)")
            print("2. Hertz (F ~ -Δx^(3/2))")
            law_choice = int(input())
            if law_choice != 1 and law_choice != 2:
                raise ValueError("wrong value")
            force_law = "Hooke" if law_choice == 1 else "Hertz"

        balls = []
        for i in range(n_balls):
            print(f"\nBall {i + 1}:")
            x = int(input("  x position (100-900): "))
            if x < 100 or x > 900:
                raise ValueError("wrong value")
            y = int(input("  y position (100-500): "))
            if y < 100 or y > 500:
                raise ValueError("wrong value")
            vx = int(input("  x velocity (-200 to 200): "))
            if vx < -200 or vx > 200:
                raise ValueError("wrong value")
            vy = int(input("  y velocity (-200 to 200): "))
            if vy < -200 or vy > 200:
                raise ValueError("wrong value")
            radius = int(input("  radius (10-40): "))
            if radius < 10 or radius > 40:
                raise ValueError("wrong value")
            mass = int(input("  mass (5-100): "))
            if mass < 5 or mass > 100:
                raise ValueError("wrong value")
            balls.append((x, y, vx, vy, radius, mass, BALL_COLORS[i % len(BALL_COLORS)], i + 1))

        return balls, method, force_law
    except Exception as e:
        print(f"Input error: {e}. Using default configuration.")
        return [
            (300, 300, 150, 80, 25, 20, BALL_COLORS[0], 1),
            (600, 300, -100, -50, 25, 20, BALL_COLORS[1], 2)
        ], METHOD_CONSERVATION, "Hooke"


def main():
    balls_data, method, force_law = get_user_input()

    table = Table(WIDTH, HEIGHT)
    table.collision_method = method
    table.force_law = force_law

    for ball_data in balls_data:
        x, y, vx, vy, radius, mass, color, id = ball_data
        table.add_ball(Ball(x, y, vx, vy, radius, mass, color, id))

    table.set_initial_values()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Billiard Simulation - Clean Version")
    clock = pygame.time.Clock()

    running = True
    paused = False
    slow_motion = False

    while running:
        dt = clock.tick(FPS) / 1000.0
        if slow_motion:
            dt *= 0.3

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    paused = not paused
                elif event.key == K_r:
                    table = Table(WIDTH, HEIGHT)
                    table.collision_method = method
                    table.force_law = force_law
                    for ball_data in balls_data:
                        x, y, vx, vy, radius, mass, color, id = ball_data
                        table.add_ball(Ball(x, y, vx, vy, radius, mass, color, id))
                    table.set_initial_values()
                elif event.key == K_s:
                    slow_motion = not slow_motion
                elif event.key == K_1:
                    table.collision_method = METHOD_CONSERVATION
                elif event.key == K_2:
                    table.collision_method = METHOD_DEFORMATION

        if not paused:
            table.update_physics(dt)

        screen.fill((0, 0, 0))
        table.draw(screen)

        instructions = [
            "SPACE: Pause/Resume, R: Reset, S: Slow Motion",
            "1: Conservation Laws, 2: Deformation Method"
        ]
        for i, instruction in enumerate(instructions):
            text = FONT.render(instruction, True, (255, 255, 255))
            screen.blit(text, (WIDTH - text.get_width() - 10, HEIGHT - 60 + i * 20))

        pygame.display.flip()

    pygame.quit()
    sys.exit()
