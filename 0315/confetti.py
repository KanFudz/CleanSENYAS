import pygame
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Confetti Effect")

# Colors
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

# Confetti particle class
class Confetti:
    def __init__(self):
        self.x = random.randint(0, SCREEN_WIDTH)
        self.y = random.randint(0, SCREEN_HEIGHT // 2)
        self.color = random.choice(COLORS)
        self.width = random.randint(5, 15)
        self.height = random.randint(5, 15)
        self.speed = random.uniform(5, 8)  # Faster speed

    def fall(self):
        self.y += self.speed

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, (self.x, self.y, self.width, self.height))

# Main loop
running = True
confetti_particles = []

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                confetti_particles = [Confetti() for _ in range(100)]

    # Clear screen
    screen.fill((0, 0, 0))

    # Update and draw confetti
    for particle in confetti_particles:
        particle.fall()
        particle.draw(screen)

    # Remove confetti that falls off-screen
    confetti_particles = [p for p in confetti_particles if p.y < SCREEN_HEIGHT]

    # Update display
    pygame.display.flip()

    # Frame rate
    pygame.time.delay(20)

# Quit Pygame
pygame.quit()
