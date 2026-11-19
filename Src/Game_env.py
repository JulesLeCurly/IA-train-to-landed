import pygame
import math
import time
from PIL import Image
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class GameConfig:
    """Game configuration constants"""
    SCREEN_WIDTH = 540
    SCREEN_HEIGHT = 540
    GROUND_LEVEL = 70
    PLAYER_SIZE = 15

    SCREEN_SCALE = 3
    
    # Physics constants
    GRAVITY = 0.01  # m/s^2 per frame
    THRUST_POWER = 0.1  # acceleration when thrust is applied
    ROTATION_SPEED = 1  # degrees per frame
    
    # Victory conditions
    MAX_LANDING_SPEED = 1.0  # m/s
    LANDING_ANGLE_TOLERANCE = 10  # degrees from vertical


class Player:
    """Player spacecraft class with physics"""
    
    def __init__(self, config: GameConfig):
        self.config = config
        self.reset()
        
    def reset(self):
        """Reset player to initial random state"""
        import random
        self.x = self.config.SCREEN_WIDTH / 2
        self.y = self.config.SCREEN_HEIGHT / 2
        self.velocity_x = random.uniform(-10, 10)
        self.velocity_y = random.uniform(0, 10)
        self.rotation = random.randint(1, 360)
        self.thrust_active = False
        """
        self.velocity_x = 0
        self.velocity_y = 0
        self.rotation = 1
        """
    def update(self, dt: float = 1.0):
        """Update player physics"""
        # Apply gravity
        self.velocity_y -= self.config.GRAVITY
        
        # Apply thrust if active
        if self.thrust_active:
            thrust_angle_rad = math.radians(self.rotation)
            self.velocity_x += math.sin(thrust_angle_rad) * self.config.THRUST_POWER
            self.velocity_y += math.cos(thrust_angle_rad) * self.config.THRUST_POWER
        
        # Update position
        self.x += self.velocity_x
        self.y -= self.velocity_y
        
        # Wrap around screen horizontally
        if self.x > self.config.SCREEN_WIDTH:
            self.x = 0
        elif self.x < 0:
            self.x = self.config.SCREEN_WIDTH
            
        # Normalize rotation
        self.rotation = self.rotation % 360
        if self.rotation == 0:
            self.rotation = 360
    
    def rotate_left(self):
        """Rotate counterclockwise"""
        self.rotation -= self.config.ROTATION_SPEED
        
    def rotate_right(self):
        """Rotate clockwise"""
        self.rotation += self.config.ROTATION_SPEED
        
    def activate_thrust(self):
        """Turn on thrust"""
        self.thrust_active = True
        
    def deactivate_thrust(self):
        """Turn off thrust"""
        self.thrust_active = False
    
    @property
    def speed(self) -> float:
        """Calculate current speed magnitude"""
        return math.sqrt(self.velocity_x**2 + self.velocity_y**2)
    
    @property
    def altitude(self) -> float:
        """Calculate altitude above ground"""
        return (self.config.SCREEN_HEIGHT - self.config.GROUND_LEVEL) - self.y
    
    @property
    def velocity_angle(self) -> float:
        """Calculate velocity direction in degrees"""
        if self.velocity_x == 0 and self.velocity_y == 0:
            return 0
        angle = math.degrees(math.atan2(self.velocity_x, self.velocity_y))
        return (angle + 180) % 360 if angle != 0 else 360


def get_ai_observations(player: Player, config: GameConfig) -> Dict[str, float]:
    """
    Dictionary of observations available to the AI neural network
    
    Returns:
        Dictionary with all observable state information
    """
    return {
        # Position
        'altitude': player.altitude,
        
        # Velocity
        'velocity_x': player.velocity_x,
        'velocity_y': player.velocity_y,
        'speed': player.speed,
        'velocity_angle': player.velocity_angle,
        
        # Orientation
        'rotation': player.rotation,
        
        # Derived metrics
        'horizontal_speed': abs(player.velocity_x),
        'vertical_speed': player.velocity_y,
        'is_descending': player.velocity_y < 0,
    }


def get_ai_action_space() -> Dict[str, str]:
    """
    Dictionary defining possible actions the AI can take
    
    Returns:
        Dictionary describing all possible actions
    """
    return {
        'rotate_left': 'Rotate spacecraft counterclockwise',
        'rotate_right': 'Rotate spacecraft clockwise',
        'thrust_100': 'Activate main engine thrust',
        'thrust_50': 'Activate main engine thrust',
        'thrust_20': 'Activate main engine thrust',
        'thrust_5': 'Activate main engine thrust',
        'no_action': 'Do nothing (coast)',
    }


def ai_controller(observations: Dict[str, float]) -> Dict[str, bool]:
    """
    Placeholder for AI neural network controller
    
    Args:
        observations: Current game state observations
        
    Returns:
        Dictionary of actions to take this frame
    """
    # TODO: Replace with actual neural network
    # For now, return no actions
    return {
        'rotate_left': False,
        'rotate_right': False,
        'thrust': False,
    }


class Game:
    """Main game class"""
    
    def __init__(self, ai_controlled: bool = True):
        self.config = GameConfig()
        self.ai_controlled = ai_controlled
        
        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption("Lunar Lander")
        self.screen = pygame.display.set_mode((self.config.SCREEN_WIDTH, 
                                               self.config.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", int(64 / self.config.SCREEN_SCALE))
        
        # Load assets (placeholder colors if images not found)
        self.load_assets()
        
        # Game state
        self.player = Player(self.config)
        self.generation = 0
        self.victories = 0
        self.defeats = 0
        self.running = True
        
    def load_assets(self):
        """Load game assets"""
        # Ouvrir l'image
        Background_image = Image.open('Images/Background.png')

        # Récupérer largeur et hauteur
        width, height = Background_image.size

        self.background_ratio_height = height / self.config.SCREEN_HEIGHT

        self.background_size = (self.config.SCREEN_WIDTH, self.config.SCREEN_HEIGHT * self.background_ratio_height)

        self.background = pygame.transform.scale(
            pygame.image.load('Images/Background.png'),
            self.background_size
        )

        self.High_space = pygame.transform.scale(
            pygame.image.load('Images/High_space.png'),
            (self.config.SCREEN_WIDTH, self.config.SCREEN_WIDTH)
        )

        self.player_sprite = pygame.transform.scale(
            pygame.image.load('Images/Rocket.png'),
            (self.config.PLAYER_SIZE, self.config.PLAYER_SIZE)
        )
        self.player_thrust_sprite = pygame.transform.scale(
            pygame.image.load('Images/Rocket_Fire.png'),
            (self.config.PLAYER_SIZE, self.config.PLAYER_SIZE)
        )
        
    
    def create_simple_sprite(self, thrust: bool) -> pygame.Surface:
        """Create a simple triangle sprite"""
        size = self.config.PLAYER_SIZE
        sprite = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Draw triangle pointing up
        points = [(size//2, 0), (0, size), (size, size)]
        pygame.draw.polygon(sprite, (200, 200, 200), points)
        
        if thrust:
            # Draw thrust flame
            flame_points = [(size//2, size), (size//3, size + size//3), 
                          (2*size//3, size + size//3)]
            pygame.draw.polygon(sprite, (255, 100, 0), flame_points)
        
        return sprite
    
    def handle_input(self):
        """Handle player or AI input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    self.reset_game()
        
        if self.ai_controlled:
            # Get observations and AI decision
            observations = get_ai_observations(self.player, self.config)
            actions = ai_controller(observations)
            
            # Apply AI actions
            if actions['rotate_left']:
                self.player.rotate_left()
            if actions['rotate_right']:
                self.player.rotate_right()
            if actions['thrust']:
                self.player.activate_thrust()
            else:
                self.player.deactivate_thrust()
        else:
            # Manual control
            keys = pygame.key.get_pressed()
            mouse_pressed = pygame.mouse.get_pressed()[0]
            mouse_pos = pygame.mouse.get_pos()
            
            if keys[pygame.K_LEFT] or (mouse_pressed and mouse_pos[0] < self.config.SCREEN_WIDTH / 2):
                self.player.rotate_left()
            if keys[pygame.K_RIGHT] or (mouse_pressed and mouse_pos[0] > self.config.SCREEN_WIDTH / 2):
                self.player.rotate_right()
            if keys[pygame.K_SPACE] or (mouse_pressed and mouse_pos[1] > self.config.SCREEN_HEIGHT / 2):
                self.player.activate_thrust()
            else:
                self.player.deactivate_thrust()
    
    def check_collision(self) -> Tuple[bool, bool]:
        """
        Check for landing or crash
        
        Returns:
            (has_landed, is_success)
        """
        ground_y = self.config.SCREEN_HEIGHT - self.config.GROUND_LEVEL
        
        if self.player.y >= ground_y:
            # Check landing conditions
            is_slow_enough = self.player.speed < self.config.MAX_LANDING_SPEED
            is_upright = (self.player.rotation > 360 - self.config.LANDING_ANGLE_TOLERANCE or 
                         self.player.rotation < self.config.LANDING_ANGLE_TOLERANCE)
            
            return True, (is_slow_enough and is_upright)
        
        return False, False
    
    def draw(self):
        """Render the game"""
        print(self.player.altitude)
        y = int(self.config.SCREEN_HEIGHT - self.background_size[1])
        if self.player.altitude > self.config.SCREEN_HEIGHT / 2:
            y += self.player.altitude - (self.config.SCREEN_HEIGHT / 2)
        self.screen.blit(self.background, (0, y))
        #print(int(self.background_size[1] - self.player.altitude), int(self.background_size[1]), int(self.player.altitude))
        
        # Draw ground line
        ground_y = self.config.SCREEN_HEIGHT - self.config.GROUND_LEVEL
        if self.player.altitude > self.config.SCREEN_HEIGHT / 2:
            ground_y = ground_y + (self.player.altitude - (self.config.SCREEN_HEIGHT / 2))
        pygame.draw.line(self.screen, (255, 0, 0), 
                        (0, ground_y), (self.config.SCREEN_WIDTH, ground_y), 2)
        
        # Draw player
        sprite = self.player_thrust_sprite if self.player.thrust_active else self.player_sprite
        rotated_sprite = pygame.transform.rotate(sprite, 360 - self.player.rotation)
        if self.player.altitude > self.config.SCREEN_HEIGHT / 2:
            sprite_rect = rotated_sprite.get_rect(center=(self.player.x, self.config.SCREEN_HEIGHT / 2))
        else:
            sprite_rect = rotated_sprite.get_rect(center=(self.player.x, self.player.y))
        self.screen.blit(rotated_sprite, sprite_rect)
        
        # Draw HUD
        self.draw_hud()
        
        pygame.display.flip()
    
    def draw_hud(self):
        """Draw heads-up display"""
        y_offset = 0
        line_height = int(64 / self.config.SCREEN_SCALE)
        
        # Speed and velocity
        text = self.font.render(
            f"Speed: {self.player.speed:.2f} m/s | Vx: {self.player.velocity_x:.2f} | Vy: {self.player.velocity_y:.2f}",
            True, (0, 255, 0)
        )
        self.screen.blit(text, (10, y_offset))
        y_offset += line_height
        
        # Rotation and altitude
        text = self.font.render(
            f"Rotation: {self.player.rotation:.0f}° | Altitude: {self.player.altitude:.0f}",
            True, (255, 255, 0)
        )
        self.screen.blit(text, (10, y_offset))
        y_offset += line_height
        
        # Stats
        text = self.font.render(
            f"Gen: {self.generation} | Wins: {self.victories} | Fails: {self.defeats}",
            True, (0, 200, 255)
        )
        self.screen.blit(text, (10, y_offset))
        y_offset += line_height
        
        # Mode
        mode = "AI" if self.ai_controlled else "Manual"
        text = self.font.render(f"Mode: {mode}", True, (255, 255, 255))
        self.screen.blit(text, (10, y_offset))
    
    def reset_game(self):
        """Reset for next attempt"""
        self.player.reset()
        self.generation += 1
        time.sleep(0.2)
    
    def run(self):
        """Main game loop"""
        while self.running:
            self.handle_input()
            self.player.update()
            
            # Check for landing/crash
            has_landed, is_success = self.check_collision()
            if has_landed:
                if is_success:
                    self.victories += 1
                    print(f"SUCCESS! Rotation: {self.player.rotation:.0f}°, Speed: {self.player.speed:.2f} m/s")
                else:
                    self.defeats += 1
                    print(f"CRASH! Rotation: {self.player.rotation:.0f}°, Speed: {self.player.speed:.2f} m/s")
                
                time.sleep(2)
                self.reset_game()
            
            self.draw()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()


if __name__ == "__main__":
    # Set ai_controlled=False for manual play, True for AI
    game = Game(ai_controlled=False)
    game.run()