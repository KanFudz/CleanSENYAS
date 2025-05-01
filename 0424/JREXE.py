import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pygame
import sys
import cv2
import json
import time
import tensorflow.lite as tflite
import mediapipe as mp
import numpy as np
import threading
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import socket
import traceback

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def get_collision_rect(img):
    """ Returns a rect representing the non-transparent area of an image """
    mask = pygame.mask.from_surface(img)
    if mask.count() == 0:
        return img.get_rect()
    
    collision_rect = mask.get_bounding_rects()[0]
    # Adjust the collision rect to match the full image position
    full_rect = img.get_rect()
    collision_rect.x += full_rect.x
    collision_rect.y += full_rect.y
    return collision_rect

def format_lessons(lessons, max_length=25):
    """Format lessons into lines of a specified maximum length."""
    formatted_lines = []
    current_line = ""
    for lesson in lessons:
        if len(current_line) + len(lesson) + 2 > max_length:
            formatted_lines.append(current_line)
            current_line = lesson
        else:
            if current_line:
                current_line += ", " + lesson
            else:
                current_line = lesson
    if current_line:
        formatted_lines.append(current_line)
    return formatted_lines

class State:
    def __init__(self, game):
        self.game = game
    
    def enter(self):
        pass
    
    def exit(self):
        pass
    
    def handle_event(self, event):
        pass
    
    def update(self):
        pass
    
    def render(self):
        pass

class VideoState(State):
    def __init__(self, game, video_key, next_state=None, audio_file=None):
        super().__init__(game)
        self.video_key = video_key
        self.next_state = next_state
        self.audio_file = audio_file
        self.sound = None if audio_file is None else pygame.mixer.Sound(audio_file)

    def enter(self):
        self.game.videos[self.video_key].set(cv2.CAP_PROP_POS_FRAMES, 0)
        if self.sound:
            self.sound.play()  # Play the corresponding audio file

    def exit(self):
        if self.sound:
            self.sound.stop()  # Stop audio when exiting

    def update(self):
        ret, frame = self.game.videos[self.video_key].read()
        if ret:
            surface = pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (1024, 600)).swapaxes(0, 1))
            self.game.screen.blit(surface, (0, 0))
        else:
            if self.sound:
                self.sound.stop()  # Stop audio when video ends
            self.game.videos[self.video_key].release()
            if self.next_state:
                self.game.change_state(self.next_state)

class OnScreenKeyboardState(State):
    def __init__(self, game, initial_text=''):
        super().__init__(game)
        self.text = initial_text
        self.font = pygame.font.Font(resource_path("FONTS/ARIAL.ttf"), 36)
        
        # Load Done button image
        self.done_button_image = pygame.image.load(resource_path("BUTTONS/DONE.png")).convert_alpha()
        self.done_button_rect = self.done_button_image.get_rect()  # Adjust position as needed
        self.done_button_collision = get_collision_rect(self.done_button_image)

        self.shift = False  # Track shift key state

        # On-screen keyboard layout with a Shift key
        self.keyboard_keys = [
            "1234567890",
            "QWERTYUIOP",
            "ASDFGHJKL",
            "↑ZXCVBNM←",
            "@ ."
        ]

        self.key_width = 80
        self.key_height = 80
        self.key_margin = 10
        self.spacebar_width = 700  # Longer spacebar

        self.hovered_button = None  # Track hovered button

    def draw_keyboard(self):
        """Draw the on-screen keyboard."""
        y_offset = 100  # Starting position for the keyboard

        for row in self.keyboard_keys:
            row_width = sum(
                self.spacebar_width if key == " " else self.key_width for key in row
            ) + (len(row) - 1) * self.key_margin

            x_offset = (self.game.screen.get_width() - row_width) // 2

            for key in row:
                key_rect = pygame.Rect(x_offset, y_offset, self.spacebar_width if key == " " else self.key_width, self.key_height)

                # Draw key
                pygame.draw.rect(self.game.screen, (200, 200, 200), key_rect, border_radius=10)

                # Display uppercase or lowercase letters
                key_display = key.upper() if self.shift and key.isalpha() else key.lower()
                key_text = self.font.render(key_display, True, (0, 0, 0))
                text_rect = key_text.get_rect(center=key_rect.center)
                self.game.screen.blit(key_text, text_rect.topleft)

                # Move to the next key
                x_offset += (self.spacebar_width + self.key_margin) if key == " " else (self.key_width + self.key_margin)

            y_offset += self.key_height + self.key_margin

    def handle_keyboard_click(self, pos):
        """Handle clicking on the on-screen keyboard."""
        y_offset = 100

        for row in self.keyboard_keys:
            row_width = sum(
                self.spacebar_width if key == " " else self.key_width for key in row
            ) + (len(row) - 1) * self.key_margin

            x_offset = (self.game.screen.get_width() - row_width) // 2

            for key in row:
                key_rect = pygame.Rect(x_offset, y_offset, self.spacebar_width if key == " " else self.key_width, self.key_height)

                if key_rect.collidepoint(pos):
                    if key == "←":
                        self.text = self.text[:-1]  # Backspace
                    elif key == " ":
                        self.text += " "  # Space
                    elif key == "↑":
                        self.shift = not self.shift  # Toggle shift state
                    else:
                        self.text += key.upper() if self.shift else key.lower()  # Add character

                x_offset += (self.spacebar_width + self.key_margin) if key == " " else (self.key_width + self.key_margin)

            y_offset += self.key_height + self.key_margin

    def handle_event(self, event):
        if event.type in [pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN]:
            if self.done_button_collision.collidepoint(event.pos):
                self.hovered_button = self.done_button_collision
            else:
                self.hovered_button = None

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.done_button_collision.collidepoint(event.pos):
                self.game.change_state("playing_lgsign", self.text)
            else:
                self.handle_keyboard_click(event.pos)
        
        # Check delete button
        if event.type in [pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN]:
            if self.done_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.done_button_collision:
                    pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
                self.hovered_button = self.done_button_collision

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                self.dragging = True
                self.drag_start_y = event.pos[1]

            if self.done_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()

        # Handle external keyboard input
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:  # Enter key
                self.game.change_state("playing_lgsign", self.text)
            elif event.key == pygame.K_BACKSPACE:  # Backspace key
                self.text = self.text[:-1]
            elif event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:  # Shift key
                self.shift = not self.shift
            else:
                # Add the typed character to the text
                char = event.unicode
                if self.shift and char.isalpha():
                    char = char.upper()
                self.text += char

    def render(self):
        self.game.screen.fill((255, 255, 255))  # Clear screen

        # Draw input box
        pygame.draw.rect(self.game.screen, pygame.Color('white'), pygame.Rect(10, 10, 900, 50), 2)
        txt_surface = self.font.render(self.text, True, pygame.Color('black'))
        self.game.screen.blit(txt_surface, (15, 15))

        # Draw the on-screen keyboard
        self.draw_keyboard()

        # Draw Done button with the image 
        self.game.screen.blit(self.done_button_image, self.done_button_rect.topleft)
        if self.hovered_button == self.done_button_collision:
            pygame.draw.rect(self.game.screen, (0, 255, 0), self.done_button_collision, 3)

class VideoWithSignInState(VideoState):
    def __init__(self, game, video_key, next_state=None, audio_file=None, next_button_collision_height=50):
        super().__init__(game, video_key, next_state, audio_file)
        self.font = pygame.font.Font(None, 24)
        
        # Create input box and button using consistent approach
        self.input_box_rect = pygame.Rect(362, 300, 300, 50)
        # Create an invisible surface for the input box for consistency with button approach
        self.input_box_surf = pygame.Surface((300, 50), pygame.SRCALPHA)
        self.input_box_surf.fill((0, 0, 0, 0))  # Transparent

        # Load NEXT.png button image
        self.next_button_img = pygame.image.load(resource_path("BUTTONS/NEXT.png")).convert_alpha()
        self.next_button_rect = self.next_button_img.get_rect(topleft=(462, 370))
        self.next_button_collision = get_collision_rect(self.next_button_img)

        # Add back button
        self.back_button_img = pygame.image.load(resource_path("BUTTONS/BACK.png")).convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect()
        self.back_button_collision = get_collision_rect(self.back_button_img)
        
        # Adjust the height, width, x, and y of the collision rectangle for back button
        self.text = ''
        self.active = False
        self.last_frame = None
        self.hovered_button = None

    def enter(self):
        super().enter()
        if self.game.current_state_data:
            self.text = self.game.current_state_data
        self.active = False

    def update(self):
        ret, frame = self.game.videos[self.video_key].read()
        if ret:
            self.last_frame = pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (1024, 600)).swapaxes(0, 1))
            self.game.screen.blit(self.last_frame, (0, 0))
        else:
            if self.last_frame:
                self.game.screen.blit(self.last_frame, (0, 0))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            # Check next button hover
            if self.next_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.next_button_collision:
                    pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
                self.hovered_button = self.next_button_collision
            # Check back button hover
            elif self.back_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.back_button_collision:
                    pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
                self.hovered_button = self.back_button_collision
            else:
                self.hovered_button = None
                
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.input_box_rect.collidepoint(event.pos):
                self.active = not self.active
                self.game.change_state("on_screen_keyboard", self.text)
            else:
                self.active = False
                
            # Handle next button click
            if self.next_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                print(f"Entered text: {self.text}")
                # Save the profile name and create a save file
                if self.text.strip():  # Only save if text isn't empty
                    self.save_profile(self.text)
                # Go to home screen
                self.game.change_state("playing_home")
                
            # Handle back button click
            elif self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                # Go back to blgsign screen
                self.game.change_state("playing_blgsign")
        
        if event.type == pygame.KEYDOWN:
            if self.active:
                if event.key == pygame.K_RETURN:
                    print(f"Entered text: {self.text}")
                    # Save the profile name and create a save file
                    if self.text.strip():  # Only save if text isn't empty
                        self.save_profile(self.text)
                    # Go to home screen
                    self.game.change_state("playing_home")
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode

    def save_profile(self, profile_name):
        """Save the profile name to a save file"""
        # Set current profile in the game
        self.game.current_profile = profile_name
        
        from datetime import datetime
        current_date = datetime.now().strftime("%m/%d/%Y")

        # Create save directory if it doesn't exist
        if not os.path.exists("saves"):
            os.makedirs("saves")
        
        # Create a new save file with initial data
        save_data = {
            "name": profile_name,
            "created at": current_date,
            "progress": {
                "completed lessons": []
            }
        }
        
        # Save to JSON file
        with open(f"saves/{profile_name}.json", "w") as f:
            json.dump(save_data, f, indent=4)
        
        print(f"Profile '{profile_name}' saved successfully.")

    def render(self):
        if self.last_frame:
            self.game.screen.blit(self.last_frame, (0, 0))
            
        # Draw input box
        pygame.draw.rect(self.game.screen, pygame.Color('white'), self.input_box_rect, 2)
        txt_surface = self.font.render(self.text, True, pygame.Color('black'))
        self.game.screen.blit(txt_surface, (self.input_box_rect.x + 5, self.input_box_rect.y + (self.input_box_rect.height - txt_surface.get_height()) // 2))
        
        # Draw NEXT button
        self.game.screen.blit(self.next_button_img, self.next_button_rect.topleft)
        
        # Draw BACK button
        self.game.screen.blit(self.back_button_img, self.back_button_rect.topleft)
        
        # Draw highlight if any button is hovered
        if self.hovered_button == self.next_button_collision:
            pygame.draw.rect(self.game.screen, (0, 255, 0), self.next_button_collision, 3)
        elif self.hovered_button == self.back_button_collision:
            pygame.draw.rect(self.game.screen, (0, 255, 0), self.back_button_collision, 3)

class WelcomeState(State):
    def __init__(self, game, background_video, button_data, audio_file=None):
        super().__init__(game)
        self.background_video = background_video
        self.buttons = []
        
        # Load button images properly
        for img_path, _, state in button_data:
            img = pygame.image.load(resource_path(img_path)).convert_alpha()
            img_rect = img.get_rect()
            collision_rect = get_collision_rect(img)
            self.buttons.append((img, img_rect, collision_rect, state))
            
        self.audio_file = audio_file
        self.sound = None if audio_file is None else pygame.mixer.Sound(audio_file)
        self.last_frame = None
        self.video_started = False
        self.buttons_active = True  
        self.hovered_button = None  

    def enter(self):
        self.video_started = False
        self.buttons_active = True  
        self.hovered_button = None  
        self.game.videos[self.background_video].set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.game.videos[self.background_video].read()
        if ret:
            self.last_frame = pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (1024, 600)).swapaxes(0, 1))

    def exit(self):
        self.game.videos[self.background_video].release()
        if self.sound:
            self.sound.stop()  # Stop audio when exiting

    def update(self):
        if not self.video_started:
            if self.last_frame:
                self.game.screen.blit(self.last_frame, (0, 0))
            for image, rect, collision_rect, _ in self.buttons:
                self.game.screen.blit(image, rect.topleft)

                if collision_rect == self.hovered_button:
                    pygame.draw.rect(self.game.screen, (0, 255, 0), collision_rect, 3)  # GREEN highlight on hover

        else:
            ret, frame = self.game.videos[self.background_video].read()
            if ret:
                self.last_frame = pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (1024, 600)).swapaxes(0, 1))
                self.game.screen.blit(self.last_frame, (0, 0))
            else:
                self.game.videos[self.background_video].release()
                self.game.change_state("playing_intro")

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION and self.buttons_active:
            hovered = None
            for _, _, collision_rect, _ in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    hovered = collision_rect
                    break

            if hovered and hovered != self.hovered_button:
                pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
            
            self.hovered_button = hovered

        if event.type == pygame.MOUSEBUTTONDOWN and self.buttons_active:
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                    if self.sound:
                        self.sound.play()  # Play the corresponding audio file
                    self.video_started = True
                    self.buttons_active = False  
                    break

class UserTypeState(VideoState):
    def __init__(self, game, video_key, button_data, audio_file=None):
        super().__init__(game, video_key, None, audio_file)
        self.buttons = []
        
        # Load button images properly
        for img_path, _, state in button_data:
            img = pygame.image.load(img_path).convert_alpha()
            img_rect = img.get_rect()
            collision_rect = get_collision_rect(img)
            self.buttons.append((img, img_rect, collision_rect, state))
            
        self.last_frame = None
        self.video_finished = False
        self.buttons_active = True  
        self.hovered_button = None  

    def enter(self):
        super().enter()
        self.video_finished = False
        self.buttons_active = True  
        self.hovered_button = None  

    def exit(self):
        super().exit()

    def update(self):
        if not self.video_finished:
            ret, frame = self.game.videos[self.video_key].read()
            if ret:
                self.last_frame = pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (1024, 600)).swapaxes(0, 1))
                self.game.screen.blit(self.last_frame, (0, 0))
            else:
                self.video_finished = True  
        else:
            if self.last_frame:
                self.game.screen.blit(self.last_frame, (0, 0))
            for image, rect, collision_rect, _ in self.buttons:
                self.game.screen.blit(image, rect.topleft)

                if collision_rect == self.hovered_button:
                    pygame.draw.rect(self.game.screen, (0, 255, 0), collision_rect, 3)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION and self.buttons_active:
            hovered = None
            for _, _, collision_rect, _ in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    hovered = collision_rect
                    break

            if hovered and hovered != self.hovered_button:
                pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()

            self.hovered_button = hovered

        if event.type == pygame.MOUSEBUTTONDOWN and self.video_finished and self.buttons_active:
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                    self.game.change_state(state)
                    self.buttons_active = False  
                    break

# New class for the BLGSIGN state with NEW GAME and LOAD GAME buttons
class BLGSignState(State):
    def __init__(self, game, video_key, button_data, audio_file=None):
        super().__init__(game)
        self.video_key = video_key
        self.buttons = []
        
        # Load button images properly
        for img_path, _, state in button_data:
            img = pygame.image.load(img_path).convert_alpha()
            img_rect = img.get_rect()
            collision_rect = get_collision_rect(img)
            self.buttons.append((img, img_rect, collision_rect, state))
        
        # Load back button image
        self.back_button_img = pygame.image.load(resource_path("BUTTONS/BACK.png")).convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect()
        self.back_button_collision = get_collision_rect(self.back_button_img)

        self.audio_file = audio_file
        self.sound = None if audio_file is None else pygame.mixer.Sound(audio_file)
        self.last_frame = None
        self.video_finished = False
        self.buttons_active = True  
        self.hovered_button = None  

    def enter(self):
        self.game.videos[self.video_key].set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.video_finished = False
        self.buttons_active = True  
        self.hovered_button = None  
        if self.sound:
            self.sound.play()  # Play the corresponding audio file

    def exit(self):
        self.game.videos[self.video_key].release()
        if self.sound:
            self.sound.stop()  # Stop audio when exiting

    def update(self):
        if not self.video_finished:
            ret, frame = self.game.videos[self.video_key].read()
            if ret:
                self.last_frame = pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (1024, 600)).swapaxes(0, 1))
                self.game.screen.blit(self.last_frame, (0, 0))
            else:
                self.video_finished = True  
        
        # Always display the buttons after the video is finished
        if self.video_finished:
            if self.last_frame:
                self.game.screen.blit(self.last_frame, (0, 0))
            for image, rect, collision_rect, _ in self.buttons:
                self.game.screen.blit(image, rect.topleft)

                if collision_rect == self.hovered_button:
                    pygame.draw.rect(self.game.screen, (0, 255, 0), collision_rect, 3)  # Cyan highlight
            
            # Draw the back button
            self.game.screen.blit(self.back_button_img, self.back_button_rect.topleft)
            if self.hovered_button == self.back_button_collision:
                pygame.draw.rect(self.game.screen, (0, 255, 0), self.back_button_collision, 3)  # Cyan highlight

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION and self.buttons_active:
            hovered = None
            for _, _, collision_rect, _ in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    hovered = collision_rect
                    break

            if hovered and hovered != self.hovered_button:
                pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()

            self.hovered_button = hovered

            # Check if the back button is hovered
            if self.back_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.back_button_collision:
                    pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
                self.hovered_button = self.back_button_collision

        if event.type == pygame.MOUSEBUTTONDOWN and self.video_finished and self.buttons_active:
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                    self.game.change_state(state)
                    break

            # Check if the back button is clicked
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                self.game.change_state("playing_usertype")

class LoadGameState(State):
    def __init__(self, game, video_key, audio_file=None):
        super().__init__(game)
        self.video_key = video_key
        self.audio_file = audio_file
        self.sound = None if audio_file is None else pygame.mixer.Sound(audio_file)
        self.last_frame = None
        self.font = pygame.font.Font(None, 30)
        self.small_font = pygame.font.Font(None, 24)  # Smaller font for save data
        self.profile_buttons = []
        self.selected_profile_data = None  # To store the loaded profile data
        self.lines_to_draw = []  # Initialize lines_to_draw as an instance variable
        
        # Load back button image
        self.back_button_img = pygame.image.load(resource_path("BUTTONS/BACK.png")).convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect()
        self.back_button_collision = get_collision_rect(self.back_button_img)

        # Load LOAD and DELETE button images
        self.load_button_img = pygame.image.load(resource_path("BUTTONS/LOAD.png")).convert_alpha()
        self.load_button_rect = self.load_button_img.get_rect()
        self.load_button_collision = get_collision_rect(self.load_button_img)

        self.delete_button_img = pygame.image.load(resource_path("BUTTONS/DELETE.png")).convert_alpha()
        self.delete_button_rect = self.delete_button_img.get_rect()
        self.delete_button_collision = get_collision_rect(self.delete_button_img)

        # Load background image
        self.background_img = pygame.image.load(resource_path("SCENES/SENYASPIC.png")).convert()

        # Initialize scroll settings for both profile list and data panel
        self.hovered_button = None
        self.selected_profile = None  # Track the selected profile
        
        # Profile scroll settings
        self.scroll_offset = 0
        self.dragging = False
        self.drag_start_y = 0
        
        # Data panel scroll settings - match the profile scroll implementation
        self.data_scroll_offset = 0
        self.data_dragging = False
        self.data_drag_start_y = 0
        
        # Define data panel dimensions
        self.data_panel = pygame.Rect(50, 150, 280, 400)
        
        # Define bottom margin for content to prevent overlap
        self.bottom_margin = 10  # Margin at the bottom of the data panel
        
        # Define colors for different data types
        self.colors = {
            'default': pygame.Color('white'),
            'profile_name': pygame.Color('green'),
            'created_at': pygame.Color('green'),
            'galaxy_explorer': pygame.Color('blue'),
            'cosmic_copy': pygame.Color('red'),
            'star_quest': pygame.Color('orange')
        }

    def wrap_text(self, text, font, max_width):
        """Wrap text to fit within a given width."""
        words = text.split(' ')
        lines = []
        current_line = []
        current_width = 0
        
        # Handle case where text has no spaces
        if len(words) == 1:
            text_width = font.size(text)[0]
            if text_width <= max_width:
                return [text]
            
            # Split long text without spaces
            result = []
            current = ''
            for char in text:
                test = current + char
                if font.size(test)[0] <= max_width:
                    current = test
                else:
                    result.append(current)
                    current = char
            if current:
                result.append(current)
            return result

        # Normal word wrapping
        for word in words:
            word_surface = font.render(word, True, pygame.Color('white'))
            word_width = word_surface.get_width()
            
            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width + font.size(' ')[0]
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
                current_width = word_width

        if current_line:
            lines.append(' '.join(current_line))
        return lines
        
    def enter(self):
        # Ensure the video starts from the beginning
        self.game.videos[self.video_key].set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Get the first frame immediately to ensure we have something to display
        ret, frame = self.game.videos[self.video_key].read()
        if ret:
            self.last_frame = pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (1024, 600)).swapaxes(0, 1))
        
        self.load_profiles()
        if self.sound:
            self.sound.play()
        
    def exit(self):
        if self.sound:
            self.sound.stop()
        
    def load_profiles(self):
        """Load existing profiles from the saves directory"""
        self.profile_buttons = []
        
        if not os.path.exists("saves"):
            return
        
        save_files = [f for f in os.listdir("saves") if f.endswith(".json")]
        
        y_position = 150
        padding_x = 20  # Horizontal padding
        padding_y = 15  # Vertical padding
        button_width = 400  # Base width
        
        for i, file in enumerate(save_files):
            profile_name = file[:-5]  # Remove the .json extension
            
            # Get wrapped text lines and calculate required height
            wrapped_lines = self.wrap_text(profile_name, self.font, button_width - (padding_x * 2))
            line_height = self.font.get_linesize()
            text_height = len(wrapped_lines) * line_height
            button_height = text_height + (padding_y * 2)  # Add padding to top and bottom
            
            # Create button rect with adaptive height
            button_rect = pygame.Rect(362, y_position, button_width, button_height)
            self.profile_buttons.append((profile_name, button_rect))
            
            # Update y_position for next button with some spacing
            y_position += button_height + 10  # 10 pixels gap between buttons
    
    def load_profile_data(self, profile_name):
        """Load the data from a specific profile"""
        try:
            with open(f"saves/{profile_name}.json", "r") as f:
                self.selected_profile_data = json.load(f)
                print(f"Loaded data for profile: {profile_name}")
                
                # After loading profile data, prepare the lines to draw
                self.prepare_lines_to_draw()
        except Exception as e:
            print(f"Error loading profile data: {e}")
            self.selected_profile_data = None
            self.lines_to_draw = []
    
    def prepare_lines_to_draw(self):
        """Prepare the lines to draw for the selected profile data"""
        self.lines_to_draw = []
        if not self.selected_profile_data:
            return
            
        # Add progress information
        progress = self.selected_profile_data.get('progress', {}).get('completed_lessons', {})
        
        # Note: We're not adding "PROGRESS:" to lines_to_draw
        # It will be drawn separately as a fixed header
        
        for category, lessons in progress.items():
            category_text = f"{category.replace('_', ' ').title()}:"
            # Store category with its text to determine color later
            self.lines_to_draw.append((category_text, category))
            
            # Group lessons by type
            grouped_lessons = {}
            for lesson in lessons:
                lesson_type, lesson_value = lesson.split(": ")
                if lesson_type not in grouped_lessons:
                    grouped_lessons[lesson_type] = []
                grouped_lessons[lesson_type].append(lesson_value)
            
            for lesson_type, lesson_values in grouped_lessons.items():
                formatted_lines = format_lessons(lesson_values)
                # Add a tuple with the line text and category for color coding
                self.lines_to_draw.append((f"{lesson_type}:", category))
                for line in formatted_lines:
                    self.lines_to_draw.append((line, category))
    
    def update(self):
        # If we don't have a frame yet, try to get one
        if self.last_frame is None:
            ret, frame = self.game.videos[self.video_key].read()
            if ret:
                self.last_frame = pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (1024, 600)).swapaxes(0, 1))
    
    def handle_event(self, event):
        # Handle profile list scrolling
        if event.type == pygame.MOUSEMOTION:
            # Handle profile list dragging
            if self.dragging:
                delta_y = event.pos[1] - self.drag_start_y
                self.scroll_offset += delta_y
                # Limit scrolling based on the number of profiles
                self.scroll_offset = max(min(self.scroll_offset, 0), -max(0, (len(self.profile_buttons) - 1) * 60 - 400))
                self.drag_start_y = event.pos[1]
            # Handle data panel dragging
            elif self.data_dragging:
                delta_y = event.pos[1] - self.data_drag_start_y
                self.data_scroll_offset += delta_y
                
                # Get content dimensions
                content_start_y = self.data_panel.y + 130  # Estimated fixed header height
                visible_height = self.data_panel.height - (content_start_y - self.data_panel.y) - self.bottom_margin
                total_content_height = len(self.lines_to_draw) * 25  # line_height
                
                # Limit scrolling based on content height and bottom margin
                max_scroll = max(0, total_content_height - visible_height)
                self.data_scroll_offset = max(min(self.data_scroll_offset, 0), -max_scroll)
                
                self.data_drag_start_y = event.pos[1]
            else:
                # Check profile buttons
                self.hovered_button = None
                for _, button_rect in self.profile_buttons:
                    if button_rect.move(0, self.scroll_offset).collidepoint(event.pos):
                        if self.hovered_button != button_rect:
                            pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
                        self.hovered_button = button_rect
                        break
                
                # Check back button
                if self.back_button_collision.collidepoint(event.pos):
                    if self.hovered_button != self.back_button_collision:
                        pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
                    self.hovered_button = self.back_button_collision

                # Check load button
                if self.load_button_collision.collidepoint(event.pos):
                    if self.hovered_button != self.load_button_collision:
                        pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
                    self.hovered_button = self.load_button_collision

                # Check delete button
                if self.delete_button_collision.collidepoint(event.pos):
                    if self.hovered_button != self.delete_button_collision:
                        pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
                    self.hovered_button = self.delete_button_collision
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                # Check if the click is within the profile buttons area
                profile_area = pygame.Rect(362, 150, 300, 420)
                if profile_area.collidepoint(event.pos):
                    self.dragging = True
                    self.drag_start_y = event.pos[1]
                
                # Check if the click is within the data panel scrollable area
                # Calculate content area
                content_start_y = self.data_panel.y + 130  # Estimated fixed header height
                content_area = pygame.Rect(
                    self.data_panel.x, 
                    content_start_y,  
                    self.data_panel.width, 
                    self.data_panel.height - (content_start_y - self.data_panel.y) - self.bottom_margin
                )
                
                if content_area.collidepoint(event.pos) and self.selected_profile_data:
                    self.data_dragging = True
                    self.data_drag_start_y = event.pos[1]
                
                # Check profile buttons
                for profile_name, button_rect in self.profile_buttons:
                    if button_rect.move(0, self.scroll_offset).collidepoint(event.pos):
                        pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                        self.selected_profile = profile_name
                        self.load_profile_data(profile_name)
                        
                        # Reset data scroll offset when selecting a new profile
                        self.data_scroll_offset = 0
                        return
                
                # Check back button
                if self.back_button_collision.collidepoint(event.pos):
                    pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                    self.game.change_state("playing_blgsign")

                # Check load button
                if self.load_button_collision.collidepoint(event.pos) and self.selected_profile:
                    pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                    self.game.current_profile = self.selected_profile
                    print(f"Loaded profile: {self.selected_profile}")
                    self.game.change_state("playing_home")

                # Check delete button
                if self.delete_button_collision.collidepoint(event.pos) and self.selected_profile:
                    pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                    os.remove(f"saves/{self.selected_profile}.json")
                    print(f"Deleted profile: {self.selected_profile}")
                    self.selected_profile = None
                    self.selected_profile_data = None
                    self.lines_to_draw = []
                    self.load_profiles()
        
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                self.dragging = False
                self.data_dragging = False

    def render(self):
        # Draw background and overlay
        self.game.screen.blit(self.background_img, (0, 0))
        if self.last_frame:
            self.game.screen.blit(self.last_frame, (0, 0))
        
        overlay = pygame.Surface((1024, 600), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.game.screen.blit(overlay, (0, 0))
        
        # Draw title
        title_text = self.font.render("SELECT A PROFILE", True, pygame.Color('white'))
        title_rect = title_text.get_rect(center=(512, 50))
        self.game.screen.blit(title_text, title_rect)
        
        # Draw profile data panel
        if self.selected_profile_data:
            pygame.draw.rect(self.game.screen, (30, 30, 60), self.data_panel)
            pygame.draw.rect(self.game.screen, (0, 200, 200), self.data_panel, 2)
            
            # Draw headers and profile info with text wrapping
            header_text = self.small_font.render("PROFILE:", True, pygame.Color('white'))
            header_rect = header_text.get_rect(x=self.data_panel.x + 10, y=self.data_panel.y + 10)
            self.game.screen.blit(header_text, header_rect)
            
            # Wrap and draw profile name
            wrapped_name = self.wrap_text(self.selected_profile, self.small_font, self.data_panel.width - 20)
            y_offset = self.data_panel.y + 35
            for line in wrapped_name:
                name_text = self.small_font.render(line, True, self.colors['profile_name'])
                name_rect = name_text.get_rect(x=self.data_panel.x + 10, y=y_offset)
                self.game.screen.blit(name_text, name_rect)
                y_offset += 25

            # Draw creation date with new color
            created_at_label = self.small_font.render("CREATED AT:", True, pygame.Color('white'))
            created_at_label_rect = created_at_label.get_rect(x=self.data_panel.x + 10, y=y_offset)
            self.game.screen.blit(created_at_label, created_at_label_rect)

            created_at_value = self.small_font.render(
                f"{self.selected_profile_data.get('created at', 'N/A')}", 
                True, self.colors['created_at']
            )
            created_at_value_rect = created_at_value.get_rect(x=created_at_label_rect.right + 5, y=y_offset)
            self.game.screen.blit(created_at_value, created_at_value_rect)
            
            # Set a fixed position for the PROGRESS header with proper spacing
            progress_text = self.small_font.render("PROGRESS:", True, pygame.Color('white'))
            progress_y = y_offset + 30  # Add more spacing between created_at and progress
            progress_rect = progress_text.get_rect(x=self.data_panel.x + 10, y=progress_y)
            self.game.screen.blit(progress_text, progress_rect)

            # Define content start position with enough gap after the PROGRESS header
            content_start_y = progress_y + 30  # Increase this value for more spacing

            # Create a clipping rect for the scrollable content area
            # This will stop content from being drawn in the bottom margin area
            content_area = pygame.Rect(
                self.data_panel.x,
                content_start_y,
                self.data_panel.width,
                self.data_panel.height - (content_start_y - self.data_panel.y) - self.bottom_margin
            )

            # Save the current clip area
            original_clip = self.game.screen.get_clip()

            # Set the clip area to prevent drawing outside the content area
            self.game.screen.set_clip(content_area)

            # Draw scrollable progress content
            line_height = 25  # Reduce line height to fit more content

            for i, line_data in enumerate(self.lines_to_draw):
                line_text, category = line_data
                y_pos = content_start_y + (i * line_height) + self.data_scroll_offset
                
                # Skip rendering if position is outside the visible area
                if y_pos < content_area.top - line_height or y_pos > content_area.bottom:
                    continue
                
                # Adjust indentation based on line content
                indent = 10
                if ":" not in line_text:
                    indent = 30
                elif any(word in line_text for word in ["Alphabets", "Number", "Phrase", "Fingerspelling"]):
                    indent = 20
                
                # Determine text color based on category
                color_key = 'default'
                if category == 'galaxy_explorer':
                    color_key = 'galaxy_explorer'
                elif category == 'cosmic_copy':
                    color_key = 'cosmic_copy'
                elif category == 'star_quest':
                    color_key = 'star_quest'
                
                line_surface = self.small_font.render(line_text, True, self.colors[color_key])
                self.game.screen.blit(line_surface, (self.data_panel.x + indent, y_pos))

            # Restore the original clip area
            self.game.screen.set_clip(original_clip)
            
        # Draw profile buttons with text wrapping
        profile_height = 60
        max_visible_profiles = 7
        start_index = max(0, -self.scroll_offset // profile_height)
        end_index = min(len(self.profile_buttons), start_index + max_visible_profiles)
        
        for i in range(start_index, end_index):
            profile_name, button_rect = self.profile_buttons[i]
            adjusted_rect = button_rect.move(0, self.scroll_offset)
            
            # Draw button background and border
            pygame.draw.rect(self.game.screen, (50, 50, 100), adjusted_rect)
            border_color = (0, 255, 0) if profile_name == self.selected_profile else (255, 255, 255)
            pygame.draw.rect(self.game.screen, border_color, adjusted_rect, 2)
            
            # Wrap and draw profile name in button
            wrapped_lines = self.wrap_text(profile_name, self.font, button_rect.width - 40)  # 40 = padding_x * 2
            line_height = self.font.get_linesize()
            total_text_height = len(wrapped_lines) * line_height
            
            # Calculate starting y position to center text vertically
            y_offset = adjusted_rect.y + (adjusted_rect.height - total_text_height) // 2
            
            for line in wrapped_lines:
                text_surface = self.font.render(line, True, pygame.Color('white'))
                text_rect = text_surface.get_rect(centerx=adjusted_rect.centerx, y=y_offset)
                self.game.screen.blit(text_surface, text_rect)
                y_offset += line_height
        
        # Draw control buttons
        self.game.screen.blit(self.back_button_img, self.back_button_rect.topleft)
        self.game.screen.blit(self.load_button_img, self.load_button_rect.topleft)
        self.game.screen.blit(self.delete_button_img, self.delete_button_rect.topleft)
        
        # Draw button highlights
        if self.hovered_button:
            pygame.draw.rect(self.game.screen, (0, 255, 0), self.hovered_button, 3)
        
        # Show message if no profiles
        if not self.profile_buttons:
            no_profiles_text = self.font.render("No saved profiles found", True, pygame.Color('white'))
            no_profiles_rect = no_profiles_text.get_rect(center=(512, 300))
            self.game.screen.blit(no_profiles_text, no_profiles_rect)

class HomeState(VideoState):
    def __init__(self, game, video_key, next_state=None, audio_file=None):
        super().__init__(game, video_key, next_state, audio_file)
        
        # Load button images using consistent approach
        self.buttons = []
        
        button_data = [
            ("BUTTONS/COSMIC BUTTON.png", "playing_cosmic"),
            ("BUTTONS/GALAXY BUTTON.png", "playing_galaxy"),
            ("BUTTONS/STAR BUTTON.png", "playing_star")
        ]
        
        for img_path, state in button_data:
            img = pygame.image.load(img_path).convert_alpha()
            img_rect = img.get_rect()
            collision_rect = get_collision_rect(img)
            self.buttons.append((img, img_rect, collision_rect, state))
        
        # Add back button
        self.back_button_img = pygame.image.load(resource_path("BUTTONS/BACK.png")).convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect()
        self.back_button_collision = get_collision_rect(self.back_button_img)


        self.last_frame = None
        self.hovered_button = None
        self.font = pygame.font.Font(None, 24)

    def enter(self):
        super().enter()

    def exit(self):
        super().exit()

    def update(self):
        ret, frame = self.game.videos[self.video_key].read()
        if ret:
            self.last_frame = pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (1024, 600)).swapaxes(0, 1))
            self.game.screen.blit(self.last_frame, (0, 0))
            
            # Draw buttons on top of the video frame with their original positions
            for image, rect, collision_rect, _ in self.buttons:
                self.game.screen.blit(image, rect.topleft)
                
                # Highlight hovered button
                if collision_rect == self.hovered_button:
                    pygame.draw.rect(self.game.screen, (0, 255, 0), collision_rect, 3)  # Cyan highlight
            # Draw back button
            self.game.screen.blit(self.back_button_img, self.back_button_rect.topleft)
            if self.hovered_button == self.back_button_collision:
                pygame.draw.rect(self.game.screen, (0, 255, 0), self.back_button_collision, 3)

            # Display current profile name
            if hasattr(self.game, 'current_profile') and self.game.current_profile:
                profile_text = self.font.render(f"Profile: {self.game.current_profile}", True, pygame.Color('white'))
                self.game.screen.blit(profile_text, (10, 10))
        else:
            # If video ends, loop it
            self.game.videos[self.video_key].set(cv2.CAP_PROP_POS_FRAMES, 0)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            hovered = None
            for _, _, collision_rect, _ in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    hovered = collision_rect
                    break

            # Check back button
            if self.back_button_collision.collidepoint(event.pos):
                hovered = self.back_button_collision

            if hovered and hovered != self.hovered_button:
                pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
            
            self.hovered_button = hovered
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                    # Actually change the state now
                    print(f"Button clicked: {state}")
                    self.game.change_state(state)
                    return
                
            # Check back button
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                self.game.change_state("playing_usertype")

class Confetti:
    def __init__(self, screen_width, screen_height):
        self.x = random.randint(0, screen_width)
        self.y = random.randint(0, screen_height // 3)  # Start in top third
        self.color = random.choice([(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                                  (255, 255, 0), (255, 0, 255), (0, 255, 255)])
        self.width = random.randint(5, 15)
        self.height = random.randint(5, 15)
        self.speed = random.uniform(5, 12)
        self.rotation = random.randint(0, 360)
        self.rotation_speed = random.uniform(-8, 8)

    def fall(self):
        self.y += self.speed
        self.rotation += self.rotation_speed

    def draw(self, surface):
        # Create a surface for the rotated rectangle
        confetti_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(confetti_surface, self.color, (0, 0, self.width, self.height))
        
        # Rotate the surface
        rotated_surface = pygame.transform.rotate(confetti_surface, self.rotation)
        
        # Get the rect of the rotated surface and set its center to the confetti position
        rect = rotated_surface.get_rect(center=(self.x, self.y))
        
        # Draw the rotated surface
        surface.blit(rotated_surface, rect.topleft)

class GalaxyExplorerState(VideoState):
    def __init__(self, game, video_key, next_state=None, audio_file=None):
        super().__init__(game, video_key, next_state, audio_file)
        
        # Load button images
        self.buttons = []
        
        button_data = [
            ("BUTTONS/ALPHABETS.png", "playing_alphabets"),
            ("BUTTONS/NUMBERS.png", "playing_numbers"),
            ("BUTTONS/PHRASES.png", "playing_phrases")
        ]
        
        for img_path, state in button_data:
            img = pygame.image.load(img_path).convert_alpha()
            img_rect = img.get_rect()
            collision_rect = get_collision_rect(img)
            self.buttons.append((img, img_rect, collision_rect, state))
        
        # Add back button
        self.back_button_img = pygame.image.load(resource_path("BUTTONS/BACK.png")).convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect()
        self.back_button_collision = get_collision_rect(self.back_button_img)

        self.last_frame = None
        self.hovered_button = None

    def enter(self):
        super().enter()

    def exit(self):
        super().exit()

    def update(self):
        ret, frame = self.game.videos[self.video_key].read()
        if ret:
            self.last_frame = pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (1024, 600)).swapaxes(0, 1))
            self.game.screen.blit(self.last_frame, (0, 0))
            
            # Draw buttons on top of the video frame
            for image, rect, collision_rect, _ in self.buttons:
                self.game.screen.blit(image, rect.topleft)
                
                # Highlight hovered button
                if collision_rect == self.hovered_button:
                    pygame.draw.rect(self.game.screen, (0, 255, 0), collision_rect, 3)
            
            # Draw back button
            self.game.screen.blit(self.back_button_img, self.back_button_rect.topleft)
            if self.hovered_button == self.back_button_collision:
                pygame.draw.rect(self.game.screen, (0, 255, 0), self.back_button_collision, 3)
        else:
            # If video ends, loop it
            self.game.videos[self.video_key].set(cv2.CAP_PROP_POS_FRAMES, 0)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            # Check category buttons
            hovered = None
            for _, _, collision_rect, _ in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    hovered = collision_rect
                    break
            
            # Check back button
            if self.back_button_collision.collidepoint(event.pos):
                hovered = self.back_button_collision
            
            # Play hover sound if we have a new hovered button
            if hovered and hovered != self.hovered_button:
                pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
            
            self.hovered_button = hovered
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check category buttons
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                    # Transition to the appropriate state
                    print(f"Button clicked: {state}")
                    self.game.change_state(state)
                    return
            
            # Check back button
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                self.game.change_state("playing_home")

class GalaxyExplorerAlphabetState(VideoState):
    def __init__(self, game, video_key, next_state=None, audio_file=None):
        super().__init__(game, video_key, next_state, audio_file)
        
        # Load button images
        self.buttons = []
        
        button_data = [
            ("BUTTONS/Alphabets/GALPHAA.png", "playing_a"), ("BUTTONS/Alphabets/GALPHAB.png", "playing_b"),
            ("BUTTONS/Alphabets/GALPHAC.png", "playing_c"), ("BUTTONS/Alphabets/GALPHAD.png", "playing_d"),
            ("BUTTONS/Alphabets/GALPHAE.png", "playing_e"), ("BUTTONS/Alphabets/GALPHAF.png", "playing_f"),
            ("BUTTONS/Alphabets/GALPHAG.png", "playing_g"), ("BUTTONS/Alphabets/GALPHAH.png", "playing_h"),
            ("BUTTONS/Alphabets/GALPHAI.png", "playing_i"), ("BUTTONS/Alphabets/GALPHAJ.png", "playing_j"),
            ("BUTTONS/Alphabets/GALPHAK.png", "playing_k"), ("BUTTONS/Alphabets/GALPHAL.png", "playing_l"),
            ("BUTTONS/Alphabets/GALPHAM.png", "playing_m"), ("BUTTONS/Alphabets/GALPHAN.png", "playing_n"),
            ("BUTTONS/Alphabets/GALPHAO.png", "playing_o"), ("BUTTONS/Alphabets/GALPHAP.png", "playing_p"),
            ("BUTTONS/Alphabets/GALPHAQ.png", "playing_q"), ("BUTTONS/Alphabets/GALPHAR.png", "playing_r"),
            ("BUTTONS/Alphabets/GALPHAS.png", "playing_s"), ("BUTTONS/Alphabets/GALPHAT.png", "playing_t"),
            ("BUTTONS/Alphabets/GALPHAU.png", "playing_u"), ("BUTTONS/Alphabets/GALPHAV.png", "playing_v"),
            ("BUTTONS/Alphabets/GALPHAW.png", "playing_w"), ("BUTTONS/Alphabets/GALPHAX.png", "playing_x"),
            ("BUTTONS/Alphabets/GALPHAY.png", "playing_y"), ("BUTTONS/Alphabets/GALPHAZ.png", "playing_z")
        ]
        
        for img_path, state in button_data:
            img = pygame.image.load(img_path).convert_alpha()
            img_rect = img.get_rect()
            collision_rect = get_collision_rect(img)
            self.buttons.append((img, img_rect, collision_rect, state))
        
        # Add back button
        self.back_button_img = pygame.image.load(resource_path("BUTTONS/BACK.png")).convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect()
        self.back_button_collision = get_collision_rect(self.back_button_img)

        self.last_frame = None
        self.hovered_button = None

    def enter(self):
        super().enter()

    def exit(self):
        super().exit()

    def update(self):
        ret, frame = self.game.videos[self.video_key].read()
        if ret:
            self.last_frame = pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (1024, 600)).swapaxes(0, 1))
            self.game.screen.blit(self.last_frame, (0, 0))
            
            # Draw buttons on top of the video frame
            for image, rect, collision_rect, _ in self.buttons:
                self.game.screen.blit(image, rect.topleft)
                
                # Highlight hovered button
                if collision_rect == self.hovered_button:
                    pygame.draw.rect(self.game.screen, (0, 255, 0), collision_rect, 3)
            
            # Draw back button
            self.game.screen.blit(self.back_button_img, self.back_button_rect.topleft)
            if self.hovered_button == self.back_button_collision:
                pygame.draw.rect(self.game.screen, (0, 255, 0), self.back_button_collision, 3)
        else:
            # If video ends, loop it
            self.game.videos[self.video_key].set(cv2.CAP_PROP_POS_FRAMES, 0)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            # Check category buttons
            hovered = None
            for _, _, collision_rect, _ in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    hovered = collision_rect
                    break
            
            # Check back button
            if self.back_button_collision.collidepoint(event.pos):
                hovered = self.back_button_collision
            
            # Play hover sound if we have a new hovered button
            if hovered and hovered != self.hovered_button:
                pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
            
            self.hovered_button = hovered
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check category buttons
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                    # Transition to the appropriate state
                    print(f"Button clicked: {state}")
                    self.game.change_state(state)
                    return
            
            # Check back button
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                self.game.change_state("playing_galaxy")

class AlphabetDisplayState(State):
    def __init__(self, game, image_path, expected_letter, webcam_position=(700, 150), webcam_size=(300, 225)):
        super().__init__(game)
        self.original_image = pygame.image.load(image_path).convert_alpha()

        # Scale the image to fit within 1024x600 while maintaining aspect ratio
        self.image = pygame.transform.smoothscale(self.original_image, self.get_scaled_dimensions(self.original_image, 1024, 600))
        self.image_rect = self.image.get_rect(center=(1024 // 2, 600 // 2))  # Center the image

        # Add confetti-related attributes
        self.confetti_particles = []
        self.confetti_triggered = False

        # Load confetti sound effect
        self.confetti_sound = pygame.mixer.Sound(resource_path("AUDIO/CELEB.mp3"))

        # Back button (Initialize once)
        self.back_button_img = pygame.image.load(resource_path("BUTTONS/BACK.png")).convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect()
        self.back_button_collision = get_collision_rect(self.back_button_img)

        # Next button
        self.next_button_img = pygame.image.load(resource_path("BUTTONS/NXT.png")).convert_alpha()
        self.next_button_rect = self.next_button_img.get_rect()  # Position at top right
        self.next_button_collision = get_collision_rect(self.next_button_img)

        # Previous button
        self.prev_button_img = pygame.image.load(resource_path("BUTTONS/PREV.png")).convert_alpha()
        self.prev_button_rect = self.prev_button_img.get_rect()  # Position at top left
        self.prev_button_collision = get_collision_rect(self.prev_button_img)

        self.last_frame = None
        self.hovered_button = None

        # Webcam feed parameters
        self.webcam_position = (600, 152)
        self.webcam_size = (350, 263)
        self.webcam = None  # Initialize webcam as None

        # Load the TFLite model
        self.interpreter = tflite.Interpreter(model_path=resource_path("MODEL/asl_mlp_model_v2.tflite"))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Mediapipe Hands setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # Label map
        self.labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

        # Timer for try again message
        self.start_time = None
        self.correct = False

        # Expected letter for this state
        self.expected_letter = expected_letter
        
        # Debug mode to help diagnose issues
        self.debug_mode = False
        self.last_prediction = None
        self.last_confidence = None

    def get_scaled_dimensions(self, image, max_width, max_height):
        """Returns new dimensions for the image while maintaining aspect ratio"""
        img_width, img_height = image.get_size()
        scale_factor = min(max_width / img_width, max_height / img_height)
        return int(img_width * scale_factor), int(img_height * scale_factor)

    def enter(self):
        self.webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Initialize webcam
        self.correct = False
        self.start_time = None
        self.confetti_triggered = False
        self.confetti_particles = []
        self.last_prediction = None
        self.last_confidence = None

    def exit(self):
        if self.webcam:
            self.webcam.release()  # Release the webcam
            self.webcam = None

    def update(self):
        self.game.screen.fill((0, 0, 0))  # Clear screen
        self.game.screen.blit(self.image, self.image_rect.topleft)  # Centered display

        # Update webcam feed
        if self.webcam:
            ret, webcam_frame = self.webcam.read()
            if ret:
                # Flip the webcam frame horizontally to mirror it
                webcam_frame = cv2.flip(webcam_frame, 1)
                h, w, _ = webcam_frame.shape
                roi = webcam_frame[:, w//2:]  # Right side ROI
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                result = self.hands.process(roi_rgb)

                # Only process hand landmarks if no celebration is active
                if result.multi_hand_landmarks and not self.confetti_triggered:
                    for hand_landmarks in result.multi_hand_landmarks:
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y])  # Use only x and y coordinates
                        
                        # Convert landmarks to NumPy array and reshape
                        input_data = np.array(landmarks, dtype=np.float32).reshape(1, -1)
                        
                        # Ensure the input data has the correct shape
                        if input_data.shape[1] == self.input_details[0]['shape'][1]:
                            # Perform inference
                            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                            self.interpreter.invoke()
                            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                            prediction = np.argmax(output_data)
                            
                            # Store last prediction and confidence for debugging
                            self.last_prediction = self.labels[prediction]
                            self.last_confidence = output_data[0][prediction]
                            
                            # *** CHANGED: Reduced confidence threshold from 0.99 to 0.85 for letter "I" specifically ***
                            confidence = output_data[0][prediction]
                            confidence_threshold = 0.85 if self.expected_letter == "I" else 0.95
                            
                            if confidence >= confidence_threshold and self.labels[prediction] == self.expected_letter:
                                self.correct = True
                                if self.start_time is None:
                                    self.start_time = time.time()  # Start the timer
                                    self.save_progress()  # Save progress when correct

                                    # Trigger confetti effect when correct sign is made
                                    if not self.confetti_triggered:
                                        self.confetti_particles = [Confetti(1024, 600) for _ in range(100)]
                                        self.confetti_triggered = True
                                        self.confetti_sound.play()  # Play confetti sound when triggered
                            else:
                                # Only set to false if we're not in celebration mode
                                if not self.confetti_triggered:
                                    self.correct = False

                        # Draw landmarks
                        self.mp_drawing.draw_landmarks(
                            roi, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                        )
                elif self.confetti_triggered:
                    # We're in celebration mode, keep the correct state
                    pass
                else:
                    # No hand landmarks detected and not in celebration
                    self.correct = None

                # Display result
                if self.correct is True:
                    result_text = "Correct"
                elif self.correct is False:
                    result_text = "Try Again"  # Display "Try Again" only if a wrong gesture is performed
                else:
                    result_text = ""  # No gesture detected, display nothing

                result_surface = self.game.font.render(result_text, True, pygame.Color('white'))

                # Calculate the x-coordinate to center the text below the webcam
                result_x = self.webcam_position[0] + (self.webcam_size[0] - result_surface.get_width()) // 2
                result_y = self.webcam_position[1] + self.webcam_size[1] + 8

                self.game.screen.blit(result_surface, (result_x, result_y))
                
                # Display debug information if debug mode is enabled
                if self.debug_mode and self.last_prediction is not None:
                    debug_text = f"Prediction: {self.last_prediction}, Confidence: {self.last_confidence:.2f}"
                    debug_surface = self.game.font.render(debug_text, True, pygame.Color('yellow'))
                    self.game.screen.blit(debug_surface, (result_x, result_y + 30))

                # Convert the webcam frame to a surface and blit it
                webcam_surface = pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB), self.webcam_size).swapaxes(0, 1))
                self.game.screen.blit(webcam_surface, self.webcam_position)

        # Update and draw confetti particles
        for particle in self.confetti_particles[:]:
            particle.fall()
            particle.draw(self.game.screen)
            
            # Remove particles that fall off screen
            if particle.y > 600:
                self.confetti_particles.remove(particle)
        
        # Check if celebration has ended (no more confetti particles)
        if self.confetti_triggered and len(self.confetti_particles) == 0:
            self.confetti_triggered = False
            self.correct = False  # Reset the correct state
            self.start_time = None  # Reset the timer

        # Draw back button
        self.game.screen.blit(self.back_button_img, self.back_button_rect.topleft)

        # Draw next button
        self.game.screen.blit(self.next_button_img, self.next_button_rect.topleft)

        # Draw previous button
        self.game.screen.blit(self.prev_button_img, self.prev_button_rect.topleft)

        # Draw hover effect
        if self.hovered_button == self.back_button_collision:
            pygame.draw.rect(self.game.screen, (0, 255, 0), self.back_button_collision, 3)
        elif self.hovered_button == self.next_button_collision:
            pygame.draw.rect(self.game.screen, (0, 255, 0), self.next_button_collision, 3)
        elif self.hovered_button == self.prev_button_collision:
            pygame.draw.rect(self.game.screen, (0, 255, 0), self.prev_button_collision, 3)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            if self.back_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.back_button_collision:
                    pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
                self.hovered_button = self.back_button_collision
            elif self.next_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.next_button_collision:
                    pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
                self.hovered_button = self.next_button_collision
            elif self.prev_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.prev_button_collision:
                    pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
                self.hovered_button = self.prev_button_collision
            else:
                self.hovered_button = None
                
        # Toggle debug mode with 'D' key
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_d:
                self.debug_mode = not self.debug_mode
                print(f"Debug mode {'enabled' if self.debug_mode else 'disabled'}")

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                self.game.change_state("playing_alphabets")  # Return to alphabet selection
            elif self.next_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                # Find the next letter in sequence
                if self.game.current_state_name in self.game.alphabet_sequence:
                    current_index = self.game.alphabet_sequence.index(self.game.current_state_name)
                    if current_index < len(self.game.alphabet_sequence) - 1:
                        next_state = self.game.alphabet_sequence[current_index + 1]
                    else:
                        next_state = "playing_alphabets"  # Return to alphabet selection if last letter is reached
                else:
                    next_state = "playing_alphabets"  # Default fallback

                self.game.change_state(next_state)
            elif self.prev_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                # Find the previous letter in sequence
                if self.game.current_state_name in self.game.alphabet_sequence:
                    current_index = self.game.alphabet_sequence.index(self.game.current_state_name)
                    if current_index > 0:
                        prev_state = self.game.alphabet_sequence[current_index - 1]
                    else:
                        prev_state = "playing_alphabets"  # Return to alphabet selection if first letter is reached
                else:
                    prev_state = "playing_alphabets"  # Default fallback

                self.game.change_state(prev_state)

    def save_progress(self):
        """Save the progress of the current profile"""
        if self.game.current_profile:
            save_path = f"saves/{self.game.current_profile}.json"
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    save_data = json.load(f)
                
                # Ensure completed_lessons is a dictionary
                if not isinstance(save_data["progress"].get("completed_lessons"), dict):
                    save_data["progress"]["completed_lessons"] = {}
                
                if "galaxy_explorer" not in save_data["progress"]["completed_lessons"]:
                    save_data["progress"]["completed_lessons"]["galaxy_explorer"] = []
                
                progress_entry = f"Alphabets: {self.expected_letter}"
                if progress_entry not in save_data["progress"]["completed_lessons"]["galaxy_explorer"]:
                    save_data["progress"]["completed_lessons"]["galaxy_explorer"].append(progress_entry)
                
                with open(save_path, "w") as f:
                    json.dump(save_data, f, indent=4)
                
                print(f"Progress saved for letter: {self.expected_letter} in Galaxy Explorer")

class GalaxyExplorerNumberState(VideoState):
    def __init__(self, game, video_key, next_state=None, audio_file=None):
        super().__init__(game, video_key, next_state, audio_file)
        
        # keep the folder name relative …
        gnum_path = "BUTTONS/Numbers"

        self.buttons = []

        button_data = [
            ("GNUM0.png", "playing_0"), ("GNUM1.png", "playing_1"), ("GNUM2.png", "playing_2"),
            ("GNUM3.png", "playing_3"), ("GNUM4.png", "playing_4"), ("GNUM5.png", "playing_5"),
            ("GNUM6.png", "playing_6"), ("GNUM7.png", "playing_7"), ("GNUM8.png", "playing_8"),
            ("GNUM9.png", "playing_9")
        ]

        for gnum, state in button_data:
            full_path = resource_path(os.path.join(gnum_path, gnum))   # single wrap
            btn_surface = pygame.image.load(full_path).convert_alpha()
            btn_rect = btn_surface.get_rect()
            collision_rect = get_collision_rect(btn_surface)
            self.buttons.append((btn_surface, btn_rect, collision_rect, state))

        # Add back button
        self.back_button_img = pygame.image.load(resource_path("BUTTONS/BACK.png")).convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect()
        self.back_button_collision = get_collision_rect(self.back_button_img)
        
        self.last_frame = None
        self.hovered_button = None

    def enter(self):
        super().enter()

    def exit(self):
        super().exit()

    def update(self):
        ret, frame = self.game.videos[self.video_key].read()
        if ret:
            self.game.screen.blit(
                pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (1024, 600)).swapaxes(0, 1)), 
                (0, 0)
            )

            # Draw buttons
            for image, rect, collision_rect, _ in self.buttons:
                self.game.screen.blit(image, rect.topleft)
                if collision_rect == self.hovered_button:
                    pygame.draw.rect(self.game.screen, (0, 255, 0), collision_rect, 3)

            # Draw back button
            self.game.screen.blit(self.back_button_img, self.back_button_rect.topleft)
            if self.hovered_button == self.back_button_collision:
                pygame.draw.rect(self.game.screen, (0, 255, 0), self.back_button_collision, 3)
                
        else:
            # If video ends, loop it
            self.game.videos[self.video_key].set(cv2.CAP_PROP_POS_FRAMES, 0)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            hovered = None
            for _, _, collision_rect, _ in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    hovered = collision_rect
                    break

            if self.back_button_collision.collidepoint(event.pos):
                hovered = self.back_button_collision

            if hovered and hovered != self.hovered_button:
                pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()

            self.hovered_button = hovered

        if event.type == pygame.MOUSEBUTTONDOWN:
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                    self.game.change_state(state)
                    return

            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                self.game.change_state("playing_galaxy")

class NumberDisplayState(State):
    def __init__(self, game, image_path, expected_number, webcam_position=(700, 150), webcam_size=(300, 225)):
        super().__init__(game)
        self.original_image = pygame.image.load(image_path).convert_alpha()

        # Scale the image to fit within 1024x600 while maintaining aspect ratio
        self.image = pygame.transform.smoothscale(self.original_image, self.get_scaled_dimensions(self.original_image, 1024, 600))
        self.image_rect = self.image.get_rect(center=(1024 // 2, 600 // 2))  # Center the image

        # Add confetti-related attributes
        self.confetti_particles = []
        self.confetti_triggered = False

        # Load confetti sound effect
        self.confetti_sound = pygame.mixer.Sound(resource_path("AUDIO/CELEB.mp3"))

        # Back button (Initialize once)
        self.back_button_img = pygame.image.load(resource_path("BUTTONS/BACK.png")).convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect()
        self.back_button_collision = get_collision_rect(self.back_button_img)

        # Next button
        self.next_button_img = pygame.image.load(resource_path("BUTTONS/NXT.png")).convert_alpha()
        self.next_button_rect = self.next_button_img.get_rect()  # Position at top right
        self.next_button_collision = get_collision_rect(self.next_button_img)

        # Previous button
        self.prev_button_img = pygame.image.load(resource_path("BUTTONS/PREV.png")).convert_alpha()
        self.prev_button_rect = self.prev_button_img.get_rect()  # Position at top left
        self.prev_button_collision = get_collision_rect(self.prev_button_img)

        self.last_frame = None
        self.hovered_button = None

        # Webcam feed parameters
        self.webcam_position = (600, 152)
        self.webcam_size = (350, 263)
        self.webcam = None  # Initialize webcam as None

        # Load the TFLite model
        self.interpreter = tflite.Interpreter(model_path=resource_path("MODEL/asl_number_classifier.tflite"))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Mediapipe Hands setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # Label map
        self.labels = [str(i) for i in range(10)]

        # Timer for try again message
        self.start_time = None
        self.correct = False

        # Expected number for this state
        self.expected_number = expected_number

    def get_scaled_dimensions(self, image, max_width, max_height):
        """Returns new dimensions for the image while maintaining aspect ratio"""
        img_width, img_height = image.get_size()
        scale_factor = min(max_width / img_width, max_height / img_height)
        return int(img_width * scale_factor), int(img_height * scale_factor)

    def enter(self):
        self.webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Initialize webcam
        self.correct = False
        self.start_time = None
        self.confetti_triggered = False
        self.confetti_particles = []

    def exit(self):
        if self.webcam:
            self.webcam.release()  # Release the webcam
            self.webcam = None

    def update(self):
        self.game.screen.fill((0, 0, 0))  # Clear screen
        self.game.screen.blit(self.image, self.image_rect.topleft)  # Centered display

        # Update webcam feed
        if self.webcam:
            ret, webcam_frame = self.webcam.read()
            if ret:
                # Flip the webcam frame horizontally to mirror it
                webcam_frame = cv2.flip(webcam_frame, 1)
                h, w, _ = webcam_frame.shape
                roi = webcam_frame[:, w//2:]  # Right side ROI
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                result = self.hands.process(roi_rgb)

                # Only process hand landmarks if no celebration is active
                if result.multi_hand_landmarks and not self.confetti_triggered:
                    for hand_landmarks in result.multi_hand_landmarks:
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])  # Use x, y, and z coordinates
                        
                        # Convert landmarks to NumPy array and reshape
                        input_data = np.array(landmarks, dtype=np.float32).reshape(1, -1)
                        
                        # Ensure the input data has the correct shape
                        if input_data.shape[1] == self.input_details[0]['shape'][1]:
                            # Perform inference
                            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                            self.interpreter.invoke()
                            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                            prediction = np.argmax(output_data)
                            
                            confidence = output_data[0][prediction]
                            if confidence >= 0.70 and self.labels[prediction] == str(self.expected_number):
                                self.correct = True
                                if self.start_time is None:
                                    self.start_time = time.time()  # Start the timer
                                    self.save_progress()  # Save progress when correct

                                    # Trigger confetti effect when correct sign is made
                                    if not self.confetti_triggered:
                                        self.confetti_particles = [Confetti(1024, 600) for _ in range(100)]
                                        self.confetti_triggered = True
                                        self.confetti_sound.play()  # Play confetti sound when triggered
                            else:
                                # Only set to false if we're not in celebration mode
                                if not self.confetti_triggered:
                                    self.correct = False

                        # Draw landmarks
                        self.mp_drawing.draw_landmarks(
                            roi, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                        )
                elif self.confetti_triggered:
                    # We're in celebration mode, keep the correct state
                    pass
                else:
                    # No hand landmarks detected and not in celebration
                    self.correct = None

                # Display result
                if self.correct is True:
                    result_text = "Correct"
                elif self.correct is False:
                    result_text = "Try Again"  # Display "Try Again" only if a wrong gesture is performed
                else:
                    result_text = ""  # No gesture detected, display nothing

                result_surface = self.game.font.render(result_text, True, pygame.Color('white'))

                # Calculate the x-coordinate to center the text below the webcam
                result_x = self.webcam_position[0] + (self.webcam_size[0] - result_surface.get_width()) // 2
                result_y = self.webcam_position[1] + self.webcam_size[1] + 8

                self.game.screen.blit(result_surface, (result_x, result_y))

                # Convert the webcam frame to a surface and blit it
                webcam_surface = pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB), self.webcam_size).swapaxes(0, 1))
                self.game.screen.blit(webcam_surface, self.webcam_position)

        # Update and draw confetti particles
        for particle in self.confetti_particles[:]:
            particle.fall()
            particle.draw(self.game.screen)
            
            # Remove particles that fall off screen
            if particle.y > 600:
                self.confetti_particles.remove(particle)
        
        # Check if celebration has ended (no more confetti particles)
        if self.confetti_triggered and len(self.confetti_particles) == 0:
            self.confetti_triggered = False
            self.correct = False  # Reset the correct state
            self.start_time = None  # Reset the timer

        # Draw back button
        self.game.screen.blit(self.back_button_img, self.back_button_rect.topleft)

        # Draw next button
        self.game.screen.blit(self.next_button_img, self.next_button_rect.topleft)

        # Draw previous button
        self.game.screen.blit(self.prev_button_img, self.prev_button_rect.topleft)

        # Draw hover effect
        if self.hovered_button == self.back_button_collision:
            pygame.draw.rect(self.game.screen, (0, 255, 0), self.back_button_collision, 3)
        elif self.hovered_button == self.next_button_collision:
            pygame.draw.rect(self.game.screen, (0, 255, 0), self.next_button_collision, 3)
        elif self.hovered_button == self.prev_button_collision:
            pygame.draw.rect(self.game.screen, (0, 255, 0), self.prev_button_collision, 3)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            if self.back_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.back_button_collision:
                    pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
                self.hovered_button = self.back_button_collision
            elif self.next_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.next_button_collision:
                    pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
                self.hovered_button = self.next_button_collision
            elif self.prev_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.prev_button_collision:
                    pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
                self.hovered_button = self.prev_button_collision
            else:
                self.hovered_button = None

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                self.game.change_state("playing_numbers")  # Return to number selection
            elif self.next_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                # Find the next number in sequence
                if self.game.current_state_name in self.game.number_sequence:
                    current_index = self.game.number_sequence.index(self.game.current_state_name)
                    if current_index < len(self.game.number_sequence) - 1:
                        next_state = self.game.number_sequence[current_index + 1]
                    else:
                        next_state = "playing_numbers"  # Return to number selection if last number is reached
                else:
                    next_state = "playing_numbers"  # Default fallback

                self.game.change_state(next_state)
            elif self.prev_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                # Find the previous number in sequence
                if self.game.current_state_name in self.game.number_sequence:
                    current_index = self.game.number_sequence.index(self.game.current_state_name)
                    if current_index > 0:
                        prev_state = self.game.number_sequence[current_index - 1]
                    else:
                        prev_state = "playing_numbers"  # Return to number selection if first number is reached
                else:
                    prev_state = "playing_numbers"  # Default fallback

                self.game.change_state(prev_state)

    def save_progress(self):
        """Save the progress of the current profile"""
        if self.game.current_profile:
            save_path = f"saves/{self.game.current_profile}.json"
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    save_data = json.load(f)
                
                # Ensure completed_lessons is a dictionary
                if not isinstance(save_data["progress"].get("completed_lessons"), dict):
                    save_data["progress"]["completed_lessons"] = {}
                
                if "galaxy_explorer" not in save_data["progress"]["completed_lessons"]:
                    save_data["progress"]["completed_lessons"]["galaxy_explorer"] = []
                
                progress_entry = f"Number: {self.expected_number}"
                if progress_entry not in save_data["progress"]["completed_lessons"]["galaxy_explorer"]:
                    save_data["progress"]["completed_lessons"]["galaxy_explorer"].append(progress_entry)
                
                with open(save_path, "w") as f:
                    json.dump(save_data, f, indent=4)
                
                print(f"Progress saved for number: {self.expected_number} in Galaxy Explorer")

class GalaxyExplorerPhrasesstate(VideoState):
    def __init__(self, game, video_key, next_state=None, audio_file=None):
        super().__init__(game, video_key, next_state, audio_file)
        
        # Load button images
        self.buttons = []
        
        button_data = [
            ("BUTTONS/Phrases/THANKYOU.png", "playing_thankyou"), ("BUTTONS/Phrases/HELLO.png", "playing_hello"),
            ("BUTTONS/Phrases/ILOVEYOU.png", "playing_iloveyou"), ("BUTTONS/Phrases/SORRY.png", "playing_sorry"),                                            
        ]
        
        for img_path, state in button_data:
            img = pygame.image.load(resource_path(img_path)).convert_alpha()
            img_rect = img.get_rect()
            collision_rect = get_collision_rect(img)
            self.buttons.append((img, img_rect, collision_rect, state))
        
        # Add back button
        self.back_button_img = pygame.image.load(resource_path("BUTTONS/BACK.png")).convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect()
        self.back_button_collision = get_collision_rect(self.back_button_img)
        
        self.last_frame = None
        self.hovered_button = None

    def enter(self):
        super().enter()

    def exit(self):
        super().exit()

    def update(self):
        ret, frame = self.game.videos[self.video_key].read()
        if ret:
            self.last_frame = pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (1024, 600)).swapaxes(0, 1))
            self.game.screen.blit(self.last_frame, (0, 0))
            
            # Draw buttons on top of the video frame
            for image, rect, collision_rect, _ in self.buttons:
                self.game.screen.blit(image, rect.topleft)
                
                # Highlight hovered button
                if collision_rect == self.hovered_button:
                    pygame.draw.rect(self.game.screen, (0, 255, 0), collision_rect, 3)
            
            # Draw back button
            self.game.screen.blit(self.back_button_img, self.back_button_rect.topleft)
            if self.hovered_button == self.back_button_collision:
                pygame.draw.rect(self.game.screen, (0, 255, 0), self.back_button_collision, 3)
        else:
            # If video ends, loop it
            self.game.videos[self.video_key].set(cv2.CAP_PROP_POS_FRAMES, 0)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            # Check category buttons
            hovered = None
            for _, _, collision_rect, _ in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    hovered = collision_rect
                    break
            
            # Check back button
            if self.back_button_collision.collidepoint(event.pos):
                hovered = self.back_button_collision
            
            # Play hover sound if we have a new hovered button
            if hovered and hovered != self.hovered_button:
                pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
            
            self.hovered_button = hovered
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check category buttons
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                    # Transition to the appropriate state
                    print(f"Button clicked: {state}")
                    self.game.change_state(state)
                    return
            
            # Check back button
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                self.game.change_state("playing_galaxy")

class PhraseDisplayState(State):
    def __init__(self, game, image_path, expected_phrase, webcam_position=(700, 150), webcam_size=(300, 225)):
        super().__init__(game)
        self.original_image = pygame.image.load(resource_path(image_path)).convert_alpha()

        # Scale the image to fit within 1024x600 while maintaining aspect ratio
        self.image = pygame.transform.smoothscale(self.original_image, self.get_scaled_dimensions(self.original_image, 1024, 600))
        self.image_rect = self.image.get_rect(center=(1024 // 2, 600 // 2))  # Center the image

        # Add confetti-related attributes
        self.confetti_particles = []
        self.confetti_triggered = False

        # Load the TFLite model
        self.interpreter = tflite.Interpreter(model_path=resource_path('MODEL/gesture_model.tflite'))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load confetti sound effect
        self.confetti_sound = pygame.mixer.Sound(resource_path("AUDIO/CELEB.mp3"))

        # Back button (Initialize once)
        self.back_button_img = pygame.image.load(resource_path("BUTTONS/BACK.png")).convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect()
        self.back_button_collision = get_collision_rect(self.back_button_img)

        # Next button
        self.next_button_img = pygame.image.load(resource_path("BUTTONS/NXT.png")).convert_alpha()
        self.next_button_rect = self.next_button_img.get_rect()  # Position at top right
        self.next_button_collision = get_collision_rect(self.next_button_img)

        # Previous button
        self.prev_button_img = pygame.image.load(resource_path("BUTTONS/PREV.png")).convert_alpha()
        self.prev_button_rect = self.prev_button_img.get_rect()  # Position at top left
        self.prev_button_collision = get_collision_rect(self.prev_button_img)

        self.last_frame = None
        self.hovered_button = None
        
        # Mediapipe Holistic setup
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # Webcam feed parameters
        self.webcam_position = (600, 152)
        self.webcam_size = (350, 263)
        self.webcam = None  # Initialize webcam as None

        # Initialize sequence and gesture recognition variables
        self.sequence = []
        self.sequence_length = 30
        self.predictions = []
        self.threshold = 0.85
        self.min_consecutive_predictions = 5
        self.prediction_history_size = 15
        self.confidence_threshold = 0.90

        # Expected phrase for this state
        self.expected_phrase = expected_phrase

        # Label map
        self.actions = np.array(['hello', 'thanks', 'iloveyou', 'sorry'])
        self.colors = [(245,117,16), (117,245,16), (16,117,245)]
        
        # Correction for the model-expected phrase mapping
        self.phrase_mapping = {
            'hello': 'HELLO',
            'thanks': 'THANKYOU',  # Map "thanks" to "THANKYOU"
            'iloveyou': 'ILOVEYOU',
            'sorry': 'SORRY'
        }

    def get_scaled_dimensions(self, image, max_width, max_height):
        """Returns new dimensions for the image while maintaining aspect ratio"""
        img_width, img_height = image.get_size()
        scale_factor = min(max_width / img_width, max_height / img_height)
        return int(img_width * scale_factor), int(img_height * scale_factor)

    def enter(self):
        self.webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Initialize webcam
        self.correct = False
        self.start_time = None
        self.confetti_triggered = False
        self.confetti_particles = []
        self.sequence = []
        self.predictions = []

    def exit(self):
        if self.webcam:
            self.webcam.release()  # Release the webcam
            self.webcam = None

    def mediapipe_detection(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, lh, rh])

    def has_hands(self, results):
        return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None

    def detect_motion(self, keypoints, threshold=0.01):
        hand_points = keypoints[-126:]  # Get hand keypoints (last 126 values)
        return np.mean(np.abs(hand_points)) > threshold

    def update(self):
        self.game.screen.fill((0, 0, 0))  # Clear screen
        self.game.screen.blit(self.image, self.image_rect.topleft)  # Centered display

        # Update webcam feed
        if self.webcam:
            ret, frame = self.webcam.read()
            if ret:
                # Flip the webcam frame horizontally
                frame = cv2.flip(frame, 1)
                
                # Make detections
                image, results = self.mediapipe_detection(frame)
                
                # Process frames for gesture recognition
                if self.has_hands(results) and not self.confetti_triggered:
                    keypoints = self.extract_keypoints(results)
                    
                    if self.detect_motion(keypoints):
                        self.sequence.append(keypoints)
                        self.sequence = self.sequence[-self.sequence_length:]
                        
                        if len(self.sequence) == self.sequence_length:
                            try:
                                # Prepare input data
                                input_data = np.array(self.sequence, dtype=np.float32)
                                input_data = (input_data - np.mean(input_data)) / (np.std(input_data) + 1e-6)
                                input_data = np.expand_dims(input_data, axis=0)

                                # Set the tensor and run inference
                                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                                self.interpreter.invoke()
                                
                                # Get prediction results
                                res = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
                                max_prob = res[np.argmax(res)]
                                
                                if max_prob > self.threshold:
                                    prediction = self.actions[np.argmax(res)]
                                    self.predictions.append(prediction)
                                    
                                    if len(self.predictions) > self.prediction_history_size:
                                        self.predictions = self.predictions[-self.prediction_history_size:]
                                    
                                    if (len(self.predictions) >= self.min_consecutive_predictions and 
                                        len(set(self.predictions[-self.min_consecutive_predictions:])) == 1 and 
                                        max_prob > self.confidence_threshold):
                                        
                                        current_pred = self.predictions[-1]
                                        # Map the prediction to expected format using the mapping dictionary
                                        mapped_pred = self.phrase_mapping.get(current_pred, current_pred.upper())
                                        
                                        if mapped_pred == self.expected_phrase:
                                            self.correct = True
                                            if not self.confetti_triggered:
                                                self.confetti_particles = [Confetti(1024, 600) for _ in range(100)]
                                                self.confetti_triggered = True
                                                self.confetti_sound.play()
                                                self.save_progress()
                                        else:
                                            self.correct = False
                                    
                            except Exception as e:
                                print(f"Inference error: {e}")

                # Draw landmarks
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
                if results.left_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

                # Display result
                if self.correct is True:
                    result_text = "Correct"
                elif self.correct is False:
                    result_text = "Try Again"
                else:
                    result_text = ""

                result_surface = self.game.font.render(result_text, True, pygame.Color('white'))
                result_x = self.webcam_position[0] + (self.webcam_size[0] - result_surface.get_width()) // 2
                result_y = self.webcam_position[1] + self.webcam_size[1] + 8
                self.game.screen.blit(result_surface, (result_x, result_y))

                # Display webcam feed
                webcam_surface = pygame.surfarray.make_surface(
                    cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), self.webcam_size).swapaxes(0, 1))
                self.game.screen.blit(webcam_surface, self.webcam_position)

        # Update and draw confetti particles
        for particle in self.confetti_particles[:]:
            particle.fall()
            particle.draw(self.game.screen)
            if particle.y > 600:
                self.confetti_particles.remove(particle)
        
        # Check if celebration has ended
        if self.confetti_triggered and len(self.confetti_particles) == 0:
            self.confetti_triggered = False
            self.correct = False
            self.start_time = None

        # Draw buttons
        self.game.screen.blit(self.back_button_img, self.back_button_rect.topleft)
        self.game.screen.blit(self.next_button_img, self.next_button_rect.topleft)
        self.game.screen.blit(self.prev_button_img, self.prev_button_rect.topleft)

        # Draw hover effects
        if self.hovered_button:
            pygame.draw.rect(self.game.screen, (0, 255, 0), self.hovered_button, 3)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            if self.back_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.back_button_collision:
                    pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
                self.hovered_button = self.back_button_collision
            elif self.next_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.next_button_collision:
                    pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
                self.hovered_button = self.next_button_collision
            elif self.prev_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.prev_button_collision:
                    pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
                self.hovered_button = self.prev_button_collision
            else:
                self.hovered_button = None

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                self.game.change_state("playing_phrases")
            elif self.next_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                if self.game.current_state_name in self.game.phrase_sequence:
                    current_index = self.game.phrase_sequence.index(self.game.current_state_name)
                    if current_index < len(self.game.phrase_sequence) - 1:
                        next_state = self.game.phrase_sequence[current_index + 1]
                    else:
                        next_state = "playing_phrases"
                else:
                    next_state = "playing_phrases"
                self.game.change_state(next_state)
            elif self.prev_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                if self.game.current_state_name in self.game.phrase_sequence:
                    current_index = self.game.phrase_sequence.index(self.game.current_state_name)
                    if current_index > 0:
                        prev_state = self.game.phrase_sequence[current_index - 1]
                    else:
                        prev_state = "playing_phrases"
                else:
                    prev_state = "playing_phrases"
                self.game.change_state(prev_state)

    def save_progress(self):
        if self.game.current_profile:
            save_path = f"saves/{self.game.current_profile}.json"
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    save_data = json.load(f)
                
                if not isinstance(save_data["progress"].get("completed_lessons"), dict):
                    save_data["progress"]["completed_lessons"] = {}
                
                if "galaxy_explorer" not in save_data["progress"]["completed_lessons"]:
                    save_data["progress"]["completed_lessons"]["galaxy_explorer"] = []
                
                progress_entry = f"Phrase: {self.expected_phrase}"
                if progress_entry not in save_data["progress"]["completed_lessons"]["galaxy_explorer"]:
                    save_data["progress"]["completed_lessons"]["galaxy_explorer"].append(progress_entry)
                
                with open(save_path, "w") as f:
                    json.dump(save_data, f, indent=4)
                
                print(f"Progress saved for phrase: {self.expected_phrase} in Galaxy Explorer")
                
class CosmicCopyState(State):
    def __init__(self, game):
        super().__init__(game)
        self.current_item = None
        self.expected_value = None

        # Load the TFLite models
        self.alphabet_model = tflite.Interpreter(model_path=resource_path("MODEL/asl_mlp_model_v2.tflite"))
        self.alphabet_model.allocate_tensors()
        self.number_model = tflite.Interpreter(model_path=resource_path("MODEL/asl_number_classifier.tflite"))
        self.number_model.allocate_tensors()
        self.phrase_model = tflite.Interpreter(model_path=resource_path("MODEL/gesture_model.tflite"))
        self.phrase_model.allocate_tensors()
        
        # Input/output details for phrase model
        self.phrase_input_details = self.phrase_model.get_input_details()
        self.phrase_output_details = self.phrase_model.get_output_details()

        # Mediapipe setup for hand recognition (for alphabet and numbers)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        
        # Mediapipe Holistic setup (for phrases)
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        self.mp_drawing = mp.solutions.drawing_utils

        # Load the labels
        self.alphabet_labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        self.number_labels = [str(i) for i in range(10)]
        self.phrase_labels = ['hello', 'thankyou', 'iloveyou', 'sorry']

        # Back button
        self.back_button_img = pygame.image.load(resource_path("BUTTONS/BACK.png")).convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect()
        self.back_button_collision = get_collision_rect(self.back_button_img)

        self.hovered_button = None
        self.webcam = None  # Initialize webcam as None

        # Timer for try again message
        self.start_time = None
        self.correct = False

        # Add confetti-related attributes
        self.confetti_particles = []
        self.confetti_triggered = False
        
        # Load confetti sound effect
        self.confetti_sound = pygame.mixer.Sound(resource_path("AUDIO/CELEB.mp3"))

        # Webcam feed parameters
        self.webcam_position = (600, 152)
        self.webcam_size = (350, 263)
        
        # Track recently used items to prevent repeats
        self.recent_items = {"alphabet": [], "number": [], "phrase": []}
        self.last_category = None
        
        # Sequence variables for phrase recognition (same as in PhraseDisplayState)
        self.sequence = []
        self.sequence_length = 30
        self.predictions = []
        self.threshold = 0.85
        self.min_consecutive_predictions = 5
        self.prediction_history_size = 15
        self.confidence_threshold = 0.90
        self.phrase_actions = np.array(['hello', 'thanks', 'iloveyou', 'sorry'])

    def enter(self):
        self.webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Initialize webcam
        self.randomize_item()
        self.correct = False
        self.start_time = None
        self.confetti_particles = []  # Reset confetti particles
        self.confetti_triggered = False
        # Reset sequence and predictions for phrase recognition
        self.sequence = []
        self.predictions = []

    def exit(self):
        if self.webcam:
            self.webcam.release()  # Release the webcam
            self.webcam = None

    def randomize_item(self):
        # Choose a category, avoiding the last one if possible
        available_categories = ["alphabet", "number", "phrase"]
        if self.last_category and len(available_categories) > 1:
            available_categories.remove(self.last_category)
        
        choice = random.choice(available_categories)
        self.last_category = choice
        
        if choice == "alphabet":
            # Get list of items not recently used
            available_items = [item for item in self.alphabet_labels 
                              if item not in self.recent_items["alphabet"]]
            
            # If all items have been recently used, reset the tracking
            if not available_items:
                available_items = self.alphabet_labels
                self.recent_items["alphabet"] = []
            
            # Select a random item from available options
            self.expected_value = random.choice(available_items)
            
            # Update recent items (keep track of last 5 items)
            self.recent_items["alphabet"].append(self.expected_value)
            if len(self.recent_items["alphabet"]) > 5:
                self.recent_items["alphabet"].pop(0)
                
            self.current_item = pygame.image.load(resource_path("GAME PROPER", "COSMIC COPY ALPHABET", f"{self.expected_value}.png")).convert_alpha()
            
        elif choice == "number":
            available_items = [item for item in self.number_labels 
                              if item not in self.recent_items["number"]]
            
            if not available_items:
                available_items = self.number_labels
                self.recent_items["number"] = []
                
            self.expected_value = random.choice(available_items)
            
            self.recent_items["number"].append(self.expected_value)
            if len(self.recent_items["number"]) > 3:  # Track last 3 numbers
                self.recent_items["number"].pop(0)
                
            self.current_item = pygame.image.load(resource_path("GAME PROPER", "COSMIC COPY NUMBER", f"{self.expected_value}.png")).convert_alpha()
            
        else:  # phrase
            available_items = [item for item in self.phrase_labels 
                               if item not in self.recent_items["phrase"]]
            
            if not available_items:
                available_items = self.phrase_labels
                self.recent_items["phrase"] = []
                
            self.expected_value = random.choice(available_items)
            
            self.recent_items["phrase"].append(self.expected_value)
            if len(self.recent_items["phrase"]) > 2:  # Track last 2 phrases
                self.recent_items["phrase"].pop(0)
                
            self.current_item = pygame.image.load(resource_path("GAME PROPER", "COSMIC COPY PHRASES", f"{self.expected_value.upper()}.png")).convert_alpha()

        self.current_item = pygame.transform.smoothscale(self.current_item, self.get_scaled_dimensions(self.current_item, 1024, 600))
        self.current_item_rect = self.current_item.get_rect(center=(1024 // 2, 600 // 2))

    def get_scaled_dimensions(self, image, max_width, max_height):
        """Returns new dimensions for the image while maintaining aspect ratio"""
        img_width, img_height = image.get_size()
        scale_factor = min(max_width / img_width, max_height / img_height)
        return int(img_width * scale_factor), int(img_height * scale_factor)
    
    # New methods for phrase recognition (from PhraseDisplayState)
    def mediapipe_detection(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    
    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, lh, rh])
    
    def has_hands(self, results):
        return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None
    
    def detect_motion(self, keypoints, threshold=0.01):
        hand_points = keypoints[-126:]  # Get hand keypoints (last 126 values)
        return np.mean(np.abs(hand_points)) > threshold

    def update(self):
        self.game.screen.fill((0, 0, 0))  # Clear screen
        self.game.screen.blit(self.current_item, self.current_item_rect.topleft)  # Centered display

        # Update webcam feed
        if self.webcam:
            ret, webcam_frame = self.webcam.read()
            if ret:
                # Flip the webcam frame horizontally to mirror it
                webcam_frame = cv2.flip(webcam_frame, 1)
                
                # Handle the decision to proceed to the next item
                if self.correct is True and self.start_time:
                    if time.time() - self.start_time > 5:  # Wait for 5 seconds
                        self.randomize_item()  # Proceed to the next item
                        self.correct = False  # Reset correct status
                        self.start_time = None  # Reset start time
                        self.confetti_triggered = False  # Reset confetti trigger for next item
                        # Reset sequence and predictions for phrase recognition
                        self.sequence = []
                        self.predictions = []
                    # Continue rendering confetti, webcam, landmarks, and "Correct" message during the delay
                    self.render_webcam_and_confetti(webcam_frame, None)
                    return  # Skip further processing

                # Process the frame based on the type of expected value
                if self.expected_value in self.phrase_labels:
                    # Use holistic model and sequence-based processing for phrases
                    image, results = self.mediapipe_detection(webcam_frame)
                    
                    if self.has_hands(results) and not self.confetti_triggered:
                        keypoints = self.extract_keypoints(results)
                        
                        if self.detect_motion(keypoints):
                            self.sequence.append(keypoints)
                            self.sequence = self.sequence[-self.sequence_length:]
                            
                            if len(self.sequence) == self.sequence_length:
                                try:
                                    # Prepare input data
                                    input_data = np.array(self.sequence, dtype=np.float32)
                                    input_data = (input_data - np.mean(input_data)) / (np.std(input_data) + 1e-6)
                                    input_data = np.expand_dims(input_data, axis=0)

                                    # Set the tensor and run inference
                                    self.phrase_model.set_tensor(self.phrase_input_details[0]['index'], input_data)
                                    self.phrase_model.invoke()
                                    
                                    # Get prediction results
                                    res = self.phrase_model.get_tensor(self.phrase_output_details[0]['index'])[0]
                                    max_prob = res[np.argmax(res)]
                                    
                                    if max_prob > self.threshold:
                                        prediction = self.phrase_actions[np.argmax(res)]
                                        self.predictions.append(prediction)
                                        
                                        if len(self.predictions) > self.prediction_history_size:
                                            self.predictions = self.predictions[-self.prediction_history_size:]
                                        
                                        if (len(self.predictions) >= self.min_consecutive_predictions and 
                                            len(set(self.predictions[-self.min_consecutive_predictions:])) == 1 and 
                                            max_prob > self.confidence_threshold):
                                            
                                            current_pred = self.predictions[-1]
                                            # Map the prediction to the expected format
                                            if current_pred == "thanks":
                                                current_pred = "thankyou"
                                                
                                            if current_pred == self.expected_value:
                                                if not self.correct:  # Ensure the logic runs only once
                                                    self.correct = True
                                                    self.start_time = time.time()  # Start the timer
                                                    self.save_progress("phrase", self.expected_value)

                                                    # Trigger confetti effect when correct
                                                    if not self.confetti_triggered:
                                                        self.confetti_particles = [Confetti(1024, 600) for _ in range(100)]
                                                        self.confetti_triggered = True
                                                        self.confetti_sound.play()  # Play confetti sound when triggered
                                            else:
                                                self.correct = False
                                        
                                except Exception as e:
                                    print(f"Phrase inference error: {e}")
                    
                    # Draw holistic landmarks
                    if results.pose_landmarks:
                        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
                    if results.left_hand_landmarks:
                        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                    if results.right_hand_landmarks:
                        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                    
                    # Render updated webcam and result
                    self.render_webcam_and_confetti(image, None)
                    
                else:
                    # Use hands model for alphabet and numbers (original logic)
                    h, w, _ = webcam_frame.shape
                    roi = webcam_frame[:, w//2:]  # Right side ROI
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    result = self.hands.process(roi_rgb)

                    if result.multi_hand_landmarks:
                        for hand_landmarks in result.multi_hand_landmarks:
                            landmarks = []
                            for lm in hand_landmarks.landmark:
                                if self.expected_value in self.alphabet_labels:
                                    landmarks.extend([lm.x, lm.y])  # Use only x and y coordinates for alphabet
                                else:
                                    landmarks.extend([lm.x, lm.y, lm.z])  # Use x, y, and z coordinates for numbers

                            # Convert landmarks to NumPy array and reshape
                            input_data = np.array(landmarks, dtype=np.float32).reshape(1, -1)

                            # Perform inference based on the type of expected value
                            if self.expected_value in self.alphabet_labels:
                                self.interpreter = self.alphabet_model
                            else:  # number
                                self.interpreter = self.number_model

                            self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], input_data)
                            self.interpreter.invoke()
                            output_data = self.interpreter.get_tensor(self.interpreter.get_output_details()[0]['index'])
                            prediction = np.argmax(output_data)

                            # Check if the prediction is correct
                            if (self.expected_value in self.alphabet_labels and prediction < len(self.alphabet_labels) and self.expected_value == self.alphabet_labels[prediction]) or \
                               (self.expected_value in self.number_labels and prediction < len(self.number_labels) and self.expected_value == self.number_labels[prediction]):
                                if not self.correct:  # Ensure the logic runs only once
                                    self.correct = True
                                    self.start_time = time.time()  # Start the timer
                                    self.save_progress("alphabet" if self.expected_value in self.alphabet_labels else "number", self.expected_value)

                                    # Trigger confetti effect when correct
                                    if not self.confetti_triggered:
                                        self.confetti_particles = [Confetti(1024, 600) for _ in range(100)]
                                        self.confetti_triggered = True
                                        self.confetti_sound.play()  # Play confetti sound when triggered
                            else:
                                self.correct = False

                            # Draw landmarks
                            self.mp_drawing.draw_landmarks(
                                roi, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                            )
                    else:
                        # No hand landmarks detected, do not display "Try Again"
                        self.correct = None

                    # Render webcam, confetti, and result
                    self.render_webcam_and_confetti(webcam_frame, result)

        # Draw back button
        self.game.screen.blit(self.back_button_img, self.back_button_rect.topleft)

        # Draw hover effect
        if self.hovered_button == self.back_button_collision:
            pygame.draw.rect(self.game.screen, (0, 255, 0), self.back_button_collision, 3)

    def render_webcam_and_confetti(self, webcam_frame, result):
        """Render the webcam feed, confetti particles, landmarks, and result text."""
        # Convert the webcam frame to a surface and blit it
        webcam_surface = pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB), self.webcam_size).swapaxes(0, 1))
        self.game.screen.blit(webcam_surface, self.webcam_position)

        # Update and draw confetti particles
        for particle in self.confetti_particles[:]:
            particle.fall()
            particle.draw(self.game.screen)

            # Remove particles that fall off screen
            if particle.y > 600 or particle.x < 0 or particle.x > 1024:
                self.confetti_particles.remove(particle)

        # Draw landmarks for hand detection if available (for alphabet and numbers)
        if result and result.multi_hand_landmarks and not self.expected_value in self.phrase_labels:
            for hand_landmarks in result.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    webcam_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

        # Display result text
        if self.correct is True:
            result_text = "Correct"
        elif self.correct is False:
            result_text = "Try Again"
        else:
            result_text = ""

        result_surface = self.game.font.render(result_text, True, pygame.Color('white'))
        result_x = self.webcam_position[0] + (self.webcam_size[0] - result_surface.get_width()) // 2
        result_y = self.webcam_position[1] + self.webcam_size[1] + 10
        self.game.screen.blit(result_surface, (result_x, result_y))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            if self.back_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.back_button_collision:
                    pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
                self.hovered_button = self.back_button_collision
            else:
                self.hovered_button = None

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                self.correct = False  # Reset correct status
                self.start_time = None  # Reset start time
                self.confetti_triggered = False  # Reset confetti trigger when going back
                self.game.change_state("playing_home")

    def save_progress(self, item_type, item_value):
        """Save the progress of the current profile"""
        if self.game.current_profile:
            save_path = f"saves/{self.game.current_profile}.json"
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    save_data = json.load(f)
                
                # Ensure completed_lessons is a dictionary
                if not isinstance(save_data["progress"].get("completed_lessons"), dict):
                    save_data["progress"]["completed_lessons"] = {}
                
                if "cosmic_copy" not in save_data["progress"]["completed_lessons"]:
                    save_data["progress"]["completed_lessons"]["cosmic_copy"] = []

                if item_type == "alphabet":
                    progress_entry = f"Alphabets: {item_value}"
                elif item_type == "number":
                    progress_entry = f"Numbers: {item_value}"
                elif item_type == "phrase":
                    progress_entry = f"Phrase: {item_value}"
                else:
                    return  # Invalid item type

                if progress_entry not in save_data["progress"]["completed_lessons"]["cosmic_copy"]:
                    save_data["progress"]["completed_lessons"]["cosmic_copy"].append(progress_entry)
                    save_data["progress"]["completed_lessons"]["cosmic_copy"].sort()  # Sort the entries
                
                with open(save_path, "w") as f:
                    json.dump(save_data, f, indent=4)
                
                print(f"Progress saved for {item_type}: {item_value} in Cosmic Copy")

class StarQuestState(State):
    def __init__(self, game):
        super().__init__(game)
        self.levels = [
            ("CAT", ["C", "A", "T"]),
            ("DOG", ["D", "O", "G"]),
            ("SUN", ["S", "U", "N"]),
            ("FISH", ["F", "I", "S", "H"]),
            ("ELEPHANT", ["E", "L", "E", "P", "H", "A", "N", "T"])
        ]
        self.current_level = 0
        self.current_step = 0
        self.correct = False
        self.start_time = None
        self.word_transition_time = None  # New attribute for word transition delay
        self.load_images()
        self.load_model()
        self.setup_mediapipe()
        self.webcam = None
        self.webcam_position = (600, 152)
        self.webcam_size = (350, 263)
        self.hovered_button = None

        # Back button
        self.back_button_img = pygame.image.load(resource_path("BUTTONS/BACK.png")).convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect()
        self.back_button_collision = get_collision_rect(self.back_button_img)

    def load_images(self):
        self.images = {}
        for level, _ in self.levels:
            for i in range(len(level) + 1):
                img_path = resource_path(os.path.join("GAME PROPER", "STAR QUEST", f"{level}_{i}.png"))
                self.images[f"{level}_{i}"] = pygame.image.load(img_path).convert_alpha()

    def load_model(self):
        self.interpreter = tflite.Interpreter(model_path=resource_path("MODEL/asl_mlp_model_v2.tflite"))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

    def setup_mediapipe(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def enter(self):
        self.webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Initialize webcam
        self.current_level = 0
        self.current_step = 0
        self.correct = False
        self.start_time = None
        self.confetti_particles = []
        self.celebration_active = False
        self.celebration_start_time = None

    def exit(self):
        if self.webcam:
            self.webcam.release()  # Release the webcam
            self.webcam = None

    def update(self):
        self.game.screen.fill((0, 0, 0))
        level, steps = self.levels[self.current_level]

        # Check if we're in a word transition delay
        if self.word_transition_time and time.time() - self.word_transition_time < 5:
            prev_level_index = self.current_level - 1 if self.current_level > 0 else len(self.levels) - 1
            prev_level, prev_steps = self.levels[prev_level_index]
            img_key = f"{prev_level}_{len(prev_steps)}"
            self.game.screen.blit(self.images[img_key], (0, 0))

            # Continue drawing confetti during transition
            if self.celebration_active:
                for particle in self.confetti_particles[:]:
                    particle.fall()
                    particle.draw(self.game.screen)
                    if particle.y > 600 or particle.x < 0 or particle.x > 1024:
                        self.confetti_particles.remove(particle)

                if self.celebration_start_time and time.time() - self.celebration_start_time > 5:
                    self.celebration_active = False
                    self.celebration_start_time = None
                    self.confetti_particles = []

            return

        img_key = f"{level}_{self.current_step}"
        self.game.screen.blit(self.images[img_key], (0, 0))

        if self.webcam:
            ret, webcam_frame = self.webcam.read()
            if ret:
                webcam_frame = cv2.flip(webcam_frame, 1)
                h, w, _ = webcam_frame.shape
                roi = webcam_frame[:, w//2:]
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                result = self.hands.process(roi_rgb)

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y])
                        input_data = np.array(landmarks, dtype=np.float32).reshape(1, -1)
                        if input_data.shape[1] == self.input_details[0]['shape'][1]:
                            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                            self.interpreter.invoke()
                            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                            prediction = np.argmax(output_data)
                            if self.labels[prediction] == steps[self.current_step]:
                                self.correct = True
                                if self.start_time is None:
                                    self.start_time = time.time()
                            else:
                                self.correct = False
                        self.mp_drawing.draw_landmarks(
                            roi, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                        )
                else:
                    # No hand landmarks detected, do not display "Try Again"
                    self.correct = None

                # Display result
                if self.correct is True:
                    result_text = "Correct"
                    if self.start_time and time.time() - self.start_time > 1:
                        self.current_step += 1
                        self.correct = False
                        self.start_time = None
                        if self.current_step >= len(steps):
                            self.current_step = len(steps)
                            self.celebrate()
                            self.save_progress(level)
                            self.word_transition_time = time.time()
                            self.current_level += 1
                            if self.current_level >= len(self.levels):
                                self.current_level = 0
                            self.current_step = 0
                elif self.correct is False:
                    result_text = "Try Again"
                    self.start_time = None
                else:
                    result_text = ""

                result_surface = self.game.font.render(result_text, True, pygame.Color('white'))
                result_x = self.webcam_position[0] + (self.webcam_size[0] - result_surface.get_width()) // 2
                result_y = self.webcam_position[1] + self.webcam_size[1] + 8
                self.game.screen.blit(result_surface, (result_x, result_y))
                webcam_surface = pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB), self.webcam_size).swapaxes(0, 1))
                self.game.screen.blit(webcam_surface, self.webcam_position)

        self.game.screen.blit(self.back_button_img, self.back_button_rect.topleft)
        if self.hovered_button == self.back_button_collision:
            pygame.draw.rect(self.game.screen, (0, 255, 0), self.back_button_collision, 3)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            if self.back_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.back_button_collision:
                    pygame.mixer.Sound(resource_path("AUDIO/CURSOR ON TOP.mp3")).play()
                self.hovered_button = self.back_button_collision
            else:
                self.hovered_button = None
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound(resource_path("AUDIO/MOUSE CLICK.mp3")).play()
                self.game.change_state("playing_home")

    def celebrate(self):
        # Create confetti particles
        self.confetti_particles = [Confetti(1024, 600) for _ in range(150)]  # More particles for word completion
        self.celebration_active = True
        self.celebration_start_time = time.time()
        
        # Load confetti sound effect
        self.confetti_sound = pygame.mixer.Sound(resource_path("AUDIO/CELEB.mp3"))
        self.confetti_sound.play()  # This line plays the sound

    def save_progress(self, word):
        """Save the progress of the current profile"""
        if self.game.current_profile:
            save_path = f"saves/{self.game.current_profile}.json"
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    save_data = json.load(f)
                
                # Ensure completed_lessons is a dictionary
                if not isinstance(save_data["progress"].get("completed_lessons"), dict):
                    save_data["progress"]["completed_lessons"] = {}
                
                if "star_quest" not in save_data["progress"]["completed_lessons"]:
                    save_data["progress"]["completed_lessons"]["star_quest"] = []

                progress_entry = f"Fingerspelling: {word}"
                if progress_entry not in save_data["progress"]["completed_lessons"]["star_quest"]:
                    save_data["progress"]["completed_lessons"]["star_quest"].append(progress_entry)
                    save_data["progress"]["completed_lessons"]["star_quest"].sort()  # Sort the entries
                
                with open(save_path, "w") as f:
                    json.dump(save_data, f, indent=4)
                
                print(f"Progress saved for word: {word} in Star Quest")

# Function to check internet connectivity
def is_connected():
    try:
        # Check connectivity by connecting to Google's DNS
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        return True
    except OSError:
        return False

# Function to send email
def send_email(profile_name, completed_lessons):
    sender_email = "senyasapp@gmail.com"  # Replace with your Gmail
    sender_password = "pvxf ydsn rzxl xbby"  # Replace with your Gmail password

    # Create email content
    subject = "Completed Lessons"
    body = "Here are your completed lessons:\n\n"
    for lesson_category, lessons in completed_lessons.items():
        body += f"{lesson_category.upper()}:\n"
        for lesson in lessons:
            body += f"{lesson}\n"
        body += "\n"

    # Create email message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = profile_name
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Send email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, profile_name, msg.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Function to show the loading screen
def show_loading_screen(screen, progress):
    screen.fill((0, 0, 0))  # Black background
    
    # Load Nunito font
    # You'll need to install the font file in your project directory
    try:
        developers_font = pygame.font.Font(resource_path("fonts/Nunito-Regular.ttf"), 18)  # Smaller font for "DEVELOPERS:"
        names_font = pygame.font.Font(resource_path("fonts/Nunito-Bold.ttf"), 24)  # Larger font for names
    except:
        # Fallback if font file isn't found
        developers_font = pygame.font.Font(None, 24)
        names_font = pygame.font.Font(None, 36)
    
    # Define color palette for developer names
    colors = {
        "STEPHEN GABRIEL S. ALOJADO": (0, 255, 255),     # Cyan
        "SOFIA BIANCA J. CARDENAS": (0, 255, 255),     # Cyan
        "ANGELO LOUIS D. MALABANAN": (0, 255, 255),     # Cyan
        "JOHN REI R. MALATA": (0, 255, 255),     # Cyan
        "ARIANE MAE D. UMALI": (0, 255, 255)     # Cyan
    }
    
    # Get screen dimensions
    screen_height = screen.get_height()
    
    # Position for loading bar
    bar_y = screen_height - 100  # Position near bottom
    
    # Calculate the total content height (developers text + 5 names with spacing)
    names_height = 50 * 5  # 5 names with 50px spacing each
    developers_text = developers_font.render("DEVELOPERS:", True, (255, 255, 255))
    developers_height = developers_text.get_height() + 30  # text height + spacing to first name
    total_content_height = developers_height + names_height
    
    # Calculate starting position to center everything in the available space
    available_space = bar_y - 20  # Space from top to loading bar minus some margin
    developers_pos_y = (available_space - total_content_height) // 2 + 20  # Add a small margin from the top
    
    # Render "DEVELOPERS:" text
    screen.blit(developers_text, (screen.get_width() // 2 - developers_text.get_width() // 2, developers_pos_y))
    
    # Render developer names with specified colors
    names = [
        "STEPHEN GABRIEL S. ALOJADO",
        "SOFIA BIANCA J. CARDENAS",
        "ANGELO LOUIS D. MALABANAN",
        "JOHN REI R. MALATA",
        "ARIANE MAE D. UMALI"
    ]
    
    # Calculate vertical positioning to center the names group
    start_y = developers_pos_y + developers_text.get_height() + 30  # Start below "DEVELOPERS:" with some spacing
    
    # Render each name with its color
    for name in names:
        name_text = names_font.render(name, True, colors[name])
        screen.blit(name_text, (screen.get_width() // 2 - name_text.get_width() // 2, start_y))
        start_y += 50  # Add spacing between names
    
    # Draw the loading bar at the bottom
    bar_width = 400
    bar_height = 20
    bar_x = (screen.get_width() - bar_width) // 2
    
    # Draw background bar (dark gray)
    pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
    
    # Draw progress bar (green)
    pygame.draw.rect(screen, (0, 255, 0), (bar_x, bar_y, int(bar_width * (progress / 100)), bar_height))
    
    pygame.display.flip()
    pygame.event.pump()  # Prevent freezing

# Function to load assets
def load_assets(screen):
    total_assets = len(os.listdir(resource_path("BUTTONS"))) + len(os.listdir(resource_path("AUDIO"))) + 15  # Approximate total
    loaded_assets = 0
    
    images = {}
    for file in os.listdir(resource_path("BUTTONS")):
        if file.endswith(".png"):
            images[file] = pygame.image.load(resource_path(f"BUTTONS/{file}")).convert_alpha()
            loaded_assets += 1
            show_loading_screen(screen, loaded_assets * 100 // total_assets)
            pygame.time.delay(50)
    
    sounds = {}
    for file in os.listdir(resource_path("AUDIO")):
        if file.endswith(".mp3"):
            sounds[file] = pygame.mixer.Sound(resource_path(f"AUDIO/{file}"))
            loaded_assets += 1
            show_loading_screen(screen, loaded_assets * 100 // total_assets)
            pygame.time.delay(50)
    
    videos = {}
    video_keys = ["welcome", "intro", "usertype", "llanding", "glanding", "lgsign", "ggsign", "lplanet", "gplanet", "home", "blgsign", "gexplorer", "galpha", "gnum", "gphrases"]
    for key in video_keys:
        videos[key] = cv2.VideoCapture(resource_path(f"SCENES/{key.upper()}.mp4"))
        loaded_assets += 1
        show_loading_screen(screen, loaded_assets * 100 // total_assets)
        pygame.time.delay(50)
    
    return images, sounds, videos

import os, sys, traceback, json, pygame

# ------------------------------------------------------------------
def resource_path(rel):
    """Return absolute path to resource (works for PyInstaller)."""
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, rel)

# ------------------------------------------------------------------
class Game:
    def __init__(self):
        try:
            # ---------- Pygame / window ----------
            pygame.init()
            pygame.mixer.init()                       # may fail if no audio

            self.screen = pygame.display.set_mode((1024, 600))
            pygame.display.set_caption("SENYAS")

            # ---------- runtime fields ------------
            self.current_profile = None
            self.current_state_name = "welcome"
            self.current_state_data = None
            self.font = pygame.font.Font(None, 36)

            # ---------- load assets ---------------
            show_loading_screen(self.screen, 0)
            self.images, self.sounds, self.videos = load_assets(self.screen)

            # ---------- base states --------------
            self.states: dict[str, object] = {
                "welcome": WelcomeState(
                    self,
                    "welcome",
                    [(resource_path("BUTTONS/LAUNCH.png"), None, "playing_welcome")],
                    resource_path("AUDIO/LAUNCH SOUND.mp3")
                ),
                "playing_welcome":  VideoState(self, "welcome",  "playing_intro"),
                "playing_intro":    VideoState(self, "intro",    "playing_usertype",
                                               resource_path("AUDIO/INTRO.mp3")),
                "playing_usertype": UserTypeState(
                    self,
                    "usertype",
                    [
                        (resource_path("BUTTONS/LEARNER.png"),  None, "playing_learner_planet"),
                        (resource_path("BUTTONS/GUARDIAN.png"), None, "playing_guardian_planet")
                    ],
                    resource_path("AUDIO/USERTYPE.mp3")
                ),
                "load_game": LoadGameState(self, "blgsign"),
                "playing_learner_planet":  VideoState(self, "lplanet",  "playing_learner_landing"),
                "playing_learner_landing": VideoState(self, "llanding", "playing_blgsign",
                                                      resource_path("AUDIO/LLANDING.mp3")),
                "playing_blgsign": BLGSignState(
                    self,
                    "blgsign",
                    [
                        (resource_path("BUTTONS/NEW GAME.png"),  None, "playing_lgsign"),
                        (resource_path("BUTTONS/LOAD GAME.png"), None, "load_game")
                    ]
                ),
                "playing_lgsign":   VideoWithSignInState(self, "lgsign", "playing_home"),
                "playing_guardian_planet":  VideoState(self, "gplanet",  "playing_guardian_landing"),
                "playing_guardian_landing": VideoState(self, "glanding", "playing_home",
                                                       resource_path("AUDIO/GLANDING.mp3")),
                "playing_home":     HomeState(self, "home", None, resource_path("AUDIO/HOME.mp3")),
                "playing_galaxy":   GalaxyExplorerState(self, "gexplorer"),
                "playing_alphabets":GalaxyExplorerAlphabetState(self, "galpha"),
                "playing_numbers":  GalaxyExplorerNumberState(self, "gnum"),
                "playing_phrases":  GalaxyExplorerPhrasesstate(self, "gphrases"),
                "on_screen_keyboard": OnScreenKeyboardState(self),
                "playing_star":       StarQuestState(self)
            }

            # ---------- dynamic phrases ----------
            self.phrase_sequence = []
            for phrase in ["HELLO", "THANKYOU", "ILOVEYOU", "SORRY"]:
                s = f"playing_{phrase.lower()}"
                img = resource_path(os.path.join("GAME PROPER", "GEXPLORER PHRASES", f"{phrase}.png"))
                self.states[s] = PhraseDisplayState(self, img, phrase, None)
                self.phrase_sequence.append(s)

            # ---------- dynamic numbers ----------
            self.number_sequence = []
            for i in range(10):
                s = f"playing_{i}"
                img = resource_path(os.path.join("GAME PROPER", "GEXPLORER NUMBER", f"{i}.png"))
                self.states[s] = NumberDisplayState(self, img, expected_number=i)
                self.number_sequence.append(s)

            # ---------- dynamic alphabets --------
            self.alphabet_sequence = []
            for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                s = f"playing_{letter.lower()}"
                img = resource_path(os.path.join("GAME PROPER", "GEXPLORER ALPHABET", f"{letter}.png"))
                self.states[s] = AlphabetDisplayState(self, img, expected_letter=letter)
                self.alphabet_sequence.append(s)

            # ---------- cosmic copy --------------
            self.states["playing_cosmic"] = CosmicCopyState(self)

            # ---------- start game ---------------
            self.current_state = self.states["welcome"]
            self.current_state.enter()
            self.clock = pygame.time.Clock()

        except Exception:
            # full traceback to console & file
            traceback.print_exc()
            with open("init_error.log", "w") as f:
                traceback.print_exc(file=f)
            print("[CRITICAL] Game initialization failed.", flush=True)
            pygame.quit()
            sys.exit(1)

    # ------------------------------------------
    def change_state(self, new_state, data=None):
        self.current_state.exit()
        self.current_state = self.states[new_state]
        self.current_state_name = new_state
        self.current_state_data = data
        self.current_state.enter()

    # ------------------------------------------
    def run(self):
        while True:
            self.screen.fill((0, 0, 0))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                self.current_state.handle_event(event)

            self.current_state.update()
            self.current_state.render()
            pygame.display.flip()
            self.clock.tick(30)

# --------------------------------------------------
if __name__ == "__main__":
    Game().run()