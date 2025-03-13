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
        self.next_button_img = pygame.image.load("BUTTONS/NEXT.png").convert_alpha()
        self.next_button_rect = self.next_button_img.get_rect(topleft=(462, 370))
        self.next_button_collision = get_collision_rect(self.next_button_img)
        
        # Adjust the height of the collision rectangle
        self.next_button_collision.height = 75
        self.next_button_collision.width = 230
        self.next_button_collision.x = 398
        self.next_button_collision.y = 385 
        
        # Add back button
        self.back_button_img = pygame.image.load("BUTTONS/BACK.png").convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect(topleft=(10, 10))
        self.back_button_collision = get_collision_rect(self.back_button_img)
        
        # Adjust the height, width, x, and y of the collision rectangle for back button
        self.back_button_collision.height = 85
        self.back_button_collision.width = 90
        self.back_button_collision.x = 28
        self.back_button_collision.y = 35
        
        self.text = ''
        self.active = False
        self.last_frame = None
        self.hovered_button = None


    def enter(self):
        super().enter()
        self.text = ''
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
                    pygame.mixer.Sound("AUDIO/CURSOR ON TOP.mp3").play()
                self.hovered_button = self.next_button_collision
            # Check back button hover
            elif self.back_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.back_button_collision:
                    pygame.mixer.Sound("AUDIO/CURSOR ON TOP.mp3").play()
                self.hovered_button = self.back_button_collision
            else:
                self.hovered_button = None
                
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.input_box_rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
                
            # Handle next button click
            if self.next_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
                print(f"Entered text: {self.text}")
                # Save the profile name and create a save file
                if self.text.strip():  # Only save if text isn't empty
                    self.save_profile(self.text)
                # Go to home screen
                self.game.change_state("playing_home")
                
            # Handle back button click
            elif self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
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
        
        # Create save directory if it doesn't exist
        if not os.path.exists("saves"):
            os.makedirs("saves")
        
        # Create a new save file with initial data
        save_data = {
            "name": profile_name,
            "created_at": pygame.time.get_ticks(),
            "progress": {
                "level": 1,
                "score": 0,
                "completed_lessons": []
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
            img = pygame.image.load(img_path).convert_alpha()
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
                pygame.mixer.Sound("AUDIO/CURSOR ON TOP.mp3").play()
            
            self.hovered_button = hovered

        if event.type == pygame.MOUSEBUTTONDOWN and self.buttons_active:
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
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
                pygame.mixer.Sound("AUDIO/CURSOR ON TOP.mp3").play()

            self.hovered_button = hovered

        if event.type == pygame.MOUSEBUTTONDOWN and self.video_finished and self.buttons_active:
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
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
        self.back_button_img = pygame.image.load("BUTTONS/BACK.png").convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect(topleft=(10, 10))
        self.back_button_collision = get_collision_rect(self.back_button_img)


        # Adjust the height, width, x, and y of the collision rectangle
        self.back_button_collision.height = 85
        self.back_button_collision.width = 90
        self.back_button_collision.x = 28
        self.back_button_collision.y = 35 
        
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
                pygame.mixer.Sound("AUDIO/CURSOR ON TOP.mp3").play()

            self.hovered_button = hovered

            # Check if the back button is hovered
            if self.back_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.back_button_collision:
                    pygame.mixer.Sound("AUDIO/CURSOR ON TOP.mp3").play()
                self.hovered_button = self.back_button_collision

        if event.type == pygame.MOUSEBUTTONDOWN and self.video_finished and self.buttons_active:
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
                    self.game.change_state(state)
                    break

            # Check if the back button is clicked
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
                self.game.change_state("playing_usertype")

# Load State
class LoadGameState(State):
    def __init__(self, game, video_key, audio_file=None):
        super().__init__(game)
        self.video_key = video_key
        self.audio_file = audio_file
        self.sound = None if audio_file is None else pygame.mixer.Sound(audio_file)
        self.last_frame = None
        self.font = pygame.font.Font(None, 36)
        self.profile_buttons = []
        
        # Load back button image
        self.back_button_img = pygame.image.load("BUTTONS/BACK.png").convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect(topleft=(10, 10))
        self.back_button_collision = get_collision_rect(self.back_button_img)

        
        # Adjust the back button collision area
        self.back_button_collision.height = 85
        self.back_button_collision.width = 90
        self.back_button_collision.x = 28
        self.back_button_collision.y = 35
        
        self.hovered_button = None
        
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
        
        # Check if saves directory exists
        if not os.path.exists("saves"):
            return
        
        # Get all JSON files in the saves directory
        save_files = [f for f in os.listdir("saves") if f.endswith(".json")]
        
        # Create a button for each save file
        y_position = 150
        for i, file in enumerate(save_files):
            profile_name = file[:-5]  # Remove the .json extension
            
            # Create a rectangle for the profile button
            button_rect = pygame.Rect(362, y_position + i * 60, 300, 50)
            self.profile_buttons.append((profile_name, button_rect))
    
    def update(self):
        # If we don't have a frame yet, try to get one
        if self.last_frame is None:
            ret, frame = self.game.videos[self.video_key].read()
            if ret:
                self.last_frame = pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (1024, 600)).swapaxes(0, 1))
        
        # No need to call render here, it will be called by the game loop
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            # Check profile buttons
            self.hovered_button = None
            for _, button_rect in self.profile_buttons:
                if button_rect.collidepoint(event.pos):
                    if self.hovered_button != button_rect:
                        pygame.mixer.Sound("AUDIO/CURSOR ON TOP.mp3").play()
                    self.hovered_button = button_rect
                    break
            
            # Check back button
            if self.back_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.back_button_collision:
                    pygame.mixer.Sound("AUDIO/CURSOR ON TOP.mp3").play()
                self.hovered_button = self.back_button_collision
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check profile buttons
            for profile_name, button_rect in self.profile_buttons:
                if button_rect.collidepoint(event.pos):
                    pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
                    self.game.current_profile = profile_name
                    print(f"Loaded profile: {profile_name}")
                    self.game.change_state("playing_home")
                    return
            
            # Check back button
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
                self.game.change_state("playing_blgsign")
    
    def render(self):
        # Fill the screen with a dark background color in case we don't have a frame
        self.game.screen.fill((20, 20, 40))
        
        if self.last_frame:
            self.game.screen.blit(self.last_frame, (0, 0))
        
        # Create a semi-transparent overlay for better text readability
        overlay = pygame.Surface((1024, 600), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))  # Black with 50% transparency
        self.game.screen.blit(overlay, (0, 0))
        
        # Draw a title for the load game screen
        title_text = self.font.render("SELECT A PROFILE", True, pygame.Color('white'))
        title_rect = title_text.get_rect(center=(512, 100))
        self.game.screen.blit(title_text, title_rect)
        
        # Draw profile buttons
        for profile_name, button_rect in self.profile_buttons:
            # Draw button background
            pygame.draw.rect(self.game.screen, (50, 50, 100), button_rect)
            
            # Draw button border
            border_color = (0, 255, 0) if button_rect == self.hovered_button else (255, 255, 255)
            pygame.draw.rect(self.game.screen, border_color, button_rect, 2)
            
            # Draw profile name
            name_text = self.font.render(profile_name, True, pygame.Color('white'))
            text_rect = name_text.get_rect(center=button_rect.center)
            self.game.screen.blit(name_text, text_rect)
        
        # Draw back button
        self.game.screen.blit(self.back_button_img, self.back_button_rect.topleft)
        if self.hovered_button == self.back_button_collision:
            pygame.draw.rect(self.game.screen, (0, 255, 0), self.back_button_collision, 3)
        
        # If no profiles found, display a message
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
        self.back_button_img = pygame.image.load("BUTTONS/BACK.png").convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect(topleft=(10, 10))
        self.back_button_collision = get_collision_rect(self.back_button_img)

        
        # Adjust the back button collision area
        self.back_button_collision.height = 85
        self.back_button_collision.width = 90
        self.back_button_collision.x = 28
        self.back_button_collision.y = 35

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
                pygame.mixer.Sound("AUDIO/CURSOR ON TOP.mp3").play()
            
            self.hovered_button = hovered
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
                    # Actually change the state now
                    print(f"Button clicked: {state}")
                    self.game.change_state(state)
                    return
                
            # Check back button
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
                self.game.change_state("playing_usertype")

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
        self.back_button_img = pygame.image.load("BUTTONS/BACK.png").convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect(topleft=(10, 10))
        self.back_button_collision = get_collision_rect(self.back_button_img)

        
        # Adjust the back button collision area
        self.back_button_collision.height = 85
        self.back_button_collision.width = 90
        self.back_button_collision.x = 28
        self.back_button_collision.y = 35
        
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
                pygame.mixer.Sound("AUDIO/CURSOR ON TOP.mp3").play()
            
            self.hovered_button = hovered
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check category buttons
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
                    # Transition to the appropriate state
                    print(f"Button clicked: {state}")
                    self.game.change_state(state)
                    return
            
            # Check back button
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
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
        self.back_button_img = pygame.image.load("BUTTONS/BACK.png").convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect(topleft=(10, 10))
        self.back_button_collision = get_collision_rect(self.back_button_img)
        
        # Adjust the back button collision area
        self.back_button_collision.height = 85
        self.back_button_collision.width = 90
        self.back_button_collision.x = 28
        self.back_button_collision.y = 35
        
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
                pygame.mixer.Sound("AUDIO/CURSOR ON TOP.mp3").play()
            
            self.hovered_button = hovered
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check category buttons
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
                    # Transition to the appropriate state
                    print(f"Button clicked: {state}")
                    self.game.change_state(state)
                    return
            
            # Check back button
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
                self.game.change_state("playing_galaxy")

class AlphabetDisplayState(State):
    def __init__(self, game, image_path, expected_letter, webcam_position=(700, 150), webcam_size=(300, 225)):
        super().__init__(game)
        self.original_image = pygame.image.load(image_path).convert_alpha()

        # Scale the image to fit within 1024x600 while maintaining aspect ratio
        self.image = pygame.transform.smoothscale(self.original_image, self.get_scaled_dimensions(self.original_image, 1024, 600))
        self.image_rect = self.image.get_rect(center=(1024 // 2, 600 // 2))  # Center the image

        # Back button (Initialize once)
        self.back_button_img = pygame.image.load("BUTTONS/BACK.png").convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect(topleft=(10, 10))
        self.back_button_collision = get_collision_rect(self.back_button_img)

        # Adjust the back button collision area
        self.back_button_collision.height = 85
        self.back_button_collision.width = 90
        self.back_button_collision.x = 28
        self.back_button_collision.y = 35

        # Next button
        self.next_button_img = pygame.image.load("BUTTONS/NXT.png").convert_alpha()
        self.next_button_rect = self.next_button_img.get_rect(topright=(1014, 10))  # Position at top right
        self.next_button_collision = get_collision_rect(self.next_button_img)
        self.next_button_collision.height = 76
        self.next_button_collision.width = 255
        self.next_button_collision.x = 733  # Adjusted x position for collision
        self.next_button_collision.y = 41

        self.last_frame = None
        self.hovered_button = None

        # Webcam feed parameters
        self.webcam_position = (600, 152)
        self.webcam_size = (350, 263)
        self.webcam = None  # Initialize webcam as None

        # Load the TFLite model
        self.interpreter = tflite.Interpreter(model_path="MODEL/asl_mlp_model.tflite")
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

    def get_scaled_dimensions(self, image, max_width, max_height):
        """Returns new dimensions for the image while maintaining aspect ratio"""
        img_width, img_height = image.get_size()
        scale_factor = min(max_width / img_width, max_height / img_height)
        return int(img_width * scale_factor), int(img_height * scale_factor)

    def enter(self):
        self.webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Initialize webcam
        self.correct = False
        self.start_time = None

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
                
                if result.multi_hand_landmarks:
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
                            
                            # Check if the prediction is correct
                            if self.labels[prediction] == self.expected_letter:
                                self.correct = True
                                if self.start_time is None:
                                    self.start_time = time.time()  # Start the timer
                            else:
                                if self.start_time is None:
                                    self.start_time = time.time()
                                elif time.time() - self.start_time > 5:
                                    self.correct = False
                                    self.start_time = None

                            # Draw landmarks
                            self.mp_drawing.draw_landmarks(
                                roi, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                            )
                
                # Display result
                if self.correct:
                    result_text = "Correct"
                else:
                    result_text = "Try Again" if self.start_time else ""
                    self.start_time = None  # Reset the timer if not correct

                result_surface = self.game.font.render(result_text, True, pygame.Color('white'))

                # Calculate the x-coordinate to center the text below the webcam
                result_x = self.webcam_position[0] + (self.webcam_size[0] - result_surface.get_width()) // 2
                result_y = self.webcam_position[1] + self.webcam_size[1] + 8

                self.game.screen.blit(result_surface, (result_x, result_y))

                # Convert the webcam frame to a surface and blit it
                webcam_surface = pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB), self.webcam_size).swapaxes(0, 1))
                self.game.screen.blit(webcam_surface, self.webcam_position)

        # Draw back button
        self.game.screen.blit(self.back_button_img, self.back_button_rect.topleft)

        # Draw next button
        self.game.screen.blit(self.next_button_img, self.next_button_rect.topleft)

        # Draw hover effect
        if self.hovered_button == self.back_button_collision:
            pygame.draw.rect(self.game.screen, (0, 255, 0), self.back_button_collision, 3)
        elif self.hovered_button == self.next_button_collision:
            pygame.draw.rect(self.game.screen, (0, 255, 0), self.next_button_collision, 3)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            if self.back_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.back_button_collision:
                    pygame.mixer.Sound("AUDIO/CURSOR ON TOP.mp3").play()
                self.hovered_button = self.back_button_collision
            elif self.next_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.next_button_collision:
                    pygame.mixer.Sound("AUDIO/CURSOR ON TOP.mp3").play()
                self.hovered_button = self.next_button_collision
            else:
                self.hovered_button = None

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
                self.game.change_state("playing_alphabets")  # Return to alphabet selection
            elif self.next_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
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

class GalaxyExplorerNumberState(VideoState):
    def __init__(self, game, video_key, next_state=None, audio_file=None):
        super().__init__(game, video_key, next_state, audio_file)
        
        gnum_path = os.path.join(os.getcwd(), "BUTTONS/Numbers")  # Path for GNUM buttons
        self.buttons = []

        button_data = [
            ("GNUM0.png", "playing_0"), ("GNUM1.png", "playing_1"), ("GNUM2.png", "playing_2"),
            ("GNUM3.png", "playing_3"), ("GNUM4.png", "playing_4"), ("GNUM5.png", "playing_5"),
            ("GNUM6.png", "playing_6"), ("GNUM7.png", "playing_7"), ("GNUM8.png", "playing_8"),
            ("GNUM9.png", "playing_9")
        ]

        for gnum, state in button_data:
            btn_surface = pygame.image.load(os.path.join(gnum_path, gnum)).convert_alpha()
            btn_rect = btn_surface.get_rect()
            collision_rect = get_collision_rect(btn_surface)
            self.buttons.append((btn_surface, btn_rect, collision_rect, state))  # Store state

        # Add back button
        self.back_button_img = pygame.image.load("BUTTONS/BACK.png").convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect(topleft=(10, 10))
        self.back_button_collision = get_collision_rect(self.back_button_img)
        
        # Adjust the back button collision area
        self.back_button_collision.height = 85
        self.back_button_collision.width = 90
        self.back_button_collision.x = 28
        self.back_button_collision.y = 35
        
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
                pygame.mixer.Sound("AUDIO/CURSOR ON TOP.mp3").play()

            self.hovered_button = hovered

        if event.type == pygame.MOUSEBUTTONDOWN:
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
                    self.game.change_state(state)
                    return

            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
                self.game.change_state("playing_galaxy")

class NumberDisplayState(State):
    def __init__(self, game, image_path, expected_number, webcam_position=(700, 150), webcam_size=(300, 225)):
        super().__init__(game)
        self.original_image = pygame.image.load(image_path).convert_alpha()

        # Scale the image to fit within 1024x600 while maintaining aspect ratio
        self.image = pygame.transform.smoothscale(self.original_image, self.get_scaled_dimensions(self.original_image, 1024, 600))
        self.image_rect = self.image.get_rect(center=(1024 // 2, 600 // 2))  # Center the image

        # Back button (Initialize once)
        self.back_button_img = pygame.image.load("BUTTONS/BACK.png").convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect(topleft=(10, 10))
        self.back_button_collision = get_collision_rect(self.back_button_img)

        # Adjust the back button collision area
        self.back_button_collision.height = 85
        self.back_button_collision.width = 90
        self.back_button_collision.x = 28
        self.back_button_collision.y = 35

        # Next button
        self.next_button_img = pygame.image.load("BUTTONS/NXT.png").convert_alpha()
        self.next_button_rect = self.next_button_img.get_rect(topright=(1014, 10))  # Position at top right
        self.next_button_collision = get_collision_rect(self.next_button_img)
        self.next_button_collision.height = 76
        self.next_button_collision.width = 255
        self.next_button_collision.x = 733  # Adjusted x position for collision
        self.next_button_collision.y = 41

        self.last_frame = None
        self.hovered_button = None

        # Webcam feed parameters
        self.webcam_position = (600, 152)
        self.webcam_size = (350, 263)
        self.webcam = None  # Initialize webcam as None

        # Load the TFLite model
        self.interpreter = tflite.Interpreter(model_path="MODEL/asl_number_classifier.tflite")
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
                
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        
                        # Convert landmarks to NumPy array and reshape
                        input_data = np.array(landmarks, dtype=np.float32).reshape(1, -1)
                        
                        # Perform inference
                        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                        self.interpreter.invoke()
                        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                        prediction = np.argmax(output_data)
                        
                        # Check if the prediction is correct
                        if self.labels[prediction] == str(self.expected_number):
                            self.correct = True
                            if self.start_time is None:
                                self.start_time = time.time()  # Start the timer
                        else:
                            if self.start_time is None:
                                self.start_time = time.time()
                            elif time.time() - self.start_time > 5:
                                self.correct = False
                                self.start_time = None

                        # Draw landmarks
                        self.mp_drawing.draw_landmarks(
                            roi, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                        )
                
                # Display result
                if self.correct:
                    result_text = "Correct"
                else:
                    result_text = "Try Again" if self.start_time else ""
                    self.start_time = None  # Reset the timer if not correct

                result_surface = self.game.font.render(result_text, True, pygame.Color('white'))

                # Calculate the x-coordinate to center the text below the webcam
                result_x = self.webcam_position[0] + (self.webcam_size[0] - result_surface.get_width()) // 2
                result_y = self.webcam_position[1] + self.webcam_size[1] + 8

                self.game.screen.blit(result_surface, (result_x, result_y))

                # Convert the webcam frame to a surface and blit it
                webcam_surface = pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB), self.webcam_size).swapaxes(0, 1))
                self.game.screen.blit(webcam_surface, self.webcam_position)

        # Draw back button
        self.game.screen.blit(self.back_button_img, self.back_button_rect.topleft)

        # Draw next button
        self.game.screen.blit(self.next_button_img, self.next_button_rect.topleft)

        # Draw hover effect
        if self.hovered_button == self.back_button_collision:
            pygame.draw.rect(self.game.screen, (0, 255, 0), self.back_button_collision, 3)
        elif self.hovered_button == self.next_button_collision:
            pygame.draw.rect(self.game.screen, (0, 255, 0), self.next_button_collision, 3)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            if self.back_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.back_button_collision:
                    pygame.mixer.Sound("AUDIO/CURSOR ON TOP.mp3").play()
                self.hovered_button = self.back_button_collision
            elif self.next_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.next_button_collision:
                    pygame.mixer.Sound("AUDIO/CURSOR ON TOP.mp3").play()
                self.hovered_button = self.next_button_collision
            else:
                self.hovered_button = None

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
                self.game.change_state("playing_numbers")  # Return to number selection
            elif self.next_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
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

class GalaxyExplorerPhrasesstate(VideoState):
    def __init__(self, game, video_key, next_state=None, audio_file=None):
        super().__init__(game, video_key, next_state, audio_file)
        
        # Load button images
        self.buttons = []
        
        button_data = [
            ("BUTTONS/Phrases/GOODBYE.png", "playing_goodbye"), ("BUTTONS/Phrases/HELLO.png", "playing_hello"), ("BUTTONS/Phrases/ILOVEYOU.png", "playing_iloveyou"),
            ("BUTTONS/Phrases/NO.png", "playing_no"), ("BUTTONS/Phrases/PLEASE.png", "playing_please"), ("BUTTONS/Phrases/SORRY.png", "playing_sorry"),
            ("BUTTONS/Phrases/THANKYOU.png", "playing_thankyou"), ("BUTTONS/Phrases/YES.png", "playing_yes")                                              
        ]
        
        for img_path, state in button_data:
            img = pygame.image.load(img_path).convert_alpha()
            img_rect = img.get_rect()
            collision_rect = get_collision_rect(img)
            self.buttons.append((img, img_rect, collision_rect, state))
        
        # Add back button
        self.back_button_img = pygame.image.load("BUTTONS/BACK.png").convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect(topleft=(10, 10))
        self.back_button_collision = get_collision_rect(self.back_button_img)

        
        # Adjust the back button collision area
        self.back_button_collision.height = 85
        self.back_button_collision.width = 90
        self.back_button_collision.x = 28
        self.back_button_collision.y = 35
        
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
                pygame.mixer.Sound("AUDIO/CURSOR ON TOP.mp3").play()
            
            self.hovered_button = hovered
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check category buttons
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
                    # Transition to the appropriate state
                    print(f"Button clicked: {state}")
                    self.game.change_state(state)
                    return
            
            # Check back button
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
                self.game.change_state("playing_galaxy")

class PhraseDisplayState(State):
    def __init__(self, game, image_path, expected_phrase, webcam_position=(700, 150), webcam_size=(300, 225)):
        super().__init__(game)
        self.original_image = pygame.image.load(image_path).convert_alpha()

        # Scale the image to fit within 1024x600 while maintaining aspect ratio
        self.image = pygame.transform.smoothscale(self.original_image, self.get_scaled_dimensions(self.original_image, 1024, 600))
        self.image_rect = self.image.get_rect(center=(1024 // 2, 600 // 2))  # Center the image

        # Back button (Initialize once)
        self.back_button_img = pygame.image.load("BUTTONS/BACK.png").convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect(topleft=(10, 10))
        self.back_button_collision = get_collision_rect(self.back_button_img)

        # Adjust the back button collision area
        self.back_button_collision.height = 85
        self.back_button_collision.width = 90
        self.back_button_collision.x = 28
        self.back_button_collision.y = 35

        # Next button
        self.next_button_img = pygame.image.load("BUTTONS/NXT.png").convert_alpha()
        self.next_button_rect = self.next_button_img.get_rect(topright=(1014, 10))  # Position at top right
        self.next_button_collision = get_collision_rect(self.next_button_img)
        self.next_button_collision.height = 76
        self.next_button_collision.width = 255
        self.next_button_collision.x = 733  # Adjusted x position for collision
        self.next_button_collision.y = 41

        self.last_frame = None
        self.hovered_button = None

        # Webcam feed parameters
        self.webcam_position = (600, 152)
        self.webcam_size = (350, 263)
        self.webcam = None  # Initialize webcam as None

        # Load the TFLite model
        self.interpreter = tflite.Interpreter(model_path="MODEL/asl_phrase_model.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Mediapipe Hands setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # Label map
        self.labels = ["HELLO", "GOODBYE", "YES", "NO", "PLEASE", "SORRY", "THANKYOU", "ILOVEYOU"]

        # Timer for try again message
        self.start_time = None
        self.correct = False

        # Expected phrase for this state
        self.expected_phrase = expected_phrase

    def get_scaled_dimensions(self, image, max_width, max_height):
        """Returns new dimensions for the image while maintaining aspect ratio"""
        img_width, img_height = image.get_size()
        scale_factor = min(max_width / img_width, max_height / img_height)
        return int(img_width * scale_factor), int(img_height * scale_factor)

    def enter(self):
        self.webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Initialize webcam
        self.correct = False
        self.start_time = None

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
                
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])  # Use x, y, and z coordinates
                        
                        # Convert landmarks to NumPy array and reshape
                        input_data = np.array(landmarks, dtype=np.float32).reshape(1, -1)
                        
                        # Perform inference
                        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                        self.interpreter.invoke()
                        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                        prediction = np.argmax(output_data)
                        
                        # Check if the prediction is correct
                        if self.labels[prediction] == self.expected_phrase:
                            self.correct = True
                            if self.start_time is None:
                                self.start_time = time.time()  # Start the timer
                        else:
                            if self.start_time is None:
                                self.start_time = time.time()
                            elif time.time() - self.start_time > 5:
                                self.correct = False
                                self.start_time = None

                        # Draw landmarks
                        self.mp_drawing.draw_landmarks(
                            roi, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                        )
                
                # Display result
                if self.correct:
                    result_text = "Correct"
                else:
                    result_text = "Try Again" if self.start_time else ""
                    self.start_time = None  # Reset the timer if not correct

                result_surface = self.game.font.render(result_text, True, pygame.Color('white'))

                # Calculate the x-coordinate to center the text below the webcam
                result_x = self.webcam_position[0] + (self.webcam_size[0] - result_surface.get_width()) // 2
                result_y = self.webcam_position[1] + self.webcam_size[1] + 8

                self.game.screen.blit(result_surface, (result_x, result_y))

                # Convert the webcam frame to a surface and blit it
                webcam_surface = pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB), self.webcam_size).swapaxes(0, 1))
                self.game.screen.blit(webcam_surface, self.webcam_position)

        # Draw back button
        self.game.screen.blit(self.back_button_img, self.back_button_rect.topleft)

        # Draw next button
        self.game.screen.blit(self.next_button_img, self.next_button_rect.topleft)

        # Draw hover effect
        if self.hovered_button == self.back_button_collision:
            pygame.draw.rect(self.game.screen, (0, 255, 0), self.back_button_collision, 3)
        elif self.hovered_button == self.next_button_collision:
            pygame.draw.rect(self.game.screen, (0, 255, 0), self.next_button_collision, 3)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            if self.back_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.back_button_collision:
                    pygame.mixer.Sound("AUDIO/CURSOR ON TOP.mp3").play()
                self.hovered_button = self.back_button_collision
            elif self.next_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.next_button_collision:
                    pygame.mixer.Sound("AUDIO/CURSOR ON TOP.mp3").play()
                self.hovered_button = self.next_button_collision
            else:
                self.hovered_button = None

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
                self.game.change_state("playing_phrases")  # Return to phrase selection
            elif self.next_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
                # Find the next phrase in sequence
                if self.game.current_state_name in self.game.phrase_sequence:
                    current_index = self.game.phrase_sequence.index(self.game.current_state_name)
                    if current_index < len(self.game.phrase_sequence) - 1:
                        next_state = self.game.phrase_sequence[current_index + 1]
                    else:
                        next_state = "playing_phrases"  # Return to phrase selection if last phrase is reached
                else:
                    next_state = "playing_phrases"  # Default fallback

                self.game.change_state(next_state)

class CosmicCopyState(State):
    def __init__(self, game):
        super().__init__(game)
        self.current_item = None
        self.expected_value = None

        # Load the TFLite models
        self.alphabet_model = tflite.Interpreter(model_path="MODEL/asl_mlp_model.tflite")
        self.alphabet_model.allocate_tensors()
        self.number_model = tflite.Interpreter(model_path="MODEL/asl_number_classifier.tflite")
        self.number_model.allocate_tensors()
        self.phrase_model = tflite.Interpreter(model_path="MODEL/asl_phrase_model.tflite")
        self.phrase_model.allocate_tensors()

        # Mediapipe Hands setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # Load the labels
        self.alphabet_labels = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        self.number_labels = [str(i) for i in range(10)]
        self.phrase_labels = ["goodbye", "hello", "iloveyou", "no", "please", "sorry", "thankyou", "yes"]

        # Back button
        self.back_button_img = pygame.image.load("BUTTONS/BACK.png").convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect(topleft=(10, 10))
        self.back_button_collision = get_collision_rect(self.back_button_img)
        self.back_button_collision.height = 85
        self.back_button_collision.width = 90
        self.back_button_collision.x = 28
        self.back_button_collision.y = 35

        self.hovered_button = None
        self.webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Initialize webcam

        # Timer for try again message
        self.start_time = None
        self.correct = False

        # Webcam feed parameters
        self.webcam_position = (600, 152)
        self.webcam_size = (350, 263)

    def enter(self):
        self.randomize_item()
        self.correct = False
        self.start_time = None

    def randomize_item(self):
        choice = random.choice(["alphabet", "number", "phrase"])
        if choice == "alphabet":
            self.expected_value = random.choice(self.alphabet_labels)
            self.current_item = pygame.image.load(os.path.join("GAME PROPER", "COSMIC COPY ALPHABET", f"{self.expected_value}.png")).convert_alpha()
        elif choice == "number":
            self.expected_value = random.choice(self.number_labels)
            self.current_item = pygame.image.load(os.path.join("GAME PROPER", "COSMIC COPY NUMBER", f"{self.expected_value}.png")).convert_alpha()
        else:
            self.expected_value = random.choice(self.phrase_labels)
            self.current_item = pygame.image.load(os.path.join("GAME PROPER", "COSMIC COPY PHRASES", f"{self.expected_value.upper()}.png")).convert_alpha()

        self.current_item = pygame.transform.smoothscale(self.current_item, self.get_scaled_dimensions(self.current_item, 1024, 600))
        self.current_item_rect = self.current_item.get_rect(center=(1024 // 2, 600 // 2))

    def get_scaled_dimensions(self, image, max_width, max_height):
        """Returns new dimensions for the image while maintaining aspect ratio"""
        img_width, img_height = image.get_size()
        scale_factor = min(max_width / img_width, max_height / img_height)
        return int(img_width * scale_factor), int(img_height * scale_factor)

    def update(self):
        self.game.screen.fill((0, 0, 0))  # Clear screen
        self.game.screen.blit(self.current_item, self.current_item_rect.topleft)  # Centered display

        # Update webcam feed
        ret, webcam_frame = self.webcam.read()
        if ret:
            # Flip the webcam frame horizontally to mirror it
            webcam_frame = cv2.flip(webcam_frame, 1)
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
                            landmarks.extend([lm.x, lm.y, lm.z])  # Use x, y, and z coordinates for numbers and phrases
                    
                    # Convert landmarks to NumPy array and reshape
                    input_data = np.array(landmarks, dtype=np.float32).reshape(1, -1)
                    
                    # Perform inference based on the type of expected value
                    if self.expected_value in self.alphabet_labels:
                        self.interpreter = self.alphabet_model
                    elif self.expected_value in self.number_labels:
                        self.interpreter = self.number_model
                    else:
                        self.interpreter = self.phrase_model

                    self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], input_data)
                    self.interpreter.invoke()
                    output_data = self.interpreter.get_tensor(self.interpreter.get_output_details()[0]['index'])
                    prediction = np.argmax(output_data)
                    
                    # Check if the prediction is correct
                    if self.expected_value in self.alphabet_labels + self.number_labels:
                        if (self.expected_value in self.alphabet_labels and prediction < len(self.alphabet_labels) and self.expected_value == self.alphabet_labels[prediction]) or \
                           (self.expected_value in self.number_labels and prediction < len(self.number_labels) and self.expected_value == self.number_labels[prediction]):
                            self.correct = True
                            if self.start_time is None:
                                self.start_time = time.time()  # Start the timer
                        else:
                            if self.start_time is None:
                                self.start_time = time.time()
                            elif time.time() - self.start_time > 5:
                                self.correct = False
                                self.start_time = None
                    else:
                        if prediction < len(self.phrase_labels) and self.expected_value == self.phrase_labels[prediction]:
                            self.correct = True
                            if self.start_time is None:
                                self.start_time = time.time()  # Start the timer
                        else:
                            if self.start_time is None:
                                self.start_time = time.time()
                            elif time.time() - self.start_time > 5:
                                self.correct = False
                                self.start_time = None

                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        roi, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                    )
            
            # Display result
            if self.correct:
                result_text = "Correct"
                if self.start_time and time.time() - self.start_time > 1.5:  # Check if 1.5 seconds have passed
                    self.randomize_item()  # Proceed to the next item
                    self.correct = False  # Reset correct status
                    self.start_time = None  # Reset start time
            else:
                result_text = "Try Again" if self.start_time else ""
                self.start_time = None  # Reset the timer if not correct

            result_surface = self.game.font.render(result_text, True, pygame.Color('white'))

            # Calculate the x-coordinate to center the text below the webcam
            result_x = self.webcam_position[0] + (self.webcam_size[0] - result_surface.get_width()) // 2
            result_y = self.webcam_position[1] + self.webcam_size[1] + 10  # Adjust the y-coordinate as needed

            self.game.screen.blit(result_surface, (result_x, result_y))

            # Convert the webcam frame to a surface and blit it
            webcam_surface = pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB), self.webcam_size).swapaxes(0, 1))
            self.game.screen.blit(webcam_surface, self.webcam_position)

        # Draw back button
        self.game.screen.blit(self.back_button_img, self.back_button_rect.topleft)

        # Draw hover effect
        if self.hovered_button == self.back_button_collision:
            pygame.draw.rect(self.game.screen, (0, 255, 0), self.back_button_collision, 3)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            if self.back_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.back_button_collision:
                    pygame.mixer.Sound("AUDIO/CURSOR ON TOP.mp3").play()
                self.hovered_button = self.back_button_collision
            else:
                self.hovered_button = None

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("AUDIO/MOUSE CLICK.mp3").play()
                self.correct = False  # Reset correct status
                self.start_time = None  # Reset start time
                self.game.change_state("playing_home")

class LoadingState(State):
    def __init__(self, game):
        super().__init__(game)
        self.font = pygame.font.Font(None, 36)
        self.loading_complete = False
        self.loading_progress = 0  # Progress of the loading bar

    def enter(self):
        # Start loading resources in a separate thread or asynchronously
        self.loading_complete = False
        self.loading_progress = 0
        threading.Thread(target=self.load_resources).start()

    def load_resources(self):
        # Simulate loading resources with a delay
        # Here you can load your TensorFlow Lite model
        self.load_model()
        self.loading_complete = True

    def load_model(self):
        # Simulate loading TensorFlow Lite model
        for i in range(100):
            time.sleep(0.03)  # Simulate time taken to load model
            self.loading_progress = i + 1

    def handle_event(self, event):
        if self.loading_complete:
            self.game.change_state("welcome")

    def update(self):
        pass

    def render(self):
        self.game.screen.fill((0, 0, 0))
        loading_text = self.font.render("Loading...", True, pygame.Color('white'))
        self.game.screen.blit(loading_text, (self.game.screen.get_width() // 2 - loading_text.get_width() // 2,
                                             self.game.screen.get_height() // 2 - loading_text.get_height() // 2 - 50))

        # Draw the loading bar
        bar_width = 400
        bar_height = 30
        bar_x = self.game.screen.get_width() // 2 - bar_width // 2
        bar_y = self.game.screen.get_height() // 2 - bar_height // 2 + 50
        pygame.draw.rect(self.game.screen, pygame.Color('white'), (bar_x, bar_y, bar_width, bar_height), 2)
        pygame.draw.rect(self.game.screen, pygame.Color('green'), (bar_x, bar_y, bar_width * (self.loading_progress / 100), bar_height))


def show_loading_screen(screen, progress):
    screen.fill((0, 0, 0))
    font = pygame.font.Font(None, 50)
    text = font.render("Loading", True, (255, 255, 255))
    screen.blit(text, (screen.get_width() // 2 - text.get_width() // 2, screen.get_height() // 2 - text.get_height() // 2))
    
    # Draw loading bar
    bar_width = 400
    bar_height = 20
    bar_x = (screen.get_width() - bar_width) // 2
    bar_y = screen.get_height() // 2 + 40
    pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))  # Background bar
    pygame.draw.rect(screen, (0, 255, 0), (bar_x, bar_y, int(bar_width * (progress / 100)), bar_height))  # Green progress
    
    pygame.display.flip()
    pygame.event.pump()  # Prevent freezing

def load_assets(screen):
    total_assets = len(os.listdir("BUTTONS")) + len(os.listdir("AUDIO")) + 15  # Approximate total
    loaded_assets = 0
    
    images = {}
    for file in os.listdir("BUTTONS"):
        if file.endswith(".png"):
            images[file] = pygame.image.load(f"BUTTONS/{file}").convert_alpha()
            loaded_assets += 1
            show_loading_screen(screen, loaded_assets * 100 // total_assets)
            pygame.time.delay(50)
    
    sounds = {}
    for file in os.listdir("AUDIO"):
        if file.endswith(".mp3"):
            sounds[file] = pygame.mixer.Sound(f"AUDIO/{file}")
            loaded_assets += 1
            show_loading_screen(screen, loaded_assets * 100 // total_assets)
            pygame.time.delay(50)
    
    videos = {}
    video_keys = ["welcome", "intro", "usertype", "llanding", "glanding", "lgsign", "ggsign", "lplanet", "gplanet", "home", "blgsign", "gexplorer", "galpha", "gnum", "gphrases"]
    for key in video_keys:
        videos[key] = cv2.VideoCapture(f"SCENES/{key.upper()}.mp4")
        loaded_assets += 1
        show_loading_screen(screen, loaded_assets * 100 // total_assets)
        pygame.time.delay(50)
    
    return images, sounds, videos

class Game:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((1024, 600))
        pygame.display.set_caption("SENYAS")
        
        # Initialize the current profile
        self.current_profile = None
        
        self.current_state_name = "welcome"  # Initialize with the starting state

        # Initialize font
        self.font = pygame.font.Font(None, 36)
        
        # Show loading screen while loading assets
        show_loading_screen(self.screen, 0)
        self.images, self.sounds, self.videos = load_assets(self.screen)
        
        self.states = {
            "welcome": WelcomeState(self, "welcome", [("BUTTONS/LAUNCH.png", None, "playing_welcome")], "AUDIO/LAUNCH SOUND.mp3"),
            "playing_welcome": VideoState(self, "welcome", "playing_intro"),
            "playing_intro": VideoState(self, "intro", "playing_usertype", "AUDIO/INTRO.mp3"),  
            "playing_usertype": UserTypeState(self, "usertype", [("BUTTONS/LEARNER.png", None, "playing_learner_planet"), ("BUTTONS/GUARDIAN.png", None, "playing_guardian_planet")], "AUDIO/USERTYPE.mp3"),  
            "load_game": LoadGameState(self, "blgsign"),
            "playing_learner_planet": VideoState(self, "lplanet", "playing_learner_landing"),
            "playing_learner_landing": VideoState(self, "llanding", "playing_blgsign", "AUDIO/LLANDING.mp3"),  
            "playing_blgsign": BLGSignState(self, "blgsign", [("BUTTONS/NEW GAME.png", None, "playing_lgsign"), ("BUTTONS/LOAD GAME.png", None, "load_game")]),
            "playing_lgsign": VideoWithSignInState(self, "lgsign", "playing_home"),
            "playing_guardian_planet": VideoState(self, "gplanet", "playing_guardian_landing"),
            "playing_guardian_landing": VideoState(self, "glanding", "playing_home", "AUDIO/GLANDING.mp3"),  
            "playing_home": HomeState(self, "home", None, "AUDIO/HOME.mp3"),
            "playing_galaxy": GalaxyExplorerState(self, "gexplorer"),
            "playing_alphabets": GalaxyExplorerAlphabetState(self, "galpha"),
            "playing_numbers": GalaxyExplorerNumberState(self, "gnum"),
            "playing_phrases": GalaxyExplorerPhrasesstate(self, "gphrases"),
        }

        # Add phrase states dynamically
        self.phrase_sequence = []

        phrases = ["HELLO", "GOODBYE", "ILOVEYOU", "NO", "PLEASE", "SORRY", "THANKYOU", "YES"]
        for phrase in phrases:
            state_name = f"playing_{phrase.lower()}"
            image_path = os.path.join("GAME PROPER", "GEXPLORER PHRASES", f"{phrase}.png")
            self.states[state_name] = PhraseDisplayState(self, image_path, phrase.upper(), None)
            self.phrase_sequence.append(state_name)  # Store order dynamically

        # Add number states dynamically
        for i in range(10):
            state_name = f"playing_{i}"
            image_path = os.path.join("GAME PROPER", "GEXPLORER NUMBER", f"{i}.png")
            self.states[state_name] = NumberDisplayState(self, image_path, expected_number=i)
        
        # Define number sequence (0 → 1 → 2 → ... → 9)
            self.number_sequence = [f"playing_{i}" for i in range(10)]

            # Define alphabet sequence (A → B → C → ... → Z)
            self.alphabet_sequence = [f"playing_{letter.lower()}" for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]


        # Add alphabet states dynamically
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            state_name = f"playing_{letter.lower()}"
            image_path = os.path.join("GAME PROPER", "GEXPLORER ALPHABET", f"{letter}.png")
            self.states[state_name] = AlphabetDisplayState(self, image_path, expected_letter=letter)

        self.states["playing_cosmic"] = CosmicCopyState(self)
        self.current_state = self.states["welcome"]  # Start at welcome screen after loading
        self.current_state.enter()
        self.clock = pygame.time.Clock()
    
    def change_state(self, new_state):
        self.current_state.exit()
        self.current_state = self.states[new_state]
        self.current_state_name = new_state  # Track the new state
        self.current_state.enter()

    
    def run(self):
        while True:
            self.screen.fill((0, 0, 0))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                self.current_state.handle_event(event)
            self.current_state.update()
            self.current_state.render()
            pygame.display.flip()
            self.clock.tick(30)

if __name__ == "__main__":
    Game().run()
