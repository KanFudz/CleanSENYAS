import pygame
import sys
import cv2
import os
import json


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
        self.next_button_img = pygame.image.load("NEXT.png").convert_alpha()
        self.next_button_rect = self.next_button_img.get_rect(topleft=(462, 370))
        self.next_button_collision = get_collision_rect(self.next_button_img)
        
        # Adjust the height of the collision rectangle
        self.next_button_collision.height = 75
        self.next_button_collision.width = 230
        self.next_button_collision.x = 398
        self.next_button_collision.y = 385 
        
        # Add back button
        self.back_button_img = pygame.image.load("BACK.png").convert_alpha()
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
                    pygame.mixer.Sound("CURSOR ON TOP.mp3").play()
                self.hovered_button = self.next_button_collision
            # Check back button hover
            elif self.back_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.back_button_collision:
                    pygame.mixer.Sound("CURSOR ON TOP.mp3").play()
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
                pygame.mixer.Sound("MOUSE CLICK.mp3").play()
                print(f"Entered text: {self.text}")
                # Save the profile name and create a save file
                if self.text.strip():  # Only save if text isn't empty
                    self.save_profile(self.text)
                # Go to home screen
                self.game.change_state("playing_home")
                
            # Handle back button click
            elif self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("MOUSE CLICK.mp3").play()
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
                pygame.mixer.Sound("CURSOR ON TOP.mp3").play()
            
            self.hovered_button = hovered

        if event.type == pygame.MOUSEBUTTONDOWN and self.buttons_active:
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound("MOUSE CLICK.mp3").play()
                    if self.sound:
                        self.sound.play()  # Play the corresponding audio file
                    self.video_started = True
                    self.buttons_active = False  
                    break

class UserTypeState(State):
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
                pygame.mixer.Sound("CURSOR ON TOP.mp3").play()

            self.hovered_button = hovered

        if event.type == pygame.MOUSEBUTTONDOWN and self.video_finished and self.buttons_active:
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound("MOUSE CLICK.mp3").play()
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
        self.back_button_img = pygame.image.load("BACK.png").convert_alpha()
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
                pygame.mixer.Sound("CURSOR ON TOP.mp3").play()

            self.hovered_button = hovered

            # Check if the back button is hovered
            if self.back_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.back_button_collision:
                    pygame.mixer.Sound("CURSOR ON TOP.mp3").play()
                self.hovered_button = self.back_button_collision

        if event.type == pygame.MOUSEBUTTONDOWN and self.video_finished and self.buttons_active:
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound("MOUSE CLICK.mp3").play()
                    self.game.change_state(state)
                    break

            # Check if the back button is clicked
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("MOUSE CLICK.mp3").play()
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
        self.back_button_img = pygame.image.load("BACK.png").convert_alpha()
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
                        pygame.mixer.Sound("CURSOR ON TOP.mp3").play()
                    self.hovered_button = button_rect
                    break
            
            # Check back button
            if self.back_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.back_button_collision:
                    pygame.mixer.Sound("CURSOR ON TOP.mp3").play()
                self.hovered_button = self.back_button_collision
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check profile buttons
            for profile_name, button_rect in self.profile_buttons:
                if button_rect.collidepoint(event.pos):
                    pygame.mixer.Sound("MOUSE CLICK.mp3").play()
                    self.game.current_profile = profile_name
                    print(f"Loaded profile: {profile_name}")
                    self.game.change_state("playing_home")
                    return
            
            # Check back button
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("MOUSE CLICK.mp3").play()
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
            ("COSMIC BUTTON.png", "playing_cosmic"),
            ("GALAXY BUTTON.png", "playing_galaxy"),
            ("STAR BUTTON.png", "playing_star")
        ]
        
        for img_path, state in button_data:
            img = pygame.image.load(img_path).convert_alpha()
            img_rect = img.get_rect()
            collision_rect = get_collision_rect(img)
            self.buttons.append((img, img_rect, collision_rect, state))
        
        # Add back button
        self.back_button_img = pygame.image.load("BACK.png").convert_alpha()
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
                pygame.mixer.Sound("CURSOR ON TOP.mp3").play()
            
            self.hovered_button = hovered
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound("MOUSE CLICK.mp3").play()
                    # Actually change the state now
                    print(f"Button clicked: {state}")
                    self.game.change_state(state)
                    return
                
            # Check back button
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("MOUSE CLICK.mp3").play()
                self.game.change_state("playing_usertype")

class GalaxyExplorerState(VideoState):
    def __init__(self, game, video_key, next_state=None, audio_file=None):
        super().__init__(game, video_key, next_state, audio_file)
        
        # Load button images
        self.buttons = []
        
        button_data = [
            ("ALPHABETS.png", "playing_alphabets"),
            ("NUMBERS.png", "playing_numbers"),
            ("PHRASES.png", "playing_phrases")
        ]
        
        for img_path, state in button_data:
            img = pygame.image.load(img_path).convert_alpha()
            img_rect = img.get_rect()
            collision_rect = get_collision_rect(img)
            self.buttons.append((img, img_rect, collision_rect, state))
        
        # Add back button
        self.back_button_img = pygame.image.load("BACK.png").convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect(topleft=(10, 10))
        self.back_button_collision = get_collision_rect(self.back_button_img)

        
        # Adjust the back button collision area
        self.back_button_collision.height = 85
        self.back_button_collision.width = 90
        self.back_button_collision.x = 28
        self.back_button_collision.y = 35
        
        self.last_frame = None
        self.hovered_button = None

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
                pygame.mixer.Sound("CURSOR ON TOP.mp3").play()
            
            self.hovered_button = hovered
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check category buttons
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound("MOUSE CLICK.mp3").play()
                    # Transition to the appropriate state
                    print(f"Button clicked: {state}")
                    self.game.change_state(state)
                    return
            
            # Check back button
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("MOUSE CLICK.mp3").play()
                self.game.change_state("playing_home")

class GalaxyExplorerAlphabetState(VideoState):
    def __init__(self, game, video_key, next_state=None, audio_file=None):
        super().__init__(game, video_key, next_state, audio_file)
        
        # Load button images
        self.buttons = []
        
        button_data = [
            ("1_Alphabets/GALPHAA.png", "playing_a"), ("1_Alphabets/GALPHAB.png", "playing_b"), ("1_Alphabets/GALPHAC.png", "playing_c"),
            ("1_Alphabets/GALPHAD.png", "playing_d"), ("1_Alphabets/GALPHAE.png", "playing_e"), ("1_Alphabets/GALPHAF.png", "playing_f"),
            ("1_Alphabets/GALPHAG.png", "playing_g"), ("1_Alphabets/GALPHAH.png", "playing_h"), ("1_Alphabets/GALPHAI.png", "playing_i"),
            ("1_Alphabets/GALPHAJ.png", "playing_j"), ("1_Alphabets/GALPHAK.png", "playing_k"), ("1_Alphabets/GALPHAL.png", "playing_l"),
            ("1_Alphabets/GALPHAM.png", "playing_m"), ("1_Alphabets/GALPHAN.png", "playing_n"), ("1_Alphabets/GALPHAO.png", "playing_o"),
            ("1_Alphabets/GALPHAP.png", "playing_p"), ("1_Alphabets/GALPHAQ.png", "playing_q"), ("1_Alphabets/GALPHAR.png", "playing_r"),
            ("1_Alphabets/GALPHAS.png", "playing_s"), ("1_Alphabets/GALPHAT.png", "playing_t"), ("1_Alphabets/GALPHAU.png", "playing_u"),
            ("1_Alphabets/GALPHAV.png", "playing_v"), ("1_Alphabets/GALPHAW.png", "playing_w"), ("1_Alphabets/GALPHAX.png", "playing_x"),
            ("1_Alphabets/GALPHAY.png", "playing_y"), ("1_Alphabets/GALPHAZ.png", "playing_z")
        ]
        
        for img_path, state in button_data:
            img = pygame.image.load(img_path).convert_alpha()
            img_rect = img.get_rect()
            collision_rect = get_collision_rect(img)
            self.buttons.append((img, img_rect, collision_rect, state))
        
        # Add back button
        self.back_button_img = pygame.image.load("BACK.png").convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect(topleft=(10, 10))
        self.back_button_collision = get_collision_rect(self.back_button_img)
        
        # Adjust the back button collision area
        self.back_button_collision.height = 85
        self.back_button_collision.width = 90
        self.back_button_collision.x = 28
        self.back_button_collision.y = 35
        
        self.last_frame = None
        self.hovered_button = None

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
                pygame.mixer.Sound("CURSOR ON TOP.mp3").play()
            
            self.hovered_button = hovered
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check category buttons
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound("MOUSE CLICK.mp3").play()
                    # Transition to the appropriate state
                    print(f"Button clicked: {state}")
                    self.game.change_state(state)
                    return
            
            # Check back button
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("MOUSE CLICK.mp3").play()
                self.game.change_state("playing_galaxy")

class GalaxyExplorerNumberState(VideoState):
    def __init__(self, game, video_key, next_state=None, audio_file=None):
        super().__init__(game, video_key, next_state, audio_file)
        
        gnum_path = os.path.join(os.getcwd(), "2_Numbers")  # Path for GNUM buttons
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
        self.back_button_img = pygame.image.load("BACK.png").convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect(topleft=(10, 10))
        self.back_button_collision = get_collision_rect(self.back_button_img)
        
        # Adjust the back button collision area
        self.back_button_collision.height = 85
        self.back_button_collision.width = 90
        self.back_button_collision.x = 28
        self.back_button_collision.y = 35
        
        self.last_frame = None
        self.hovered_button = None


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
                pygame.mixer.Sound("CURSOR ON TOP.mp3").play()

            self.hovered_button = hovered

        if event.type == pygame.MOUSEBUTTONDOWN:
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound("MOUSE CLICK.mp3").play()
                    self.game.change_state(state)  # Change to the corresponding number scene
                    return

            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("MOUSE CLICK.mp3").play()
                self.game.change_state("playing_galaxy")

class NumberDisplayState(State):
    def __init__(self, game, image_path, webcam_position=(700, 150), webcam_size=(300, 225)):
        super().__init__(game)
        self.original_image = pygame.image.load(image_path).convert_alpha()

        # Scale the image to fit within 1024x600 while maintaining aspect ratio
        self.image = pygame.transform.smoothscale(self.original_image, self.get_scaled_dimensions(self.original_image, 1024, 600))
        self.image_rect = self.image.get_rect(center=(1024 // 2, 600 // 2))  # Center the image

        # Back button (Initialize once)
        self.back_button_img = pygame.image.load("BACK.png").convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect(topleft=(10, 10))
        self.back_button_collision = get_collision_rect(self.back_button_img)

        # Adjust the back button collision area
        self.back_button_collision.height = 85
        self.back_button_collision.width = 90
        self.back_button_collision.x = 28
        self.back_button_collision.y = 35
        
        self.last_frame = None
        self.hovered_button = None

        # Webcam feed parameters
        self.webcam_position = (600, 152)
        self.webcam_size = (350, 263)
        self.webcam = cv2.VideoCapture(0)  # Initialize webcam

    def get_scaled_dimensions(self, image, max_width, max_height):
        """Returns new dimensions for the image while maintaining aspect ratio"""
        img_width, img_height = image.get_size()
        scale_factor = min(max_width / img_width, max_height / img_height)
        return int(img_width * scale_factor), int(img_height * scale_factor)

    def update(self):
        self.game.screen.fill((0, 0, 0))  # Clear screen
        self.game.screen.blit(self.image, self.image_rect.topleft)  # Centered display

        # Update webcam feed
        ret, webcam_frame = self.webcam.read()
        if ret:
            # Flip the webcam frame horizontally to mirror it
            webcam_frame = cv2.flip(webcam_frame, 1)
            webcam_surface = pygame.surfarray.make_surface(cv2.resize(cv2.cvtColor(webcam_frame, cv2.COLOR_BGR2RGB), self.webcam_size).swapaxes(0, 1))
            self.game.screen.blit(webcam_surface, self.webcam_position)

            # Draw white outline around the webcam feed
            outline_rect = pygame.Rect(self.webcam_position, self.webcam_size)
            pygame.draw.rect(self.game.screen, (255, 255, 255), outline_rect, 5)

        # Draw back button
        self.game.screen.blit(self.back_button_img, self.back_button_rect.topleft)

        # Draw hover effect
        if self.hovered_button == self.back_button_collision:
            pygame.draw.rect(self.game.screen, (0, 255, 0), self.back_button_collision, 3)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            if self.back_button_collision.collidepoint(event.pos):
                if self.hovered_button != self.back_button_collision:
                    pygame.mixer.Sound("CURSOR ON TOP.mp3").play()
                self.hovered_button = self.back_button_collision
            else:
                self.hovered_button = None

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("MOUSE CLICK.mp3").play()
                self.game.change_state("playing_numbers")  # Return to number selection

    def exit(self):
        self.webcam.release()  # Release the webcam when exiting the state


class GalaxyExplorerPhrasesstate(VideoState):
    def __init__(self, game, video_key, next_state=None, audio_file=None):
        super().__init__(game, video_key, next_state, audio_file)
        
        # Load button images
        self.buttons = []
        
        button_data = [
            ("3_Phrases/GOODBYE.png", "playing_goodbye"), ("3_Phrases/HELLO.png", "playing_hello"), ("3_Phrases/ILOVEYOU.png", "playing_iloveyou"),
            ("3_Phrases/NO.png", "playing_no"), ("3_Phrases/PLEASE.png", "playing_please"), ("3_Phrases/SORRY.png", "playing_sorry"),
            ("3_Phrases/THANKYOU.png", "playing_thankyou"), ("3_Phrases/YES.png", "playing_yes")                                              
        ]
        
        for img_path, state in button_data:
            img = pygame.image.load(img_path).convert_alpha()
            img_rect = img.get_rect()
            collision_rect = get_collision_rect(img)
            self.buttons.append((img, img_rect, collision_rect, state))
        
        # Add back button
        self.back_button_img = pygame.image.load("BACK.png").convert_alpha()
        self.back_button_rect = self.back_button_img.get_rect(topleft=(10, 10))
        self.back_button_collision = get_collision_rect(self.back_button_img)

        
        # Adjust the back button collision area
        self.back_button_collision.height = 85
        self.back_button_collision.width = 90
        self.back_button_collision.x = 28
        self.back_button_collision.y = 35
        
        self.last_frame = None
        self.hovered_button = None

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
                pygame.mixer.Sound("CURSOR ON TOP.mp3").play()
            
            self.hovered_button = hovered
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check category buttons
            for _, _, collision_rect, state in self.buttons:
                if collision_rect.collidepoint(event.pos):
                    pygame.mixer.Sound("MOUSE CLICK.mp3").play()
                    # Transition to the appropriate state
                    print(f"Button clicked: {state}")
                    self.game.change_state(state)
                    return
            
            # Check back button
            if self.back_button_collision.collidepoint(event.pos):
                pygame.mixer.Sound("MOUSE CLICK.mp3").play()
                self.game.change_state("playing_galaxy")
                

class Game:
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((1024, 600))
        pygame.display.set_caption("SENYAS")
        
        # Initialize the current profile
        self.current_profile = None
        
        self.videos = {key: cv2.VideoCapture(f"{key.upper()}.mp4") for key in [
            "welcome", "intro", "usertype", "llanding",
            "glanding", "lgsign", "ggsign", "lplanet", "gplanet", "home", "blgsign", "gexplorer",
            "galpha", "gnum", "gphrases"
        ]}

        self.states = {
            "welcome": WelcomeState(self, "welcome", [
                ("LAUNCH.png", None, "playing_welcome")
            ], "LAUNCH SOUND.mp3"),
            "playing_welcome": VideoState(self, "welcome", "playing_intro"),
            "playing_intro": VideoState(self, "intro", "playing_usertype", "INTRO.mp3"),  
            "playing_usertype": UserTypeState(self, "usertype", [
                ("LEARNER.png", None, "playing_learner_planet"),
                ("GUARDIAN.png", None, "playing_guardian_planet")
            ], "USERTYPE.mp3"),  
            "load_game": LoadGameState(self, "blgsign"),
            "playing_learner_planet": VideoState(self, "lplanet", "playing_learner_landing"),
            "playing_learner_landing": VideoState(self, "llanding", "playing_blgsign", "LLANDING.mp3"),  
            "playing_blgsign": BLGSignState(self, "blgsign", [
                ("NEW GAME.png", None, "playing_lgsign"),
                ("LOAD GAME.png", None, "load_game")
            ]),
            "playing_lgsign": VideoWithSignInState(self, "lgsign", "playing_home"),
            "playing_guardian_planet": VideoState(self, "gplanet", "playing_guardian_landing"),
            "playing_guardian_landing": VideoState(self, "glanding", "playing_home", "GLANDING.mp3"),  
            "playing_home": HomeState(self, "home", None, "HOME.mp3"),
            "playing_galaxy": GalaxyExplorerState(self, "gexplorer", None, "HOME.mp3"),
            "playing_alphabets": GalaxyExplorerAlphabetState(self, "galpha", None, "GALPHASONG.mp3"),
            "playing_numbers": GalaxyExplorerNumberState(self, "gnum", None, "GNUMSONG.mp3"),
            "playing_phrases": GalaxyExplorerPhrasesstate(self, "gphrases", None, "GPHRASESSONG.mp3")
        }

        # Add number states dynamically
        for i in range(10):
            state_name = f"playing_{i}"
            image_path = os.path.join("GAME PROPER", "GEXPLORER NUMBER", f"{i}.png")
            self.states[state_name] = NumberDisplayState(self, image_path)

        self.current_state = self.states["welcome"]
        self.current_state.enter()
        self.clock = pygame.time.Clock()
    
    def change_state(self, new_state):
        if hasattr(self.current_state, 'sound') and self.current_state.sound:
            self.current_state.sound.stop()
            
        self.current_state.exit()
        self.current_state = self.states[new_state]
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