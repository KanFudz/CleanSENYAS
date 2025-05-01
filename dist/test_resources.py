import os
import sys
from pathlib import Path

def verify_resources():
    required_folders = [
        "AUDIO", "BUTTONS", "FONTS", "MODEL", 
        "SCENES", "GAME PROPER", "saves"
    ]
    
    base_path = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
    
    for folder in required_folders:
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            print(f"ERROR: Missing folder {folder} at {folder_path}")
            continue
            
        print(f"Checking {folder}...")
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, 'rb') as f:
                        f.read(1)
                    print(f"  ✓ {os.path.relpath(full_path, base_path)}")
                except Exception as e:
                    print(f"  ✗ Error accessing {full_path}: {str(e)}")

if __name__ == "__main__":
    verify_resources()