import keyboard
import time

# Function that runs when a hotkey is triggered
def hotkey_action(numpad_key):
    print(f"Ctrl + {numpad_key} is pressed!")

# Function to check when Ctrl is pressed and a numpad key is pressed simultaneously
def check_hotkey():
    while True:
        # Check if 'Ctrl' is being held down
        if keyboard.is_pressed('ctrl'):
            # List of numpad keys to check
            numpad_keys = [
                '0', '1', '2', '3', '4', '5', '6',
                '7', '8', '9', '+', '-', '*', '/'
            ]
            
            # Check for each numpad key if it's being pressed
            for key in numpad_keys:
                if keyboard.is_pressed(key):
                    # hotkey_action(key)
                    time.sleep(0.1)  # Prevent multiple triggers in a short time
                    break  # Once a key is pressed, break the loop
        time.sleep(0.1)  # Sleep to prevent high CPU usage

# Run the check_hotkey function
check_hotkey()
