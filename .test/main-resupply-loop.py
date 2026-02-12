"""
Resupply Loop Macro Script
Continuously executes the Resupply strategem in a loop until user exits (Ctrl+C)

Loop sequence:
1. Hold down Left Ctrl
2. Execute Resupply strategem code (4423)
3. Release Left Ctrl
4. Left mouse click
5. Delay 1 second
6. Repeat
"""

import pyautogui
import time
import keyboard

# Resupply strategem code from CSV
RESUPPLY_CODE = "4423"

# Set pyautogui pause between actions
pyautogui.PAUSE = 0.030

# Global exit flag
exit_macro = False


def execute_resupply_sequence():
    """Execute the Resupply strategem key sequence (4423)"""
    for key in str(RESUPPLY_CODE):
        match key:
            case '1':
                pyautogui.keyDown("left")
                pyautogui.keyUp("left")
                print("←", end="", flush=True)
            case '2':
                pyautogui.keyDown("up")
                pyautogui.keyUp("up")
                print("↑", end="", flush=True)
            case '3':
                pyautogui.keyDown("right")
                pyautogui.keyUp("right")
                print("→", end="", flush=True)
            case '4':
                pyautogui.keyDown("down")
                pyautogui.keyUp("down")
                print("↓", end="", flush=True)
            case _:
                print(f"Unknown key: {key}", end="", flush=True)


def on_hotkey_exit():
    """Callback function when Z+C hotkey is pressed"""
    global exit_macro
    exit_macro = True
    print("\n\nMacro stopped by user (Z+C)")


def resupply_loop_macro():
    """
    Main loop for resupply macro
    
    Sequence:
    - Hold Ctrl + execute Resupply sequence
    - Release Ctrl + left mouse click
    - Wait 1 second
    """
    global exit_macro
    
    # Register Z+C hotkey
    keyboard.add_hotkey('z+c', on_hotkey_exit)
    
    print("Resupply Loop Macro Started")
    print("Press Z+C to exit")
    print()
    
    iteration = 0
    while not exit_macro:
        iteration += 1
        print(f"[Iteration {iteration}] ", end="")
        
        # Hold down Left Ctrl using keyboard module
        print("(Ctrl Hold) ", end="", flush=True)
        keyboard.press("left ctrl")
        time.sleep(.5)
        
        # Execute Resupply sequence
        execute_resupply_sequence()
        time.sleep(.5)

        keyboard.release("left ctrl")
        print(" (Ctrl Released) ", end="", flush=True)      
        # Left mouse click
        time.sleep(.5)
        pyautogui.mouseDown()
        time.sleep(.05)
        pyautogui.mouseUp()
        print("(Click) ", end="", flush=True)
        
        # Delay 1 second
        print("(Waiting 1s)")
        time.sleep(1)

        for i in range(4):
            time.sleep(1)
            pyautogui.keyDown("e")
            pyautogui.keyUp("e")
            print("Pressed E ", end="", flush=True)
            print(f"(E) > {i+1}/4 ", end="", flush=True)

    
    # Cleanup
    keyboard.remove_hotkey('z+c')

if __name__ == "__main__":
    resupply_loop_macro()
