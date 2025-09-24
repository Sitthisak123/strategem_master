# Strategem Master

Strategem Master is a Python desktop overlay tool for Helldivers 2 players. It uses OCR to detect strategem icons from your screen, displays them in a transparent overlay, and lets you trigger strategem key sequences with hotkeys.

---

## Features

- **Overlay Window:** Transparent PyQt5 overlay showing strategem slots and names.
- **OCR Detection:** Uses OpenCV and Tesseract to read strategem names from screenshots.
- **Hotkey Control:** Activate strategems or overlay actions using customizable hotkeys.
- **Automated Input:** Sends key sequences for strategems using PyAutoGUI.

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/strategem_master.git
   cd strategem_master
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR:**
   - Download and install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).
   - Update the path in `main.py` if needed:
     ```python
     pytesseract.pytesseract.tesseract_cmd = r"C:\Path\To\Tesseract-OCR\tesseract.exe"
     ```

---

## Usage

1. **Run the program:**
   ```sh
   python main.py
   ```

2. **Hotkeys:**
   - `Ctrl + ]` — Take a screenshot and update overlay with detected strategems.
   - `Ctrl + 1-4` — Activate strategem in slot 1-4.
   - `Ctrl + g` — Trigger "reinforce" strategem.
   - `Ctrl + v` — Trigger "resupply" strategem.
   - `Ctrl + Esc` — Exit the program.

3. **Overlay:**
   - The overlay window stays on top and displays detected strategems.
   - Selected strategem is highlighted when activated.

---

## Configuration

- **Strategem Data:** All strategem key sequences and names are in `main.py` (`strategems_all` dictionary).
- **Overlay Appearance:** Edit styles in `overlay_window.PY` for custom colors and font sizes.
- **Hotkeys:** Change hotkey mappings in `main.py` as needed.

---

## Troubleshooting

- **No strategems detected:** Make sure Tesseract OCR is installed and the path is correct.
- **Overlay not showing:** Check PyQt5 installation and run as administrator if needed.
- **Key input not working:** Some games may block simulated input; run as administrator.

---

## License

MIT License

---

## Credits

- [PyQt5](https://riverbankcomputing.com/software/pyqt/intro)
- [OpenCV](https://opencv.org/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [PyAutoGUI](https://pyautogui.readthedocs.io/en/latest/)

---

## Contributing

Pull requests and issues are welcome! Please open an issue for bugs or feature requests.

---

**Enjoy automated strategem mastery!**