# overlay_window.py
import sys
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPalette, QColor

class OverlayWindow(QWidget):
    defaultStyleText = "color: rgba(230, 230, 230, 0.8); font-size: 30px; font-family: monospace;"
    selectedStyleText = "color: rgba(30, 255, 30, 1); font-size: 38px; font-family: monospace;"
    overlayTimeout = 2500
    def __init__(self):
        super().__init__()
        self._visibility_timer = QTimer()
        self._visibility_timer.setSingleShot(True)
        self._visibility_timer.timeout.connect(lambda: self.toggle_visibility(False))

        # Set window properties
        self.setWindowTitle("Overlay Info")
        self.setGeometry(400, 100, 300, 150)  # Position and size of window
        
        self.setAttribute(Qt.WA_TranslucentBackground) # Make window transparent bg
        self.setWindowFlag(Qt.FramelessWindowHint)  # Remove borders
        self.setWindowFlag(Qt.WindowTransparentForInput)  # Click-through
        self.setWindowFlag(Qt.WindowStaysOnTopHint)  # Keep window on top
        self.setWindowFlag(Qt.Tool)  # Do not show in taskbar

        # Optional: subtle background for readability
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(110, 110, 100, 50))  # semi-transparent black
        self.setPalette(palette)

        # Layout and labels
        layout = QVBoxLayout()

        # Information to display in the labels (slot-num: [name])
        self.info_labels = []
        info = [
            "slot-1: -unknown-",
            "slot-2: -unknown-",
            "slot-3: -unknown-",
            "slot-4: -unknown-"
        ]

        # Create and add labels to the layout
        for item in info:
            label = QLabel(item)
            label.setStyleSheet(self.defaultStyleText)  # Style the text
            label.setAlignment(Qt.AlignLeft)
            layout.addWidget(label)
            self.info_labels.append(label)

        # Set the layout of the window
        self.setLayout(layout)

    def update_labels(self, slot_data=[], loading=False):
        n = len(slot_data)
        print(f"Updating labels with {slot_data}\{ len(slot_data) } items")
        for i, label in enumerate(self.info_labels):
            slot_num = i + 1
            if loading:
                label.setText(f"slot-{slot_num}: loading...")
                label.setStyleSheet(self.defaultStyleText)
                label.setVisible(True)
                print("Loading...")
                continue
            if i > n - 1:
                label.setText(f"slot-{slot_num}: -Empty slot-")
                label.setStyleSheet(self.defaultStyleText)
                label.setVisible(True)
                continue
            label.setText(f"slot-{slot_num}: {slot_data[i]['name']}")
            label.setStyleSheet(self.defaultStyleText)
            label.setVisible(True)  # Ensure label is visible when updated
        # Set Timer here to clear labels after 5 seconds
        if not loading:
            self._visibility_timer.start(self.overlayTimeout)
        else:
            self._visibility_timer.stop()

    def clear_labels(self):
        for label in self.info_labels:
            label.setText("")

    def toggle_visibility(self, value=None):
        if value is not None:
            for label in self.info_labels:
                label.setVisible(value)
        else:
            for label in self.info_labels:
                label.setVisible(not label.isVisible())

    def stg_Selected(self, slot_num):
        slot_num = int(slot_num)
        for i, label in enumerate(self.info_labels):
            if i == slot_num - 1:
                label.setStyleSheet(self.selectedStyleText)
                label.setVisible(True)  # Ensure label is visible when selected
            else:
                label.setStyleSheet(self.defaultStyleText)
                # Ensure label is visible when not selected
                label.setVisible(True)
        self._visibility_timer.start(self.overlayTimeout)
