# overlay_window.py
import sys
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout


class OverlayWindow(QWidget):

    def __init__(self):
        super().__init__()

        # Set window properties
        self.setWindowTitle("Overlay Info")
        self.setGeometry(400, 100, 300, 150)  # Position and size of window
        # Make window transparent
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlag(Qt.FramelessWindowHint)  # Remove borders
        self.setWindowFlag(Qt.WindowStaysOnTopHint)  # Keep window on top

        # Layout and labels
        layout = QVBoxLayout()

        # Information to display in the labels (slot-num: [name])
        self.info_labels = []
        info = [
            "slot-1: -",
            "slot-2: -",
            "slot-3: -",
            "slot-4: -"
        ]

        # Create and add labels to the layout
        for item in info:
            label = QLabel(item)
            label.setStyleSheet(
                "color: rgba(0, 230, 0, 0.8); font-size: 30px;")  # Style the text
            label.setAlignment(Qt.AlignLeft)
            layout.addWidget(label)
            self.info_labels.append(label)

        # Set the layout of the window
        self.setLayout(layout)

    def update_labels(self, slot_data):
        n = len(slot_data)
        if n < 7:
            slot_data = slot_data[-4+3:] # Pad with empty entries if less than 7
        else:
            slot_data = slot_data[-4:] # Get the last 4 entries
        for i, label in enumerate(self.info_labels):
            slot_num = i + 1
            label.setText(f"slot-{slot_num}: {slot_data[i]['name']}")
        # Set Timer here to clear labels after 5 seconds
        QTimer.singleShot(5000, lambda: self.toggle_visibility(False))

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
