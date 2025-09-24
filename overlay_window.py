# overlay_window.py
import sys
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout


class OverlayWindow(QWidget):
    defaultStyleText = "color: rgba(0, 230, 0, 0.8); font-size: 30px;"
    selectedStyleText = "color: rgba(0, 60, 150, 0.8); font-size: 38px;"

    def __init__(self):
        super().__init__()
        self._visibility_timer = QTimer()
        self._visibility_timer.setSingleShot(True)
        self._visibility_timer.timeout.connect(lambda: self.toggle_visibility(False))

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
        if n == 0 and not loading:
            self.clear_labels()
            return
        # print(f"Updating labels with {slot_data}")
        for i, label in enumerate(self.info_labels):
            slot_num = i + 1
            if loading:
                label.setText(f"slot-{slot_num}: loading...")
                label.setVisible(True)
                print("Loading...")
                continue
            if i > n - 1:
                label.setText(f"slot-{slot_num}: -----")
                label.setVisible(True)
                continue
            label.setText(f"slot-{slot_num}: {slot_data[i]['name']}")
            label.setVisible(True)  # Ensure label is visible when updated
        # Set Timer here to clear labels after 5 seconds
        self._visibility_timer.start(5000)

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
        self._visibility_timer.start(5000)
