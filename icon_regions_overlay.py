# icon_regions_overlay.py
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap, QPalette, QColor, QImage

class IconRegionsOverlay(QWidget):
    overlayTimeout = 3000  # Hide timeout (3 seconds)
    destroyTimeout = 3500  # Destroy timeout (3.5 seconds)
    
    def __init__(self):
        super().__init__()
        self._visibility_timer = QTimer()
        self._visibility_timer.setSingleShot(True)
        self._visibility_timer.timeout.connect(self._on_hide_timeout)
        
        self._destroy_timer = QTimer()
        self._destroy_timer.setSingleShot(True)
        self._destroy_timer.timeout.connect(self.cleanup_and_destroy)
        
        # Set window properties
        self.setWindowTitle("Icon Regions Detection")
        self.setGeometry(750, 100, 640, 850)  # Position and size of window
        self._first_show = True
        
        self.setAttribute(Qt.WA_TranslucentBackground)  # Make window transparent bg
        self.setWindowFlag(Qt.FramelessWindowHint)  # Remove borders
        self.setWindowFlag(Qt.WindowTransparentForInput)  # Click-through
        self.setWindowFlag(Qt.WindowStaysOnTopHint)  # Keep window on top
        self.setWindowFlag(Qt.Tool)  # Do not show in taskbar
        
        # Set background
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(110, 110, 100, 50))
        self.setPalette(palette)
        
        # Layout and image label
        layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(self.image_label)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
    
    def display_icon_regions(self, screenshot, icon_boxes):
        """
        Display the screenshot with detected icon regions drawn as rectangles
        
        Args:
            screenshot: numpy array of the screenshot (BGR format)
            icon_boxes: list of (y, x, w, h) tuples for detected regions
        """
        # Extract HUD area (same as in main.py)
        hud = screenshot[30:800, 30:600]
        
        # Create a copy to draw on
        display_img = hud.copy()
        
        # Draw bounding boxes for each detected icon region
        for idx, (y, x, w, h) in enumerate(icon_boxes):
            # Draw rectangle
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw index label
            cv2.putText(display_img, str(idx + 1), (x + 5, y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Convert BGR to RGB for display
        display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
        # Convert numpy array to QImage
        h, w, ch = display_img_rgb.shape
        bytes_per_line = 3 * w
        qt_image = QImage(display_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Display in label
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)
        
        # Show window on first display
        if self._first_show:
            self.show()
            self._first_show = False
        
        # Make sure window is visible and raised
        self.setVisible(True)
        self.raise_()
        self.activateWindow()
        
        # Stop any active timers before restarting
        if self._visibility_timer.isActive():
            self._visibility_timer.stop()
        if self._destroy_timer.isActive():
            self._destroy_timer.stop()
        
        # Start new timers
        self._visibility_timer.start(self.overlayTimeout)
        self._destroy_timer.start(self.destroyTimeout)
    
    def _on_hide_timeout(self):
        """Callback for visibility timer - hide the window"""
        self.setVisible(False)
    
    def toggle_visibility(self, value=None):
        if value is not None:
            self.setVisible(value)
        else:
            self.setVisible(not self.isVisible())
    
    def cleanup_and_destroy(self):
        """Clean up resources and destroy the overlay window"""
        # Stop any active timers
        if self._visibility_timer.isActive():
            self._visibility_timer.stop()
        if self._destroy_timer.isActive():
            self._destroy_timer.stop()
        
        # Clear image label
        self.image_label.clear()
        self.image_label.setPixmap(QPixmap())
        
        # Hide and destroy window
        self.setVisible(False)
        self.close()
