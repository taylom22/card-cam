import sys
import json
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy, QMessageBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor
import cv2
import numpy as np

class LorcanaDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lorcana Card Detector")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)
        
        # Set dark purple background
        palette = QPalette()
        # Background Color
        palette.setColor(QPalette.Window, QColor(48, 25, 52))  # Dark Purple Background
        # Text Color
        palette.setColor(QPalette.WindowText, Qt.white)  # White text
        self.setPalette(palette)
        
        # Configuration variables
        self.TEMPLATE_SIZE = (200, 280)
        self.ORB_FEATURES = 1000                # Default: 1000
        self.MATCH_DISTANCE_THRESHOLD = 50      # Default: 50
        self.MIN_GOOD_MATCHES = 15              # Default: 15
        self.MIN_CARD_AREA = 10                 # Default: 100
        self.MAX_CARD_AREA = 100000             # Default: 100000
        self.DETECTION_INTERVAL = 3             # Default: 3
        self.CONFIDENCE_THRESHOLD = 0.1         # Default: 0.1

        # Overlay Config
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX    # Default: cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.5                   # Default: 0.5
        self.FONT_COLOR = (0, 255, 0)           # Default: (0, 255, 0)
        self.FONT_THICKNESS = 1                 # Default: 1
        self.LINE_SPACING = 20                  # Default: 20

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.frame_count = 0
        self.detected_cards = []
        self.templates = {}
        self.card_data = {}
        self.last_detected_card_id = None

        self.orb = cv2.ORB_create(nfeatures=self.ORB_FEATURES, scoreType=cv2.ORB_FAST_SCORE)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.init_ui()
        self.load_card_data()
        self.load_card_templates()
        self.init_camera()

    def init_ui(self):
        # Create a widget for the webcam feed
        webcam_widget = QWidget()
        webcam_layout = QVBoxLayout(webcam_widget)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Allow expansion
        self.image_label.setMinimumSize(1, 1)  # Set a minimum size
        webcam_layout.addWidget(self.image_label)

        self.info_label = QLabel(self)
        self.info_label.setAlignment(Qt.AlignCenter)
        webcam_layout.addWidget(self.info_label)

        # Create a widget for the identified card with a black background
        card_widget = QWidget()
        card_widget.setStyleSheet("background-color: black;")
        card_layout = QVBoxLayout(card_widget)
        card_layout.setAlignment(Qt.AlignCenter)  # Center the layout within the widget

        self.card_name_label = QLabel(self)
        self.card_name_label.setAlignment(Qt.AlignCenter)
        self.card_name_label.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")  # Large text for card name
        card_layout.addWidget(self.card_name_label)

        self.card_subname_label = QLabel(self)
        self.card_subname_label.setAlignment(Qt.AlignCenter)
        self.card_subname_label.setStyleSheet("color: white; font-size: 18px;")  # Smaller text for subname
        card_layout.addWidget(self.card_subname_label)

        self.card_image_label = QLabel(self)
        self.card_image_label.setAlignment(Qt.AlignCenter)
        self.card_image_label.setFixedSize(300, 420)
        card_layout.addWidget(self.card_image_label)

        self.card_info_label = QLabel(self)
        self.card_info_label.setAlignment(Qt.AlignCenter)
        self.card_info_label.setStyleSheet("color: white;")
        card_layout.addWidget(self.card_info_label)

        # Add both widgets to the main layout
        self.layout.addWidget(webcam_widget, 2)  # Stretch factor to expand 2/3 of the frame
        self.layout.addWidget(card_widget, 1)  # Stretch factor to expand 1/3 of the frame

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 ms

    def init_camera(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Try using DirectShow
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Camera Error", "Could not open camera. Please check your camera connection and permissions.")
            sys.exit()

    def load_card_data(self):
        json_path = os.path.join(self.script_dir, 'cards.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.card_data = {card['id']: card for card in data['cards']}

    def load_card_templates(self):
        cards_dir = os.path.join(self.script_dir, 'cards')
        for filename in os.listdir(cards_dir):
            if filename.endswith('.png'):
                card_name = filename.split('_Set')[0].replace('_', ' ')
                card_id = next((card['id'] for card in self.card_data.values() if card['fullName'] == card_name), None)
                if card_id:
                    template_path = os.path.join(cards_dir, filename)
                    template = cv2.imread(template_path, 0)
                    template = cv2.resize(template, self.TEMPLATE_SIZE)
                    kp, des = self.orb.detectAndCompute(template, None)
                    self.templates[card_id] = (template, kp, des)
        
        print(f"Loaded {len(self.templates)} templates")

    def detect_card(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = self.orb.detectAndCompute(gray_frame, None)

        print(f"Number of keypoints in frame: {len(kp_frame)}")

        detected_cards = []
        for card_id, (template, kp_temp, des_temp) in self.templates.items():
            matches = self.bf.match(des_temp, des_frame)
            matches = sorted(matches, key=lambda x: x.distance)

            good_matches = [m for m in matches if m.distance < self.MATCH_DISTANCE_THRESHOLD]
            if len(good_matches) > self.MIN_GOOD_MATCHES:
                src_pts = np.float32([kp_temp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    h, w = template.shape
                    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    
                    area = cv2.contourArea(dst)
                    print(f"Detected area for {card_id}: {area}")
                    
                    if self.MIN_CARD_AREA < area < self.MAX_CARD_AREA:
                        confidence = len(good_matches) / len(kp_temp)
                        if confidence > self.CONFIDENCE_THRESHOLD:
                            detected_cards.append((card_id, dst, confidence))

        return detected_cards

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            if self.frame_count % self.DETECTION_INTERVAL == 0:
                self.detected_cards = self.detect_card(frame)
            
            for card_id, pts, confidence in self.detected_cards:
                pts = np.int32(pts).reshape(-1, 2)
                frame = cv2.polylines(frame, [pts], True, self.FONT_COLOR, 2)
                
                card_info = self.card_data[card_id]
                name = card_info['fullName']

                text_pos = (pts[0][0], pts[0][1] - self.LINE_SPACING)
                cv2.putText(frame, name, text_pos, 
                            self.FONT, self.FONT_SCALE, self.FONT_COLOR, self.FONT_THICKNESS)

            if self.detected_cards:
                new_card_id = self.detected_cards[0][0]
                if new_card_id != self.last_detected_card_id:
                    self.update_card_display(new_card_id)
                    self.last_detected_card_id = new_card_id

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            print("Failed to capture frame")

    def update_card_display(self, card_id):
        card_info = self.card_data[card_id]
        effects_text = ', '.join(effect['text'] for effect in card_info.get('effects', []))
        
        # Set the card name and subname
        self.card_name_label.setText(card_info['fullName'])
        self.card_subname_label.setText(card_info.get('subname', ''))

        # Set the card details below the image
        info_text = (
            f"Type: {card_info['type']}\n"
            f"Cost: {card_info['cost']}\n"
            f"Lore: {card_info.get('lore', 'N/A')}\n"
            f"Strength: {card_info.get('strength', 'N/A')}\n"
            f"Willpower: {card_info.get('willpower', 'N/A')}\n"
            f"Rarity: {card_info.get('rarity', 'N/A')}\n"
            f"Artist: {card_info.get('artist', 'N/A')}\n"
            f"Story: {card_info.get('story', 'N/A')}\n"
            f"Effects: {effects_text}"
        )
        self.card_info_label.setText(info_text)

        # Set the card image
        card_name = card_info['fullName'].replace(' ', '_')
        card_image_path = os.path.join(self.script_dir, 'cards', f"{card_name}_Set{card_info['setNumber']}_Card{card_info['number']}_{card_info['color']}.png")
        if os.path.exists(card_image_path):
            pixmap = QPixmap(card_image_path)
            pixmap = pixmap.scaled(300, 420, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.card_image_label.setPixmap(pixmap)
        else:
            print(f"No card found for {card_name}")
            self.card_image_label.setText("Image not found")
            self.card_image_label.setStyleSheet("color: white;")  # Set text color to white

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LorcanaDetector()
    window.setGeometry(100, 100, 1200, 600)
    window.show()
    sys.exit(app.exec_())