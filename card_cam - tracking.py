import sys
import json
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QMessageBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor
import cv2
import numpy as np

class Card:
    def __init__(self, card_id, bbox):
        self.card_id = card_id
        self.bbox = tuple(map(int, bbox))  # Ensure bbox values are integers
        self.tracker = cv2.TrackerMIL_create()  # Using TrackerMIL as an alternative
        self.tracked_frames = 0
        self.lost_frames = 0

    def init_tracker(self, frame):
        frame_height, frame_width = frame.shape[:2]
        print(f"Frame dimensions: width={frame_width}, height={frame_height}")
        print(f"Initializing tracker with bbox: {self.bbox}")
        x, y, w, h = self.bbox
        if 0 <= x < frame_width and 0 <= y < frame_height and w > 0 and h > 0 and x + w <= frame_width and y + h <= frame_height:
            return self.tracker.init(frame, self.bbox)
        else:
            print("Invalid bounding box for the given frame dimensions")
            return False

    def update_tracker(self, frame):
        success, bbox = self.tracker.update(frame)
        if success:
            self.bbox = bbox
            self.tracked_frames += 1
            self.lost_frames = 0
        else:
            self.lost_frames += 1
        return success

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
        self.MAX_CARD_AREA = 1000000             # Default: 100000
        self.DETECTION_INTERVAL = 3             # Default: 3
        self.CONFIDENCE_THRESHOLD = 0.05         # Default: 0.1

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

        self.tracked_cards = []
        self.max_tracking_frames = 30  # Adjust as needed

    def init_ui(self):
        # Create a widget for the webcam feed
        webcam_widget = QWidget()
        webcam_layout = QVBoxLayout(webcam_widget)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        webcam_layout.addWidget(self.image_label)

        self.info_label = QLabel(self)
        self.info_label.setAlignment(Qt.AlignCenter)
        webcam_layout.addWidget(self.info_label)

        # Create a widget for the identified card
        card_widget = QWidget()
        card_layout = QVBoxLayout(card_widget)

        self.card_image_label = QLabel(self)
        self.card_image_label.setAlignment(Qt.AlignCenter)
        self.card_image_label.setFixedSize(300, 420)
        card_layout.addWidget(self.card_image_label)

        self.card_info_label = QLabel(self)
        self.card_info_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(self.card_info_label)

        # Add both widgets to the main layout
        self.layout.addWidget(webcam_widget)
        self.layout.addWidget(card_widget)

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
        
        if des_frame is None:
            print("No descriptors found in the frame.")
            return []

        print(f"Frame descriptors: shape={des_frame.shape}, dtype={des_frame.dtype}")

        detected_cards = []
        for card_id, (template, kp_temp, des_temp) in self.templates.items():
            if des_temp is None:
                print(f"No descriptors found for template: {card_id}")
                continue

            print(f"Template descriptors for {card_id}: shape={des_temp.shape}, dtype={des_temp.dtype}")

            if des_temp.dtype != des_frame.dtype or des_temp.shape[1] != des_frame.shape[1]:
                print(f"Descriptor type or shape mismatch for card: {card_id}")
                continue

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
                            x, y, w, h = cv2.boundingRect(dst)
                            detected_cards.append((card_id, (x, y, w, h), confidence))

        return detected_cards

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            print(f"Frame shape: {frame.shape}, Frame dtype: {frame.dtype}")
            
            # Update existing tracked cards
            for card in self.tracked_cards[:]:
                success = card.update_tracker(frame)
                if not success or card.lost_frames > 5:  # Adjust threshold as needed
                    self.tracked_cards.remove(card)
                elif card.tracked_frames > self.max_tracking_frames:
                    # Re-detect the card after a certain number of frames
                    self.tracked_cards.remove(card)
            
            # Detect new cards periodically
            if self.frame_count % self.DETECTION_INTERVAL == 0:
                detected_cards = self.detect_card(frame)
                for card_id, bbox, confidence in detected_cards:
                    print(f"Detected card: {card_id}, bbox: {bbox}, confidence: {confidence}")
                    # Check if this card is already being tracked
                    if not any(card.card_id == card_id for card in self.tracked_cards):
                        new_card = Card(card_id, bbox)
                        if new_card.init_tracker(frame):
                            self.tracked_cards.append(new_card)
                            self.update_card_display(card_id)
            
            # Draw bounding boxes for all tracked cards
            for card in self.tracked_cards:
                x, y, w, h = [int(v) for v in card.bbox]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                card_info = self.card_data[card.card_id]
                cv2.putText(frame, card_info['fullName'], (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.image_label.setPixmap(pixmap)
        else:
            print("Failed to capture frame")

    def update_card_display(self, card_id):
        card_info = self.card_data[card_id]
        info_text = f"Name: {card_info['fullName']}\nType: {card_info['type']}\nCost: {card_info['cost']}"
        self.card_info_label.setText(info_text)
        self.card_info_label.setStyleSheet("color: white;")  # Set text color to white

        card_name = card_info['fullName'].replace(' ', '_')
        card_image_path = os.path.join(self.script_dir, 'cards', f"{card_name}_Set{card_info['setNumber']}_Card{card_info['number']}_{card_info['color']}.png")
        if os.path.exists(card_image_path):
            pixmap = QPixmap(card_image_path)
            pixmap = pixmap.scaled(300, 420, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.card_image_label.setPixmap(pixmap)
        else:
            print("No card found")
            self.card_image_label.setText("Image not found")
            self.card_image_label.setStyleSheet("color: white;")  # Set text color to white

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LorcanaDetector()
    window.setGeometry(100, 100, 1200, 600)
    window.show()
    sys.exit(app.exec_())