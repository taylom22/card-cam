import sys
import json
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np

class LorcanaDetector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lorcana Card Detector")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)
        
        # Configuration variables
        self.TEMPLATE_SIZE = (200, 280)
        self.ORB_FEATURES = 1000  # Increased from 1000
        self.MATCH_DISTANCE_THRESHOLD = 50  # Increased from 30
        self.MIN_GOOD_MATCHES = 10  # Decreased from 30
        self.MIN_CARD_AREA = 1  # Decreased from 5000
        self.MAX_CARD_AREA = 1000000  # Increased from 50000
        self.DETECTION_INTERVAL = 3  # Decreased from 5
        self.CONFIDENCE_THRESHOLD = 0.05  # Decreased from 0.6
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.FONT_SCALE = 0.5
        self.FONT_COLOR = (0, 255, 0)
        self.FONT_THICKNESS = 1
        self.LINE_SPACING = 20

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.frame_count = 0
        self.detected_cards = []
        self.templates = {}
        self.card_data = {}
        self.last_detection = None
        self.detection_history = []

        self.orb = cv2.ORB_create(nfeatures=self.ORB_FEATURES, scoreType=cv2.ORB_FAST_SCORE)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.init_ui()
        self.load_card_data()
        self.load_card_templates()

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
        self.card_image_label.setFixedSize(300, 420)  # Set a fixed size for the card image
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

        self.cap = cv2.VideoCapture(0)

    def load_card_data(self):
        json_path = os.path.join(self.script_dir, 'cards', 'cards.config.json')
        with open(json_path, 'r') as f:
            self.card_data = json.load(f)

    def load_card_templates(self):
        self.templates = {}
        cards_dir = os.path.join(self.script_dir, 'cards')
        for filename in os.listdir(cards_dir):
            if filename.endswith('.png'):
                card_id = filename[:-4]  # Remove .png extension
                if card_id in self.card_data:
                    template_path = os.path.join(cards_dir, filename)
                    template = cv2.imread(template_path, 0)
                    template = cv2.resize(template, self.TEMPLATE_SIZE)
                    kp, des = self.orb.detectAndCompute(template, None)
                    self.templates[card_id] = (template, kp, des)
        
        print(f"Loaded {len(self.templates)} templates")

    def detect_card(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = self.orb.detectAndCompute(gray_frame, None)

        detected_cards = []
        for card_id, (template, kp_temp, des_temp) in self.templates.items():
            matches = self.bf.match(des_temp, des_frame)
            matches = sorted(matches, key=lambda x: x.distance)

            good_matches = [m for m in matches if m.distance < self.MATCH_DISTANCE_THRESHOLD]
            print(f"Card {card_id}: {len(good_matches)} good matches")
            
            if len(good_matches) > self.MIN_GOOD_MATCHES:
                src_pts = np.float32([kp_temp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if M is not None:
                    h, w = template.shape
                    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    
                    area = cv2.contourArea(dst)
                    print(f"Card {card_id}: Area = {area}")
                    
                    if self.MIN_CARD_AREA < area < self.MAX_CARD_AREA:
                        confidence = len(good_matches) / len(kp_temp)
                        print(f"Card {card_id}: Confidence = {confidence}")
                        if confidence > self.CONFIDENCE_THRESHOLD:
                            detected_cards.append((card_id, dst, confidence))
                        else:
                            print(f"Card {card_id}: Confidence too low")
                    else:
                        print(f"Card {card_id}: Area out of range")

        return detected_cards

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            if self.frame_count % self.DETECTION_INTERVAL == 0:
                new_detections = self.detect_card(frame)
                if new_detections:
                    self.detection_history.append(new_detections[0][0])  # Add the most confident detection
                    if len(self.detection_history) > 5:
                        self.detection_history.pop(0)
                    
                    # Use the most common detection in the last 5 frames
                    from collections import Counter
                    most_common = Counter(self.detection_history).most_common(1)
                    if most_common:
                        self.last_detection = most_common[0][0]
                        self.detected_cards = [d for d in new_detections if d[0] == self.last_detection]
                else:
                    self.detected_cards = []
            
            for card_id, pts, confidence in self.detected_cards:
                pts = np.int32(pts).reshape(-1, 2)
                frame = cv2.polylines(frame, [pts], True, self.FONT_COLOR, 2)
                
                card_info = self.card_data[card_id]
                name = card_info['name']
                subname = card_info['subname']

                text_pos_name = (pts[0][0], pts[0][1] - self.LINE_SPACING)
                text_pos_subname = (pts[0][0], pts[0][1] - 2*self.LINE_SPACING)

                cv2.putText(frame, name, text_pos_name, 
                            self.FONT, self.FONT_SCALE, self.FONT_COLOR, self.FONT_THICKNESS)
                cv2.putText(frame, subname, text_pos_subname, 
                            self.FONT, self.FONT_SCALE, self.FONT_COLOR, self.FONT_THICKNESS)

            if self.last_detection:
                self.update_card_display(self.last_detection)
            else:
                self.card_image_label.clear()
                self.card_info_label.setText("No card detected")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.image_label.setPixmap(pixmap)

    def update_card_display(self, card_id):
        card_info = self.card_data[card_id]
        info_text = f"Name: {card_info['name']}\nSubname: {card_info['subname']}"
        self.card_info_label.setText(info_text)

        # Load and display the card image
        card_image_path = os.path.join(self.script_dir, 'cards', f"{card_id}.png")
        if os.path.exists(card_image_path):
            pixmap = QPixmap(card_image_path)
            pixmap = pixmap.scaled(300, 420, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.card_image_label.setPixmap(pixmap)
        else:
            self.card_image_label.setText("Image not found")

    def closeEvent(self, event):
        self.cap.release()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LorcanaDetector()
    window.setGeometry(100, 100, 1200, 600)
    window.show()
    sys.exit(app.exec_())