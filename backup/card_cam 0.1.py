import sys
import json
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
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
        self.layout = QVBoxLayout(self.central_widget)
        
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.frame_count = 0  # Add this line
        self.detected_cards = []  # Add this line to store detected cards

        self.init_ui()
        self.load_card_data()
        self.load_card_templates()

    def init_ui(self):
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.info_label = QLabel(self)
        self.info_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.info_label)

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
                    template = cv2.resize(template, (100, 140))  # Resize template
                    self.templates[card_id] = template
        
        print(f"Loaded {len(self.templates)} templates")
        
        # Display a sample template
        if self.templates:
            sample_id, sample_template = next(iter(self.templates.items()))
            cv2.imshow("Sample Template", sample_template)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

    def detect_card(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.resize(gray_frame, (640, 480))  # Resize frame
        detected_cards = []

        print(f"Frame shape: {gray_frame.shape}")

        max_val = 0
        for card_id, template in self.templates.items():
            result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_val_temp, _, max_loc = cv2.minMaxLoc(result)
            if max_val_temp > max_val:
                max_val = max_val_temp

            threshold = 0.6  # Lowered threshold
            locations = np.where(result >= threshold)
            
            for pt in zip(*locations[::-1]):
                detected_cards.append((card_id, pt))

        print(f"Max match value: {max_val}")
        return detected_cards

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            if self.frame_count % 5 == 0:  # Only detect every 5 frames
                self.detected_cards = self.detect_card(frame)
            
            for card_id, pt in self.detected_cards:
                cv2.rectangle(frame, pt, (pt[0] + 100, pt[1] + 140), (0, 255, 0), 2)
                
                card_info = self.card_data[card_id]
                cv2.putText(frame, card_info['name'], (pt[0], pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if self.detected_cards:
                card_id = self.detected_cards[0][0]
                card_info = self.card_data[card_id]
                info_text = f"Name: {card_info['name']}\nSubname: {card_info['subname']}"
                self.info_label.setText(info_text)
            else:
                self.info_label.setText("No card detected")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.image_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.cap.release()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LorcanaDetector()
    window.setGeometry(100, 100, 800, 600)
    window.show()
    sys.exit(app.exec_())