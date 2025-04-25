from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.label import Label
import cv2
import pytesseract

ocr_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'

class WeighmentOCRApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)
        layout = BoxLayout(orientation='vertical')

        self.image = Image()
        self.label = Label(text="Weight: ", font_size=32, size_hint_y=0.2)

        layout.add_widget(self.image)
        layout.add_widget(self.label)

        Clock.schedule_interval(self.update, 1.0 / 5.0)  # 5 FPS
        return layout

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        # OCR Processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh, config=ocr_config).strip()
        self.label.text = f"Weight: {text} kg"

        # Convert image to texture
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

    def on_stop(self):
        self.capture.release()


if __name__ == '__main__':
    WeighmentOCRApp().run()
