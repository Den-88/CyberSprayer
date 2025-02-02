import threading
import time
import cv2
import numpy as np
from pyfirmata2 import Arduino
import signal
import sys

# board = Arduino("/dev/tty.usbserial-1421430")
board = Arduino("/dev/ttyUSB0")

led_pin = 13  # Пин для светодиода
relay_pin = 3  # Пин для реле

def detect_green(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])  # Нижняя граница
    upper_green = np.array([85, 255, 255])  # Верхняя граница
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = cv2.countNonZero(mask)
    height, width = frame.shape[:2]
    total_pixels = height * width
    green_ratio = green_pixels / total_pixels
    return green_ratio > 0.005, green_ratio  # Порог: 0.5% зелёных пикселей

# Параллельный поток для захвата кадров
class FrameCaptureThread(threading.Thread):
    def __init__(self, rtsp_url, frame_rate=30):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(
            f"rtspsrc location={rtsp_url} protocols=tcp latency=10 ! rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! appsink",
            cv2.CAP_GSTREAMER
        )
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.frame_rate = frame_rate  # Число кадров в секунду
        self.last_capture_time = 0  # Время последнего захвата кадра
        self.processing_frame = False  # Флаг для проверки, обрабатывается ли кадр

    def run(self):
        while self.running:
            if not self.processing_frame:  # Если кадр не обрабатывается
                current_time = time.time()
                if current_time - self.last_capture_time >= 1.0 / self.frame_rate:
                    ret, frame = self.cap.read()
                    if ret:
                        with self.lock:
                            self.frame = frame
                        self.last_capture_time = current_time
                        self.processing_frame = True  # Устанавливаем флаг обработки кадра

    def stop(self):
        self.running = False
        self.cap.release()

    def get_frame(self):
        with self.lock:
            return self.frame

    def frame_processed(self):
        """Устанавливаем флаг, что кадр обработан"""
        self.processing_frame = False

# Функция для корректного завершения программы
def signal_handler(sig, frame):
    print("Программа завершена.")
    cv2.destroyAllWindows()
    sys.exit(0)  # Выход из программы

# Запуск анализа видео из RTSP потока
def main():
    rtsp_url = "rtsp://192.168.1.203:8554/profile0"

    # Запуск потока захвата
    capture_thread = FrameCaptureThread(rtsp_url)
    capture_thread.start()

    spray_active = False
    spray_end_time = 0
    running = True

    while running:
        start_time = time.time()

        # Получаем кадр из потока
        frame = capture_thread.get_frame()
        if frame is None:
            continue

        green_detected, green_ratio = detect_green(frame)

        current_time = time.time()

        # Логика работы форсунки
        if green_detected:
            spray_active = True
            spray_end_time = current_time + 1  # Установить таймер на 1 секунду после обнаружения
        elif current_time > spray_end_time:
            spray_active = False

        # Управляем светодиодом на ардуино и форсункой
        board.digital[led_pin].write(green_detected)  # Переключаем светодиод
        board.digital[relay_pin].write(not spray_active)  # Переключаем форсунку
        print(f"Green ratio: {green_ratio:.6f}, Detected: {green_detected}, Spray: {spray_active}")

        # Обработка кадра завершена, сбрасываем флаг
        capture_thread.frame_processed()

        end_time = time.time()
        print(f"Frame processed in {end_time - start_time:.4f} seconds")

    capture_thread.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Устанавливаем обработчик сигнала для корректного завершения программы
    signal.signal(signal.SIGINT, signal_handler)
    main()
