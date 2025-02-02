import threading
import time
import cv2
import numpy as np
from pyfirmata2 import Arduino
import signal
import sys
from queue import Queue

# board = Arduino("/dev/tty.usbserial-1421430")
board = Arduino("/dev/ttyUSB0")

led_pin = 13  # Пин для светодиода
relay_pin = 3  # Пин для реле

def detect_green(frame):
    """Функция для обнаружения зеленых пикселей в кадре."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])  # Нижняя граница
    upper_green = np.array([85, 255, 255])  # Верхняя граница
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = cv2.countNonZero(mask)
    height, width = frame.shape[:2]
    total_pixels = height * width
    green_ratio = green_pixels / total_pixels
    return green_ratio > 0.005, green_ratio  # Порог: 0.5% зелёных пикселей

class FrameCaptureThread(threading.Thread):
    """Параллельный поток для захвата кадров."""
    def __init__(self, rtsp_url, frame_rate=30, frame_queue=None):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(
            f"rtspsrc location={rtsp_url} protocols=tcp latency=10 ! rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! appsink",
            cv2.CAP_GSTREAMER
        )
        self.frame_queue = frame_queue  # Очередь для кадров
        self.frame_rate = frame_rate  # Число кадров в секунду
        self.running = True

    def run(self):
        """Запуск потока захвата кадров."""
        while self.running:
            ret, frame = self.cap.read()
            if ret and self.frame_queue.qsize() < 5:  # Ограничиваем размер очереди
                self.frame_queue.put(frame)

    def stop(self):
        """Остановить поток захвата кадров."""
        self.running = False
        self.cap.release()

class FrameProcessingThread(threading.Thread):
    """Параллельный поток для обработки кадров."""
    def __init__(self, frame_queue):
        threading.Thread.__init__(self)
        self.frame_queue = frame_queue  # Очередь для кадров

    def run(self):
        """Запуск потока обработки кадров."""
        spray_active = False
        spray_end_time = 0
        while True:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()  # Получаем кадр из очереди

                # Обработка кадра для обнаружения зелёных пикселей
                green_detected, green_ratio = detect_green(frame)

                # Логика работы форсунки
                current_time = time.time()
                if green_detected:
                    spray_active = True
                    spray_end_time = current_time + 1  # Установить таймер на 1 секунду после обнаружения
                elif current_time > spray_end_time:
                    spray_active = False

                # Управление светодиодом и форсункой
                board.digital[led_pin].write(green_detected)  # Переключаем светодиод
                board.digital[relay_pin].write(not spray_active)  # Переключаем форсунку

                # Выводим информацию
                print(f"Green ratio: {green_ratio:.6f}, Detected: {green_detected}, Spray: {spray_active}")

            time.sleep(0.01)  # Небольшая задержка для предотвращения излишней загрузки процессора

# Функция для корректного завершения программы
def signal_handler(sig, frame):
    """Обработчик сигнала для корректного завершения программы."""
    print("Программа завершена.")
    cv2.destroyAllWindows()
    sys.exit(0)  # Выход из программы

def main():
    rtsp_url = "rtsp://192.168.1.203:8554/profile0"  # Укажите ваш RTSP URL

    # Очередь для кадров
    frame_queue = Queue(maxsize=5)

    # Запуск потоков
    capture_thread = FrameCaptureThread(rtsp_url, frame_queue=frame_queue)
    capture_thread.start()

    processing_thread = FrameProcessingThread(frame_queue=frame_queue)
    processing_thread.start()

    # Устанавливаем обработчик сигнала для корректного завершения программы
    signal.signal(signal.SIGINT, signal_handler)

    # Блокируем основной поток (он только для завершения программы)
    capture_thread.join()
    processing_thread.join()

if __name__ == "__main__":
    main()
