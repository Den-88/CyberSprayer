import threading
import time
import cv2
import numpy as np
from pyfirmata2 import Arduino
import signal
import sys
import psutil

# Конфигурация Arduino
ARDUINO_PORT = "/dev/ttyUSB0"  # Порт Arduino
LED_PIN = 13                   # Пин для светодиода
RELAY_PIN_1 = 2                # Пин для реле 1 (левая часть)
RELAY_PIN_2 = 3                # Пин для реле 2 (правая часть)

# Минимальная площадь объекта для обнаружения (в пикселях)
MIN_OBJECT_AREA = 500

# Настройка RTSP (используем одну камеру для эмуляции 12)
RTSP_URL = "rtsp://192.168.1.203:8555/profile0"  # Замени на адрес твоей камеры

# Инициализация Arduino
board = Arduino(ARDUINO_PORT)

def detect_green(frame, region=None):
    """Обнаружение зеленого цвета на кадре или его части."""
    if frame is None:
        return False, []  # Если кадра нет, ничего не делать

    # Если указана область, обрезаем кадр
    if region == "left":
        frame = frame[:, :frame.shape[1] // 2]  # Левая половина
    elif region == "right":
        frame = frame[:, frame.shape[1] // 2:]  # Правая половина

    # Преобразуем кадр в HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 30, 40])  # Нижняя граница зеленого
    upper_green = np.array([85, 255, 255])  # Верхняя граница зеленого

    # Создаем маску для зеленого цвета
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Находим контуры зеленых объектов
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Фильтруем контуры по минимальной площади
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_OBJECT_AREA]

    # Возвращаем результат и отфильтрованные контуры
    return len(filtered_contours) > 0, filtered_contours

class FrameCaptureThread(threading.Thread):
    """Поток для захвата кадров из RTSP-потока."""
    def __init__(self, rtsp_url):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(
            f"rtspsrc location={rtsp_url} protocols=tcp latency=0 ! rtph264depay ! h264parse ! openh264dec ! queue max-size-buffers=1 ! videoconvert ! appsink sync=false",
            cv2.CAP_GSTREAMER
        )
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        """Основной цикл потока для захвата кадров."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
        self.cap.release()  # Освобождаем ресурс при завершении

    def stop(self):
        """Остановка потока."""
        self.running = False
        self.join()

    def get_frame(self):
        """Получение текущего кадра."""
        with self.lock:
            return self.frame

def monitor_cpu_usage():
    """Функция для мониторинга загрузки процессора."""
    while True:
        cpu_usage = psutil.cpu_percent(interval=1)
        print(f"CPU Usage: {cpu_usage}%")
        time.sleep(1)

def signal_handler(sig, frame):
    """Обработчик сигнала для корректного завершения программы."""
    print("Программа завершена.")
    global running
    running = False
    capture_thread.stop()  # Останавливаем поток
    cv2.destroyAllWindows()  # Закрываем окна OpenCV
    sys.exit(0)  # Выход из программы

def main():
    """Основная функция программы."""
    global running

    # Запуск потока мониторинга процессора
    cpu_monitor_thread = threading.Thread(target=monitor_cpu_usage, daemon=True)
    cpu_monitor_thread.start()

    # Запуск потока захвата для одной камеры
    capture_thread = FrameCaptureThread(RTSP_URL)
    capture_thread.start()

    # Даем время потоку запуститься
    time.sleep(2)

    # Эмуляция 12 камер путем разделения кадра
    running = True
    while running:
        start_time = time.time()

        # Получаем кадр с камеры
        frame = capture_thread.get_frame()
        if frame is None:
            continue

        # Разделение кадра на 12 частей (3x4)
        height, width, _ = frame.shape
        regions = [
            frame[0:height//3, 0:width//4], frame[0:height//3, width//4:width//2], frame[0:height//3, width//2:3*width//4], frame[0:height//3, 3*width//4:],
            frame[height//3:2*height//3, 0:width//4], frame[height//3:2*height//3, width//4:width//2], frame[height//3:2*height//3, width//2:3*width//4], frame[height//3:2*height//3, 3*width//4:],
            frame[2*height//3:, 0:width//4], frame[2*height//3:, width//4:width//2], frame[2*height//3:, width//2:3*width//4], frame[2*height//3:, 3*width//4:]
        ]

        # Обработка каждой "виртуальной камеры"
        for i, region in enumerate(regions):
            green_detected, contours = detect_green(region)
            print(f"Camera {i+1} - Green Detected: {green_detected}")

        # Логирование времени обработки кадра
        end_time = time.time()
        print(f"Frame processed in {end_time - start_time:.4f} seconds")

    # Завершение работы
    capture_thread.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Устанавливаем обработчик сигнала для корректного завершения программы
    signal.signal(signal.SIGINT, signal_handler)
    main()
