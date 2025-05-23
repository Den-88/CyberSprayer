import threading
import time
import cv2
import numpy as np
from pyfirmata2 import Arduino
import signal
import sys

# Конфигурация Arduino
ARDUINO_PORT = "/dev/ttyUSB0"  # Порт Arduino
LED_PIN = 13                   # Пин для светодиода
RELAY_PIN_1 = 2                # Пин для реле 1 (левая часть)
RELAY_PIN_2 = 3                # Пин для реле 2 (правая часть)

# Минимальная площадь объекта для обнаружения (в пикселях)
MIN_OBJECT_AREA = 150

# Настройки RTSP
# RTSP_URL = "rtsp://192.168.1.203:8555/main"
RTSP_URL = "rtsp://admin:denis123@192.168.1.203:8555/main"

RTSP_OUTPUT_PIPELINE = (
    # "appsrc ! videoconvert ! video/x-raw,format=I420 ! x264enc tune=zerolatency bitrate=1500 speed-preset=ultrafast "
    # "! h264parse ! rtspclientsink location=rtsp://127.0.0.1:8554/test"
    "appsrc ! videoconvert ! video/x-raw,format=NV12 ! x264enc tune=zerolatency bitrate=5000 speed-preset=ultrafast key-int-max=30 "
    "! h264parse ! rtspclientsink location=rtsp://127.0.0.1:8554/test"
)

# Флаг для включения/отключения вывода изображения
ENABLE_OUTPUT = True  # По умолчанию вывод отключен

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


def draw_text_with_background(frame, text, position, font, scale, color, thickness, bg_color, alpha=0.5):
    """Добавление текста с фоном на изображение."""
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    text_w, text_h = text_size
    x, y = position

    # Рисуем полупрозрачный фон
    bg_x1, bg_y1 = x - 5, y - text_h - 5
    bg_x2, bg_y2 = x + text_w + 5, y + 5
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Добавляем текст
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)


class FrameCaptureThread(threading.Thread):
    """Поток для захвата кадров из RTSP-потока."""
    def __init__(self, rtsp_url):
        threading.Thread.__init__(self)
        # self.cap = cv2.VideoCapture(
        #     f"rtspsrc location={rtsp_url} protocols=tcp latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! queue max-size-buffers=1 ! videoconvert ! appsink sync=false",
        #     cv2.CAP_GSTREAMER
        # )

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


def signal_handler(sig, frame):
    """Обработчик сигнала для корректного завершения программы."""
    print("Программа завершена.")
    global running
    running = False

    if capture_thread:
        capture_thread.stop()  # Останавливаем поток
    if ENABLE_OUTPUT and out:
        out.release()  # Закрываем видео-поток

    cv2.destroyAllWindows()  # Закрываем окна OpenCV
    sys.exit(0)  # Выход из программы


def main():
    """Основная функция программы."""
    global running, capture_thread, out

    # Запуск потока захвата кадров
    capture_thread = FrameCaptureThread(RTSP_URL)
    capture_thread.start()

    # Даем время потоку запуститься
    time.sleep(2)
    if not capture_thread.cap.isOpened():
        print("Ошибка: невозможно открыть RTSP-поток. Завершаем программу.")
        capture_thread.stop()
        sys.exit(1)

    # Инициализация RTSP-вывода, если вывод включен
    if ENABLE_OUTPUT:
        out = cv2.VideoWriter(RTSP_OUTPUT_PIPELINE, cv2.CAP_GSTREAMER, 0, 25, (1920, 1080), True)

    # Основной цикл обработки кадров
    running = True
    spray_active_left = False
    spray_active_right = False
    spray_end_time_left = 0
    spray_end_time_right = 0

    while running:
        start_time = time.time()

        # Получаем кадр из потока
        frame = capture_thread.get_frame()
        if frame is None:
            continue

        # Рисуем вертикальную белую линию посередине
        height, width = frame.shape[:2]
        cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)

        # Анализ левой половины кадра
        green_detected_left, contours_left = detect_green(frame, region="left")
        current_time = time.time()

        # Логика работы форсунки для левой части
        if green_detected_left:
            spray_active_left = True
            spray_end_time_left = current_time + 0.3  # Таймер на 0.3 секунды
        elif current_time > spray_end_time_left:
            spray_active_left = False

        # Анализ правой половины кадра
        green_detected_right, contours_right = detect_green(frame, region="right")

        # Логика работы форсунки для правой части
        if green_detected_right:
            spray_active_right = True
            spray_end_time_right = current_time + 0.3  # Таймер на 0.3 секунды
        elif current_time > spray_end_time_right:
            spray_active_right = False

        # Управление Arduino
        board.digital[LED_PIN].write(green_detected_left or green_detected_right)  # Светодиод
        board.digital[RELAY_PIN_1].write(not spray_active_left)  # Реле 1 (левая часть)
        board.digital[RELAY_PIN_2].write(not spray_active_right)  # Реле 2 (правая часть)

        # Логирование
        print(f"Left Detected: {green_detected_left}, Spray: {spray_active_left}")
        print(f"Right Detected: {green_detected_right}, Spray: {spray_active_right}")

        # Обводка зеленых объектов на левой половине и отображение площади
        for contour in contours_left:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            area = cv2.contourArea(contour)
            cv2.putText(frame, f"S = {area}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Обводка зеленых объектов на правой половине и отображение площади
        for contour in contours_right:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x + width // 2, y), (x + width // 2 + w, y + h), (0, 255, 0), 2)
            area = cv2.contourArea(contour)
            cv2.putText(frame, f"S = {area}", (x + width // 2, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Добавление текста на кадр (левая часть)
        draw_text_with_background(
            frame,
            f"Left Detected: {green_detected_left}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if green_detected_left else (0, 0, 255),
            2,
            (0, 0, 0),
        )
        draw_text_with_background(
            frame,
            f"Left Spray: {spray_active_left}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if spray_active_left else (0, 0, 255),
            2,
            (0, 0, 0),
        )

        # Добавление текста на кадр (правая часть)
        draw_text_with_background(
            frame,
            f"Right Detected: {green_detected_right}",
            (width // 2 + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if green_detected_right else (0, 0, 255),
            2,
            (0, 0, 0),
        )
        draw_text_with_background(
            frame,
            f"Right Spray: {spray_active_right}",
            (width // 2 + 10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if spray_active_right else (0, 0, 255),
            2,
            (0, 0, 0),
        )

        # Отправка кадра в RTSP, если вывод включен
        if ENABLE_OUTPUT:
            out.write(frame)

        # Логирование времени обработки кадра
        end_time = time.time()
        print(f"Frame processed in {end_time - start_time:.4f} seconds")

    # Завершение работы
    capture_thread.stop()
    if ENABLE_OUTPUT:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Устанавливаем обработчик сигнала для корректного завершения программы
    signal.signal(signal.SIGINT, signal_handler)
    main()