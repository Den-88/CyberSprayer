import threading
import time

import cv2
import numpy as np
from pyfirmata2 import Arduino, util
import signal
import sys
import subprocess


# board = Arduino("/dev/tty.usbserial-1421430")
board = Arduino("/dev/ttyUSB0")

led_pin = 13  # Пин для светодиода
relay_pin_1 = 2  # Пин для реле 1
relay_pin_2 = 3  # Пин для реле 2

def detect_green(frame):
    if frame is None:
        return False, 0  # Если кадра нет, ничего не делать

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 30, 40])  # Нижняя граница
    upper_green = np.array([85, 255, 255])  # Верхняя граница
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = cv2.countNonZero(mask)
    height, width = frame.shape[:2]
    total_pixels = height * width
    green_ratio = green_pixels / total_pixels
    return green_ratio > 0.00400, green_ratio  # Порог: 1.0% зелёных пикселей


# Функция для добавления текста с фоном на изображение
def draw_text_with_background(frame, text, position, font, scale, color, thickness, bg_color, alpha=0.5):
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    text_w, text_h = text_size
    x, y = position
    bg_x1, bg_y1 = x - 5, y - text_h - 5
    bg_x2, bg_y2 = x + text_w + 5, y + 5
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)


# Параллельный поток для захвата кадров
class FrameCaptureThread(threading.Thread):
    def __init__(self, rtsp_url):
        threading.Thread.__init__(self)
        # self.cap = cv2.VideoCapture(
        #     f"rtspsrc location={rtsp_url} protocols=tcp latency=10 ! rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! appsink",
        #     cv2.CAP_GSTREAMER
        # )

        # self.cap = cv2.VideoCapture(
        #     f"rtspsrc location={rtsp_url} protocols=tcp latency=0 drop=true ! rtph264depay ! h264parse ! avdec_h264 ! queue max-size-buffers=1 ! videoconvert ! appsink sync=false",
        #     cv2.CAP_GSTREAMER
        # )

        self.cap = cv2.VideoCapture(
            f"rtspsrc location={rtsp_url} protocols=tcp latency=0 ! rtph264depay ! h264parse ! avdec_h264 ! queue max-size-buffers=1 ! videoconvert ! appsink sync=false",
            cv2.CAP_GSTREAMER
        )
        # rtsp_url = "rtsp://192.168.1.203:8555/profile0"
        # self.cap = cv2.VideoCapture(rtsp_url)

        self.frame = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            # time.sleep(0.01)
        self.cap.release()  # Освобождаем ресурс при завершении


    def stop(self):
        self.running = False
        self.join()  # Ждем завершения потока

    def get_frame(self):
        with self.lock:
            return self.frame

# Функция для корректного завершения программы
def signal_handler(sig, frame):
    print("Программа завершена.")

    running = False  # Останавливаем главный цикл

    if capture_thread:
        capture_thread.stop()  # Останавливаем поток
    if out:
        out.release()  # Закрываем видео-поток

    cv2.destroyAllWindows()  # Закрываем окна OpenCV

    sys.exit(0)  # Выход из программы


# Запуск анализа видео из RTSP потока
def main():
    global running, capture_thread, out

    rtsp_url = "rtsp://192.168.1.203:8555/profile0"

    # Запуск потока захвата
    capture_thread = FrameCaptureThread(rtsp_url)
    capture_thread.start()

    # Проверяем, открылся ли поток
    time.sleep(2)  # Даем время потоку запуститься
    if not capture_thread.cap.isOpened():
        print("Ошибка: невозможно открыть RTSP-поток. Завершаем программу.")
        capture_thread.stop()
        sys.exit(1)  # Завершаем с кодом ошибки


    spray_active = False
    spray_end_time = 0

    running = True

    # Устанавливаем обработчик сигнала для корректного завершения программы
    # signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, running))

    # Создаём поток для RTSP вывода
    # fourcc = cv2.VideoWriter_fourcc(*'H264')
    # out = cv2.VideoWriter('appsrc ! video/x-raw,format=BGR ! videoconvert ! x264enc ! rtspclientsink location=rtsp://127.0.0.1:8554/test', fourcc, 25, (640, 480))

    # Создаём GStreamer-пайплайн для RTSP вывода
    pipeline = (
        "appsrc ! videoconvert ! video/x-raw,format=I420 ! x264enc tune=zerolatency bitrate=1500 speed-preset=ultrafast "
        "! h264parse ! rtspclientsink location=rtsp://127.0.0.1:8554/test"
    )

    # fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, 25, (1920, 1080), True)

    # Создание RTSP сервера с использованием GStreamer
    # def start_rtsp_server():
    #     gst_command = [
    #         'gst-launch-1.0',
    #         'rtspsrc',
    #         'location=rtsp://127.0.0.1:8554/test',
    #         'protocols=tcp',
    #         'latency=10',
    #         '! decodebin',
    #         '! videoconvert',
    #         '! autovideosink'
    #     ]
    #     subprocess.Popen(gst_command)
    #
    # # Запуск RTSP сервера в отдельном процессе
    # start_rtsp_server()


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
            spray_end_time = current_time + 0.3  # Установить таймер на 1 секунду после обнаружения
        elif current_time > spray_end_time:
            spray_active = False

        # Управляем светодиодом на ардуино и форсункой
        board.digital[led_pin].write(green_detected)  # Переключаем светодиод
        board.digital[relay_pin_1].write(not spray_active)  # Переключаем форсунку
        print(f"Green ratio: {green_ratio:.6f}, Detected: {green_detected}, Spray: {spray_active}")

        # Добавление текста и статуса
        draw_text_with_background(
            frame,
            f"GREEN: {green_ratio * 100:.2f}%",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            (0, 0, 0),
        )

        spray_status = "Spray: ON" if spray_active else "Spray: OFF"
        spray_color = (0, 255, 0) if spray_active else (0, 0, 255)
        draw_text_with_background(
            frame,
            spray_status,
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            spray_color,
            2,
            (0, 0, 0),
            alpha=0.5
        )

        # Отображение результата
        if green_detected:
            draw_text_with_background(
                frame,
                "Green detected!",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                (0, 0, 0),
            )
        else:
            draw_text_with_background(
                frame,
                f"No green detected!",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                (0, 0, 0),
            )
        #
        # # Показ кадра
        # cv2.imshow("Green Color Detection", frame)

    #     # Проверка нажатия клавиши 'q' для выхода
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord("q"):
    #         running = False
    #
    #     #

        # Отправка изображения на RTSP
        out.write(frame)

        end_time = time.time()
        print(f"Frame processed in {end_time - start_time:.4f} seconds")
    #
    # capture_thread.stop()
    # cv2.destroyAllWindows()

    capture_thread.stop()
    out.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    # Устанавливаем обработчик сигнала для корректного завершения программы

    signal.signal(signal.SIGINT, signal_handler)
    main()
