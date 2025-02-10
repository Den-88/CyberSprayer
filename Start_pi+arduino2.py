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


def capture_frame(rtsp_url):
    """Захват кадров с минимальной задержкой."""
    cap = cv2.VideoCapture(
        f"rtspsrc location={rtsp_url} protocols=tcp latency=1 ! rtph265depay ! h265parse ! avdec_h265 ! videoconvert ! appsink",
        cv2.CAP_GSTREAMER
    )

    if not cap.isOpened():
        print("Не удалось открыть поток RTSP.")
        return None

    # Захват кадра без накопления
    ret, frame = cap.read()
    cap.release()

    if ret:
        return frame
    else:
        return None


def process_frame(frame):
    """Обработка кадра для обнаружения зелёных пикселей."""
    green_detected, green_ratio = detect_green(frame)

    # Логика работы форсунки
    spray_active = False
    spray_end_time = time.time() + 1  # Таймер на 1 секунду после обнаружения
    if green_detected:
        spray_active = True
    elif time.time() > spray_end_time:
        spray_active = False

    # Управление светодиодом и форсункой
    board.digital[led_pin].write(green_detected)  # Переключаем светодиод
    board.digital[relay_pin].write(not spray_active)  # Переключаем форсунку

    # Выводим информацию
    print(f"Green ratio: {green_ratio:.6f}, Detected: {green_detected}, Spray: {spray_active}")


def main():
    rtsp_url = "rtsp://192.168.1.203:8555/profile0"  # Укажите ваш RTSP URL
    running = True

    while running:
        # Захват кадра с минимальной задержкой
        frame = capture_frame(rtsp_url)
        if frame is None:
            print("Не удалось захватить кадр.")
            # time.sleep(0.1)  # Немного подождём перед повторной попыткой
            continue

        # Обработка кадра
        process_frame(frame)

        # Проверка нажатия клавиши 'q' для выхода
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            running = False

    print("Завершение программы.")


if __name__ == "__main__":
    # Устанавливаем обработчик сигнала для корректного завершения программы
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    main()
