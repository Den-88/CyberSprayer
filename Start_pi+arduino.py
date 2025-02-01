import time

import cv2
import numpy as np
from pyfirmata2 import Arduino, util

# board = Arduino("/dev/tty.usbserial-1421430")
board = Arduino("/dev/ttyUSB0")

led_pin = 13  # Пин для светодиода
relay_pin = 3  # Пин для реле

# Функция для определения наличия зелёного цвета
def detect_green(frame):
    # Преобразование в цветовое пространство HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Диапазон зелёного цвета в HSV
    lower_green = np.array([35, 40, 40])  # Нижняя граница
    upper_green = np.array([85, 255, 255])  # Верхняя граница

    # Создание маски для зелёного цвета
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Подсчёт количества зелёных пикселей
    green_pixels = cv2.countNonZero(mask)

    # Если зелёные пиксели превышают определённый порог, считаем, что зелёный цвет есть
    height, width = frame.shape[:2]
    total_pixels = height * width
    green_ratio = green_pixels / total_pixels

    return green_ratio > 0.0010, green_ratio  # Порог: 0.5% зелёных пикселей

# Функция для добавления текста с фоном на изображение
def draw_text_with_background(frame, text, position, font, scale, color, thickness, bg_color, alpha=0.5):
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    text_w, text_h = text_size
    x, y = position

    # Координаты для фона
    bg_x1, bg_y1 = x - 5, y - text_h - 5
    bg_x2, bg_y2 = x + text_w + 5, y + 5

    # Создаём полупрозрачный фон
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Добавляем текст
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)


# Запуск анализа видео из файла
def main():
    # video_path = "video2.MOV"  # Замените на путь к вашему видеофайлу
    # cap = cv2.VideoCapture(video_path)  # Открытие видеофайла
    #
    # if not cap.isOpened():
    #     print("Не удалось открыть видеофайл")
    #     return
    cap = cv2.VideoCapture(0)  # Открытие веб-камеры
    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return


    spray_active = False
    spray_end_time = 0


    while True:
        ret, frame = cap.read()
        if not ret:
            print("Видео закончилось")
            break

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

        # Отображение статуса форсунки
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
        # Показ кадра
        cv2.imshow("Green Color Detection", frame)

        # Выход по клавише 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
