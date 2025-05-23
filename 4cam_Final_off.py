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
RELAY_PIN_3 = 4                # Пин для реле 1 (левая часть)
RELAY_PIN_4 = 5                # Пин для реле 2 (правая часть)

# Минимальная площадь объекта для обнаружения (в пикселях)
MIN_OBJECT_AREA = 800 # 150 было

# Настройки RTSP
RTSP_URLS = [
    "rtsp://admin:user12345@192.168.1.201:8555/main",  # Камера 1
    "rtsp://admin:user12345@192.168.1.202:8555/main",  # Камера 2
    "rtsp://admin:user12345@192.168.1.203:8555/main",  # Камера 3
    "rtsp://admin:user12345@192.168.1.204:8555/main",  # Камера 4
]

RTSP_OUTPUT_PIPELINE = (
    "appsrc ! videoconvert ! video/x-raw,format=NV12 ! x264enc tune=zerolatency bitrate=5000 speed-preset=ultrafast key-int-max=30 "
    "! h264parse ! rtspclientsink location=rtsp://127.0.0.1:8554/test"
)

# Флаг для включения/отключения вывода изображения
ENABLE_OUTPUT = False  # По умолчанию вывод отключен

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


import queue

class FrameCaptureThread(threading.Thread):
    """Поток для захвата кадров из RTSP-потока."""
    def __init__(self, rtsp_url):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(rtsp_url)
        self.latest_frame = None  # Храним только последний кадр
        self.running = True
        self.lock = threading.Lock()  # Блокировка для потокобезопасности

    def run(self):
        """Основной цикл потока для захвата кадров."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame  # Сохраняем только последний кадр
        self.cap.release()

    def stop(self):
        """Остановка потока."""
        self.running = False
        self.join()

    def get_frame(self):
        """Получение последнего доступного кадра (не блокирующее)."""
        with self.lock:
            return self.latest_frame  # Просто возвращаем последний доступный кадр


def signal_handler(sig, frame):
    """Обработчик сигнала для корректного завершения программы."""
    print("Программа завершена.")
    global running
    running = False

    if capture_threads:
        for thread in capture_threads:
            thread.stop()  # Останавливаем потоки
    if ENABLE_OUTPUT and out:
        out.release()  # Закрываем видео-поток

    cv2.destroyAllWindows()  # Закрываем окна OpenCV
    sys.exit(0)  # Выход из программы


def resize_frame(frame, width, height):
    """Изменение размера кадра до заданных ширины и высоты."""
    return cv2.resize(frame, (width, height))


def merge_frames(frames):
    """Объединение кадров в один (горизонтально)."""
    height, width = frames[0].shape[:2]

    # Уменьшаем разрешение каждого кадра в 4 раза
    resized_frames = [resize_frame(frame, width // 4, height // 4) for frame in frames]

    # Объединяем все кадры по горизонтали
    merged_frame = np.hstack(resized_frames)  # Для горизонтальной стыковки

    return merged_frame

spray_active_left = [False] * 4
spray_end_time_left = [0] * 4
spray_active_right = [False] * 4
spray_end_time_right = [0] * 4

def process_frames(frames):
    num_parts = 6  # Количество частей кадра
    spray_active = [[False] * num_parts for _ in range(len(RTSP_URLS))]
    spray_end_time = [[0] * num_parts for _ in range(len(RTSP_URLS))]

    green_detected = [[False] * num_parts for _ in range(len(RTSP_URLS))]

    for i, frame in enumerate(frames):
        if frame is None:
            continue

        height, width = frame.shape[:2]
        part_width = width // num_parts
        current_time = time.time()

        for j in range(num_parts):
            x_start = j * part_width
            x_end = (j + 1) * part_width if j < num_parts - 1 else width
            part_frame = frame[:, x_start:x_end]

            # Анализируем часть кадра
            green_detected[i][j], contours = detect_green(part_frame, region=None)

            # Логика работы форсунки для каждой части
            if green_detected[i][j]:
                spray_active[i][j] = True
                spray_end_time[i][j] = current_time + 0.3
            elif current_time > spray_end_time[i][j]:
                spray_active[i][j] = False

            # Логирование
            print(f"Camera {i+1} Part {j+1} Detected: {green_detected[i][j]}, Spray: {spray_active[i][j]}")
            # Управление Arduino
            board.digital[LED_PIN].write(spray_active[0][4])  # Светодиод
            board.digital[RELAY_PIN_1].write(not spray_active[0][0])  # Реле 1 (левая часть)
            board.digital[RELAY_PIN_2].write(not spray_active[0][1])  # Реле 2 (правая часть)
            board.digital[RELAY_PIN_3].write(not spray_active[0][2])  # Реле 2 (правая часть)
            board.digital[RELAY_PIN_4].write(not spray_active[0][3])  # Реле 2 (правая часть)

            if ENABLE_OUTPUT:
                # Отрисовка контуров
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(part_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    area = cv2.contourArea(contour)
                    cv2.putText(part_frame, f"S = {area}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Рисуем 7 вертикальных белых линий для разделения на 6 частей
                height, width = frame.shape[:2]
                # Количество частей
                num_parts = 6
                # Расстояние между линиями
                line_positions = [int(i * width / num_parts) for i in range(1, num_parts)]
                # Добавляем линии с самого левого и правого края
                line_positions = [0] + line_positions + [width]

                # Рисуем линии
                for pos in line_positions:
                    cv2.line(frame, (pos, 0), (pos, height), (255, 255, 255), 2)

                # Добавляем нумерацию сверху
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 3.5
                font_thickness = 6
                text_color = (0, 0, 255)  # Белый цвет текста
                offset = 100  # Отступ сверху

                for j in range(num_parts):
                    # Позиция для текста (центр каждой части)
                    x_position = int((j * width / num_parts) + (width / num_parts / 2) - 10)
                    # Текст (номер)
                    cv2.putText(frame, str(i * 6 + j + 1), (x_position, offset), font, font_scale, text_color,
                                font_thickness)

                    # # Добавление текста на кадр
                    # draw_text_with_background(
                    #     frame,
                    #     f"Detected: {green_detected}",
                    #     (10, 30),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     1,
                    #     (0, 255, 0) if green_detected else (0, 0, 255),
                    #     2,
                    #     (0, 0, 0),
                    # )
                    # draw_text_with_background(
                    #     frame,
                    #     f"Spray: {spray_active[i][j]}",
                    #     (10, 70),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     1,
                    #     (0, 255, 0) if spray_active[i][j] else (0, 0, 255),
                    #     2,
                    #     (0, 0, 0),
                    # )
                    # Координаты кружков
                    circle1_center = (int(j * width / num_parts) + 50 , 50)  # Первый кружок
                    circle2_center = (int(j * width / num_parts) + 50, 100)  # Второй кружок
                    radius = 20  # Радиус кружков

                    # Цвета кружков
                    circle1_color = (0, 255, 0) if green_detected[i][j] else (0, 0, 255)  # Зеленый/красный
                    circle2_color = (255, 0, 0) if spray_active[i][j] else (0, 0, 255)  # Зеленый/красный

                    # Рисуем кружки
                    cv2.circle(frame, circle1_center, radius, circle1_color, -1)  # -1 делает круг залитым
                    cv2.circle(frame, circle2_center, radius, circle2_color, -1 if spray_active[i][j] else 0)

    if ENABLE_OUTPUT and out:
        # Объединяем кадры
        merged_frame = merge_frames(frames)
        out.write(merged_frame)


        # # Анализ левой половины кадра
        # green_detected_left, contours_left = detect_green(frame, region="left")
        # current_time = time.time()
        #
        # # Логика работы форсунки для левой части
        # if green_detected_left:
        #     spray_active_left[i] = True
        #     spray_end_time_left[i] = current_time + 0.3  # Таймер на 0.3 секунды
        # elif current_time > spray_end_time_left[i]:
        #     spray_active_left[i] = False
        #
        # # Анализ правой половины кадра
        # green_detected_right, contours_right = detect_green(frame, region="right")
        #
        # # Логика работы форсунки для правой части
        # if green_detected_right:
        #     spray_active_right[i] = True
        #     spray_end_time_right[i] = current_time + 0.3  # Таймер на 0.3 секунды
        # elif current_time > spray_end_time_right[i]:
        #     spray_active_right[i] = False

        # # Управление Arduino
        # board.digital[LED_PIN].write(green_detected_left or green_detected_right)  # Светодиод
        # board.digital[RELAY_PIN_1].write(not spray_active_left[i])  # Реле 1 (левая часть)
        # board.digital[RELAY_PIN_2].write(not spray_active_right[i])  # Реле 2 (правая часть)

        # Логирование
        # print(f"Camera {i + 1} Left Detected: {green_detected_left}, Spray: {spray_active_left[i]}")
        # print(f"Camera {i + 1} Right Detected: {green_detected_right}, Spray: {spray_active_right[i]}")

        # Отправка в RTSP вывод, если включено
        # if ENABLE_OUTPUT and out:
        #     # # Рисуем вертикальную белую линию посередине
        #     # height, width = frame.shape[:2]
        #     # cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)
        #     # Рисуем 7 вертикальных белых линий для разделения на 6 частей
        #     height, width = frame.shape[:2]
        #     # Количество частей
        #     num_parts = 6
        #     # Расстояние между линиями
        #     line_positions = [int(i * width / num_parts) for i in range(1, num_parts)]
        #     # Добавляем линии с самого левого и правого края
        #     line_positions = [0] + line_positions + [width]
        #
        #     # Рисуем линии
        #     for pos in line_positions:
        #         cv2.line(frame, (pos, 0), (pos, height), (255, 255, 255), 2)
        #
        #     # Добавляем нумерацию сверху
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     font_scale = 3.5
        #     font_thickness = 6
        #     text_color = (0, 0, 255)  # Белый цвет текста
        #     offset = 100  # Отступ сверху
        #
        #     for j in range(num_parts):
        #         # Позиция для текста (центр каждой части)
        #         x_position = int((j * width / num_parts) + (width / num_parts / 2) - 10)
        #         # Текст (номер)
        #         cv2.putText(frame, str(i * 6 + j + 1), (x_position, offset), font, font_scale, text_color,
        #                     font_thickness)
        #
        #     # # Обводка зеленых объектов на левой половине и отображение площади
        #     # for contour in contours_left:
        #     #     x, y, w, h = cv2.boundingRect(contour)
        #     #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #     #     area = cv2.contourArea(contour)
        #     #     cv2.putText(frame, f"S = {area}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        #     #
        #     # # Обводка зеленых объектов на правой половине и отображение площади
        #     # for contour in contours_right:
        #     #     x, y, w, h = cv2.boundingRect(contour)
        #     #     cv2.rectangle(frame, (x + width // 2, y), (x + width // 2 + w, y + h), (0, 255, 0), 2)
        #     #     area = cv2.contourArea(contour)
        #     #     cv2.putText(frame, f"S = {area}", (x + width // 2, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
        #     #                 2)
        #     #
        #     # # Добавление текста на кадр (левая часть)
        #     # draw_text_with_background(
        #     #     frame,
        #     #     f"Left Detected: {green_detected_left}",
        #     #     (10, 30),
        #     #     cv2.FONT_HERSHEY_SIMPLEX,
        #     #     1,
        #     #     (0, 255, 0) if green_detected_left else (0, 0, 255),
        #     #     2,
        #     #     (0, 0, 0),
        #     # )
        #     # draw_text_with_background(
        #     #     frame,
        #     #     f"Left Spray: {spray_active_left}",
        #     #     (10, 70),
        #     #     cv2.FONT_HERSHEY_SIMPLEX,
        #     #     1,
        #     #     (0, 255, 0) if spray_active_left else (0, 0, 255),
        #     #     2,
        #     #     (0, 0, 0),
        #     # )
        #     #
        #     # # Добавление текста на кадр (правая часть)
        #     # draw_text_with_background(
        #     #     frame,
        #     #     f"Right Detected: {green_detected_right}",
        #     #     (width // 2 + 10, 30),
        #     #     cv2.FONT_HERSHEY_SIMPLEX,
        #     #     1,
        #     #     (0, 255, 0) if green_detected_right else (0, 0, 255),
        #     #     2,
        #     #     (0, 0, 0),
        #     # )
        #     # draw_text_with_background(
        #     #     frame,
        #     #     f"Right Spray: {spray_active_right}",
        #     #     (width // 2 + 10, 70),
        #     #     cv2.FONT_HERSHEY_SIMPLEX,
        #     #     1,
        #     #     (0, 255, 0) if spray_active_right else (0, 0, 255),
        #     #     2,
        #     #     (0, 0, 0),
        #     # )



def main():
    """Основная функция программы."""
    global running, capture_threads, out

    # Запуск потоков захвата кадров для каждой камеры
    # capture_threads = []
    # for rtsp_url in RTSP_URLS:
    #     thread = FrameCaptureThread(rtsp_url)
    #     capture_threads.append(thread)
    #     thread.start()

    capture_threads = [FrameCaptureThread(rtsp) for rtsp in RTSP_URLS]
    for thread in capture_threads:
        thread.start()


    # Даем время потокам запуститься
    time.sleep(2)

    for thread in capture_threads:
        if not thread.cap.isOpened():
            print("Ошибка: невозможно открыть RTSP-поток. Завершаем программу.")
            thread.stop()
            sys.exit(1)

    # Инициализация RTSP-вывода, если вывод включен
    if ENABLE_OUTPUT:
        out = cv2.VideoWriter(RTSP_OUTPUT_PIPELINE, cv2.CAP_GSTREAMER, 0, 25, (10240 // 4, 1440 // 4), True)

    # Основной цикл обработки кадров
    running = True
    # spray_active_left = [False] * 4
    # spray_end_time_left = [0] * 4
    # spray_active_right = [False] * 4
    # spray_end_time_right = [0] * 4


    frame_interval = 1.0 / 25  # Интервал между кадрами для 25 FPS
    last_processed_time = time.time()  # Время последней успешной обработки

    while running:
        # start_time = time.time()
        #
        # # Получаем кадры из каждого потока
        # frames = [thread.get_frame() for thread in capture_threads]
        # for i, frame in enumerate(frames):
        #     if frame is None:
        #         continue

        current_time = time.time()
        # print(f'current_time - last_processed_time {current_time - last_processed_time} frame_interval {frame_interval}')
        # if current_time - last_processed_time < frame_interval:
        #     continue  # Ждём нужный интервал перед обработкой следующего кадра

        frames = [thread.get_frame() for thread in capture_threads]
        frames = [f for f in frames if f is not None]  # Фильтруем пустые кадры

        if frames:
            process_frames(frames)  # Функция обработки

        last_processed_time = time.time()  # Обновляем таймер


        # for i, frame in enumerate(frames):
        #     if frame is None:
        #         continue
        #
        #     # Анализ левой половины кадра
        #     green_detected_left, contours_left = detect_green(frame, region="left")
        #     current_time = time.time()
        #
        #     # Логика работы форсунки для левой части
        #     if green_detected_left:
        #         spray_active_left[i] = True
        #         spray_end_time_left[i] = current_time + 0.3  # Таймер на 0.3 секунды
        #     elif current_time > spray_end_time_left[i]:
        #         spray_active_left[i] = False
        #
        #     # Анализ правой половины кадра
        #     green_detected_right, contours_right = detect_green(frame, region="right")
        #
        #     # Логика работы форсунки для правой части
        #     if green_detected_right:
        #         spray_active_right[i] = True
        #         spray_end_time_right[i] = current_time + 0.3  # Таймер на 0.3 секунды
        #     elif current_time > spray_end_time_right[i]:
        #         spray_active_right[i] = False
        #
        #     # # Управление Arduino
        #     # board.digital[LED_PIN].write(green_detected_left or green_detected_right)  # Светодиод
        #     # board.digital[RELAY_PIN_1].write(not spray_active_left[i])  # Реле 1 (левая часть)
        #     # board.digital[RELAY_PIN_2].write(not spray_active_right[i])  # Реле 2 (правая часть)
        #
        #     # Логирование
        #     print(f"Camera {i+1} Left Detected: {green_detected_left}, Spray: {spray_active_left[i]}")
        #     print(f"Camera {i+1} Right Detected: {green_detected_right}, Spray: {spray_active_right[i]}")
        #
        #     # Отправка в RTSP вывод, если включено
        #     if ENABLE_OUTPUT and out:
        #         # # Рисуем вертикальную белую линию посередине
        #         # height, width = frame.shape[:2]
        #         # cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 255, 255), 2)
        #         # Рисуем 7 вертикальных белых линий для разделения на 6 частей
        #         height, width = frame.shape[:2]
        #         # Количество частей
        #         num_parts = 6
        #         # Расстояние между линиями
        #         line_positions = [int(i * width / num_parts) for i in range(1, num_parts)]
        #         # Добавляем линии с самого левого и правого края
        #         line_positions = [0] + line_positions + [width]
        #
        #         # Рисуем линии
        #         for pos in line_positions:
        #             cv2.line(frame, (pos, 0), (pos, height), (255, 255, 255), 2)
        #
        #         # Добавляем нумерацию сверху
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         font_scale = 3.5
        #         font_thickness = 6
        #         text_color = (0, 0, 255)  # Белый цвет текста
        #         offset = 100  # Отступ сверху
        #
        #         for j in range(num_parts):
        #             # Позиция для текста (центр каждой части)
        #             x_position = int((j * width / num_parts) + (width / num_parts / 2) - 10)
        #             # Текст (номер)
        #             cv2.putText(frame, str(i * 6 + j + 1), (x_position, offset), font, font_scale, text_color, font_thickness)
        #
        #         # Обводка зеленых объектов на левой половине и отображение площади
        #         for contour in contours_left:
        #             x, y, w, h = cv2.boundingRect(contour)
        #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #             area = cv2.contourArea(contour)
        #             cv2.putText(frame, f"S = {area}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        #
        #         # Обводка зеленых объектов на правой половине и отображение площади
        #         for contour in contours_right:
        #             x, y, w, h = cv2.boundingRect(contour)
        #             cv2.rectangle(frame, (x + width // 2, y), (x + width // 2 + w, y + h), (0, 255, 0), 2)
        #             area = cv2.contourArea(contour)
        #             cv2.putText(frame, f"S = {area}", (x + width // 2, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
        #                         2)
        #
        #         # Добавление текста на кадр (левая часть)
        #         draw_text_with_background(
        #             frame,
        #             f"Left Detected: {green_detected_left}",
        #             (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             1,
        #             (0, 255, 0) if green_detected_left else (0, 0, 255),
        #             2,
        #             (0, 0, 0),
        #         )
        #         draw_text_with_background(
        #             frame,
        #             f"Left Spray: {spray_active_left}",
        #             (10, 70),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             1,
        #             (0, 255, 0) if spray_active_left else (0, 0, 255),
        #             2,
        #             (0, 0, 0),
        #         )
        #
        #         # Добавление текста на кадр (правая часть)
        #         draw_text_with_background(
        #             frame,
        #             f"Right Detected: {green_detected_right}",
        #             (width // 2 + 10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             1,
        #             (0, 255, 0) if green_detected_right else (0, 0, 255),
        #             2,
        #             (0, 0, 0),
        #         )
        #         draw_text_with_background(
        #             frame,
        #             f"Right Spray: {spray_active_right}",
        #             (width // 2 + 10, 70),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             1,
        #             (0, 255, 0) if spray_active_right else (0, 0, 255),
        #             2,
        #             (0, 0, 0),
        #         )
        #
        # if ENABLE_OUTPUT and out:
        # # Объединяем кадры
        #     merged_frame = merge_frames(frames)
        #     out.write(merged_frame)

        # Логирование времени обработки кадра
        # end_time = time.time()
        # last_processed_time = time.time()  # Обновляем время последней успешной обработки

        # print(f"Frame processed in {end_time - start_time:.4f} seconds")
        print(f"Frame processed in {last_processed_time - current_time:.4f} seconds")

    # Завершение работы
    for thread in capture_threads:
        thread.stop()
    if ENABLE_OUTPUT:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Обработчик сигнала SIGINT для корректного завершения программы
    signal.signal(signal.SIGINT, signal_handler)
    main()
