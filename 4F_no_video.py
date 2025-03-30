import concurrent
import os
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
MIN_GREEN_PIXELS = 1400

# Настройки RTSP
RTSP_URLS = [
    "rtsp://admin:user12345@192.168.1.201:8555/main",  # Камера 1
    "rtsp://admin:user12345@192.168.1.202:8555/main",  # Камера 2
    "rtsp://admin:user12345@192.168.1.203:8555/main",  # Камера 3
    "rtsp://admin:user12345@192.168.1.204:8555/main",  # Камера 4
]

num_parts = 6  # Количество частей кадра
spray_active = [[False] * num_parts for _ in range(len(RTSP_URLS))]
spray_end_time = [[0] * num_parts for _ in range(len(RTSP_URLS))]
green_detected = [[False] * num_parts for _ in range(len(RTSP_URLS))]

# RTSP_OUTPUT_PIPELINE = (
#     "appsrc ! videoconvert ! video/x-raw,format=NV12 ! x264enc tune=zerolatency bitrate=5000 speed-preset=ultrafast key-int-max=30 "
#     "! h264parse ! rtspclientsink location=rtsp://127.0.0.1:8554/test"
# )
RTSP_OUTPUT_PIPELINE = (
    "appsrc ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 ! "
    "videoconvert ! video/x-raw,format=NV12 ! "
    "x264enc tune=zerolatency bitrate=5000 speed-preset=ultrafast key-int-max=30 "
    "! h264parse ! rtspclientsink location=rtsp://127.0.0.1:8554/test latency=0"
)

# Флаг для включения/отключения вывода изображения
ENABLE_OUTPUT = False  # По умолчанию вывод отключен

# Инициализация Arduino
board = Arduino(ARDUINO_PORT)

def clear_screen():
    """Очистка экрана перед выводом обновленных данных."""
    os.system('cls' if os.name == 'nt' else 'clear')  # Windows или Unix/Linux
    init_display()

# status_line = [(False, False)] * (len(RTSP_URLS) * num_parts)
# Цвета для вывода
GREEN = '\033[32m'  # Зеленый
RED = '\033[31m'    # Красный
RESET = '\033[0m'   # Сброс цвета

# Инициализация статусов
status_line = [(False, False, 0) for _ in range(len(RTSP_URLS) * num_parts)]

# Кэш позиций курсора для каждой строки
line_positions = {}


def init_display():
    """Инициализация дисплея с таблицей и запоминанием позиций строк."""
    sys.stdout.write("\033[?25l")  # Скрыть курсор
    headers = "Форсунка    | Камера             | Зелёный обнаружен? | Форсунка включена? | Время обработки кадра "
    print(headers)
    print("-" * len(headers))

    # Начальная позиция после заголовков (3 строка)
    current_line = 3

    for i in range(len(RTSP_URLS)):
        for j in range(num_parts):
            index = i * num_parts + j
            line_positions[index] = current_line
            current_line += 1

            nozzle_number = (i * num_parts) + (j + 1)
            nozzle_number_str = f"{nozzle_number} " if nozzle_number < 10 else str(nozzle_number)

            print(
                f"Форсунка {nozzle_number_str} | Камера {i + 1: <2} Часть {j + 1: <2} | {'НЕТ': <6}             | {'ВЫКЛ': <6}")
    sys.stdout.flush()


def update_status(i, j, detected, active, time):
    """Обновляет только нужные статусы без перерисовки всей таблицы."""
    index = i * num_parts + j
    if index >= len(status_line):
        return

    # Проверяем, изменились ли значения
    old_detected, old_active, old_time = status_line[index]
    if detected == old_detected and active == old_active and old_time == time:
        return

    status_line[index] = (detected, active, time)

    # Получаем номер строки для этого элемента
    line_num = line_positions.get(index)
    if line_num is None:
        return

    # Подготавливаем новые значения
    green_status = "ДА " if detected else "НЕТ"
    green_color = GREEN if detected else RED
    spray_status = "ВКЛ " if active else "ВЫКЛ"
    spray_color = GREEN if active else RED

    # Формируем строку для обновления
    nozzle_number = (i * num_parts) + (j + 1)
    nozzle_number_str = f"{nozzle_number} " if nozzle_number < 10 else str(nozzle_number)
    new_line = (
        f"Форсунка {nozzle_number_str} | Камера {i + 1: <2} Часть {j + 1: <2} | "
        f"{green_color}{green_status: <6}{RESET}             | "
        f"{spray_color}{spray_status: <6}{RESET}             | "
        f"{time}"
    )

    # Перемещаем курсор и обновляем строку
    sys.stdout.write(f"\033[{line_num}H")  # Перемещение к нужной строке
    sys.stdout.write(new_line)
    sys.stdout.write("\033[0K")  # Очистка до конца строки
    sys.stdout.flush()


def detect_green(frame):
    """Обнаружение зеленого цвета на кадре или его части."""

    # return False, []
    if frame is None:
        return False, []  # Если кадра нет, ничего не делать

    # Уменьшаем размер кадра (ускоряет обработку)
    # frame = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4))

    # Преобразуем кадр в HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower_green = np.array([35, 30, 40])  # Нижняя граница зеленого
    # upper_green = np.array([85, 255, 255])  # Верхняя граница зеленого
    lower_green = np.array([35, 30, 40], dtype=np.uint8)
    upper_green = np.array([85, 255, 255], dtype=np.uint8)

    # Создаем маску для зеленого цвета
    mask = cv2.inRange(hsv, lower_green, upper_green)

    if ENABLE_OUTPUT:
        # Находим контуры зеленых объектов
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Фильтруем контуры по минимальной площади
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_OBJECT_AREA]
        # Возвращаем результат и отфильтрованные контуры
        return len(filtered_contours) > 0, filtered_contours
    else:
        # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # # Проверяем, есть ли хотя бы один объект с площадью > MIN_OBJECT_AREA
        # for i in range(1, num_labels):  # Начинаем с 1, так как 0 – это фон
        #     if stats[i, cv2.CC_STAT_AREA] > MIN_OBJECT_AREA:
        #         return True, []  # Если найден хотя бы один объект, сразу возвращаем True
        # return False, []

        # # Проверяем количество зеленых пикселей
        # if np.count_nonzero(mask) > MIN_GREEN_PIXELS:
        # # if cv2.countNonZero(mask) > MIN_GREEN_PIXELS:
        #     return True, []
        # else:
        #     return False, []

        # Преобразуем маску в одномерный массив и проверяем частями
        flat_mask = mask.ravel()
        # Используем быструю итерацию с ранним выходом
        green_count = 0
        for i in range(0, len(flat_mask), 10000):  # Читаем по 1000 пикселей за раз
            green_count += np.count_nonzero(flat_mask[i:i + 10000])
            if green_count >= MIN_GREEN_PIXELS:
                return True, []  # Достигли порога, выходим сразу
        return False, []  # Недостаточно зелёного


class FrameCaptureThread(threading.Thread):
    """Поток для захвата кадров из RTSP-потока."""
    def __init__(self, rtsp_url, sleep_interval=0):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(rtsp_url)
        self.latest_frame = None  # Храним только последний кадр
        self.running = True
        self.sleep_interval = sleep_interval  # Интервал для слипа (в секундах)

    def run(self):
        """Основной цикл потока для захвата кадров."""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame  # Сохраняем только последний кадр
            time.sleep(self.sleep_interval)  # Добавляем задержку, чтобы не перегружать процессор
        self.cap.release()

    def stop(self):
        """Остановка потока."""
        self.running = False
        self.join()

    def get_frame(self):
        """Получение последнего доступного кадра (не блокирующее)."""
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

# def process_frames(frames):
#     global num_parts, spray_active, spray_end_time, green_detected  # Указываем, что это глобальная переменная
#
#     for i, frame in enumerate(frames):
#         if frame is None:
#             print("frame is None")
#             continue
#
#         height, width = frame.shape[:2]
#         part_width = width // num_parts
#         current_time = time.time()
#
#         for j in range(num_parts):
#             x_start = j * part_width
#             x_end = (j + 1) * part_width if j < num_parts - 1 else width
#             part_frame = frame[:, x_start:x_end]
#
#             # Анализируем часть кадра
#             green_detected[i][j], contours = detect_green(part_frame)
#
#             # Логика работы форсунки для каждой части
#             if green_detected[i][j]:
#                 spray_active[i][j] = True
#                 spray_end_time[i][j] = current_time + 0.3
#             elif current_time > spray_end_time[i][j]:
#                 spray_active[i][j] = False
#
#             # Логирование
#             print(f"Camera {i+1} Part {j+1} Detected: {green_detected[i][j]}, Spray: {spray_active[i][j]}")
#             # Управление Arduino
#             board.digital[LED_PIN].write(spray_active[0][4])  # Светодиод
#             board.digital[RELAY_PIN_1].write(not spray_active[0][0])  # Реле 1 (левая часть)
#             board.digital[RELAY_PIN_2].write(not spray_active[0][1])  # Реле 2 (правая часть)
#             board.digital[RELAY_PIN_3].write(not spray_active[0][2])  # Реле 2 (правая часть)
#             board.digital[RELAY_PIN_4].write(not spray_active[0][3])  # Реле 2 (правая часть)
#
#             if ENABLE_OUTPUT:
#                 # Отрисовка контуров
#                 for contour in contours:
#                     x, y, w, h = cv2.boundingRect(contour)
#                     cv2.rectangle(part_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                     area = cv2.contourArea(contour)
#                     cv2.putText(part_frame, f"S = {area}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#
#                 # Рисуем 7 вертикальных белых линий для разделения на 6 частей
#                 height, width = frame.shape[:2]
#                 # Количество частей
#                 num_parts = 6
#                 # Расстояние между линиями
#                 line_positions = [int(i * width / num_parts) for i in range(1, num_parts)]
#                 # Добавляем линии с самого левого и правого края
#                 line_positions = [0] + line_positions + [width]
#
#                 # Рисуем линии
#                 for pos in line_positions:
#                     cv2.line(frame, (pos, 0), (pos, height), (255, 255, 255), 2)
#
#                 # Добавляем нумерацию сверху
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 font_scale = 3.5
#                 font_thickness = 6
#                 text_color = (0, 0, 255)  # Белый цвет текста
#                 offset = 100  # Отступ сверху
#
#                 for j in range(num_parts):
#                     # Позиция для текста (центр каждой части)
#                     x_position = int((j * width / num_parts) + (width / num_parts / 2) - 10)
#                     # Текст (номер)
#                     cv2.putText(frame, str(i * 6 + j + 1), (x_position, offset), font, font_scale, text_color,
#                                 font_thickness)
#
#                     # Координаты кружков
#                     circle1_center = (int(j * width / num_parts) + 50 , 50)  # Первый кружок
#                     circle2_center = (int(j * width / num_parts) + 50, 100)  # Второй кружок
#                     radius = 20  # Радиус кружков
#
#                     # Цвета кружков
#                     circle1_color = (0, 255, 0) if green_detected[i][j] else (0, 0, 255)  # Зеленый/красный
#                     circle2_color = (255, 0, 0) if spray_active[i][j] else (0, 0, 255)  # Зеленый/красный
#
#                     # Рисуем кружки
#                     cv2.circle(frame, circle1_center, radius, circle1_color, -1)  # -1 делает круг залитым
#                     cv2.circle(frame, circle2_center, radius, circle2_color, -1 if spray_active[i][j] else 0)

def process_frame(i, frame, start_time):
    global num_parts, spray_active, spray_end_time, green_detected  # Указываем, что это глобальная переменная

    # for i, frame in enumerate(frames):
    if frame is None:
        print("frame is None")
        return None

    height, width = frame.shape[:2]
    part_width = width // num_parts
    current_time = time.time()

    if ENABLE_OUTPUT:
        # Создаем копию кадра для рисования
        frame_copy = frame.copy()

    for j in range(num_parts):
        x_start = j * part_width
        x_end = (j + 1) * part_width if j < num_parts - 1 else width
        part_frame = frame[:, x_start:x_end]

        # Анализируем часть кадра
        green_detected[i][j], contours = detect_green(part_frame)

        # Логика работы форсунки для каждой части
        if green_detected[i][j]:
            spray_active[i][j] = True
            spray_end_time[i][j] = current_time + 0.3
        elif current_time > spray_end_time[i][j]:
            spray_active[i][j] = False

        # Логирование
        # print(f"Camera {i+1} Part {j+1} Detected: {green_detected[i][j]}, Spray: {spray_active[i][j]}")
        finish_time = time.time()
        processing_time = f"{finish_time - start_time:.4f} сек."
        update_status(i, j, green_detected[i][j], spray_active[i][j], processing_time)

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
                cv2.rectangle(frame_copy, (x_start + x, y), (x_start + x + w, y + h), (0, 255, 0), 2)
                area = cv2.contourArea(contour)
                cv2.putText(frame_copy, f"S = {area}", (x_start + x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Рисуем 7 вертикальных белых линий для разделения на 6 частей
            height, width = frame_copy.shape[:2]
            # Расстояние между линиями
            line_positions = [int(i * width / num_parts) for i in range(1, num_parts)]
            # Добавляем линии с самого левого и правого края
            line_positions = [0] + line_positions + [width]

            # Рисуем линии
            for pos in line_positions:
                cv2.line(frame_copy, (pos, 0), (pos, height), (255, 255, 255), 2)

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
                cv2.putText(frame_copy, str(i * 6 + j + 1), (x_position, offset), font, font_scale, text_color,
                            font_thickness)

                # Координаты кружков
                circle1_center = (int(j * width / num_parts) + 50 , 50)  # Первый кружок
                circle2_center = (int(j * width / num_parts) + 50, 100)  # Второй кружок
                radius = 20  # Радиус кружков

                # Цвета кружков
                circle1_color = (0, 255, 0) if green_detected[i][j] else (0, 0, 255)  # Зеленый/красный
                circle2_color = (255, 0, 0) if spray_active[i][j] else (0, 0, 255)  # Зеленый/красный

                # Рисуем кружки
                cv2.circle(frame_copy, circle1_center, radius, circle1_color, -1)  # -1 делает круг залитым
                cv2.circle(frame_copy, circle2_center, radius, circle2_color, -1 if spray_active[i][j] else 0)

    if ENABLE_OUTPUT:
        return frame_copy
    return None


def main():
    clear_screen()
    """Основная функция программы."""
    global running, capture_threads, out

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
    previous_frames = [None] * len(capture_threads)  # Сохраняем предыдущие кадры

    while running:
        start_time = time.time()

        if ENABLE_OUTPUT and out:
            frames = []
            for i, thread in enumerate(capture_threads):
                frame = thread.get_frame()

                # Проверяем, изменился ли кадр
                if previous_frames[i] is None or not np.array_equal(previous_frames[i], frame):
                    previous_frames[i] = frame.copy()  # Обновляем предыдущий кадр
                    frames.append(process_frame(i, frame, start_time))  # Обрабатываем только измененные

                else:
                    # print("Кадр не поменялся!")
                    frames.append(previous_frames[i])  # Если кадр не изменился, используем старый

            # Объединяем кадры
            merged_frame = merge_frames(frames)
            out.write(merged_frame)

        else:
            for i, thread in enumerate(capture_threads):
                frame = thread.get_frame()
                process_frame(i, frame, start_time)

                # if previous_frames[i] is None or not np.array_equal(previous_frames[i], frame):
                #     previous_frames[i] = frame.copy()
                #     process_frame(i, frame, start_time)

        last_processed_time = time.time()  # Обновляем таймер
        # print(f"Frame processed in {last_processed_time - current_time:.4f} seconds")

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
