import cv2
import numpy as np

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

    return green_ratio > 0.05, green_ratio  # Порог: 5% зелёных пикселей

# Запуск камеры
def main():
    cap = cv2.VideoCapture(0)  # Открытие веб-камеры
    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр")
            break

        green_detected, green_ratio = detect_green(frame)

        # Отображение результата
        if green_detected:
            cv2.putText(
                frame,
                f"Green detected ({green_ratio*100:.2f}%)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame, "No green detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
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
