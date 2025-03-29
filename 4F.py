import asyncio
import cv2
import numpy as np
import time

RTSP_URLS = [
    "rtsp://admin:user12345@192.168.1.201:8555/main",  # Камера 1
    "rtsp://admin:user12345@192.168.1.202:8555/main",  # Камера 2
]


async def capture_frame(rtsp_url):
    """Асинхронный захват кадра с RTSP-потока."""
    cap = cv2.VideoCapture(rtsp_url)
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Ошибка при захвате кадра с {rtsp_url}")
            break
        yield frame
        await asyncio.sleep(0.05)  # Небольшая задержка для уменьшения нагрузки на процессор


async def process_frame(frame):
    """Обработка кадра (обнаружение зеленого цвета)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 30, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 800:
            print("Обнаружен зеленый объект!")
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame


async def main():
    print("Запуск программы...")
    tasks = [capture_frame(rtsp_url) for rtsp_url in RTSP_URLS]
    frames = [asyncio.create_task(task.__anext__()) for task in tasks]
    while True:
        done, pending = await asyncio.wait(frames, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            frame = task.result()
            frame = await process_frame(frame)
            cv2.imshow("Processed Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        for task in pending:
            task.cancel()
        frames = [asyncio.create_task(task.__anext__()) for task in tasks]


if __name__ == "__main__":
    asyncio.run(main())
