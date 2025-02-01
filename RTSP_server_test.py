import cv2
import gi
import numpy as np

gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Инициализация GStreamer
Gst.init(None)

rtsp_url = "rtsp://192.168.0.17:8554/profile0"
udp_host = "127.0.0.1"
udp_port = 5000

# Открываем RTSP поток
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("❌ Error: Couldn't open RTSP stream")
    exit()

# GStreamer pipeline
pipeline_desc = f"""
    appsrc name=mysource format=GST_FORMAT_TIME is-live=true block=true do-timestamp=true
    ! videoconvert
    ! video/x-raw,format=I420
    ! x264enc bitrate=500 speed-preset=ultrafast tune=zerolatency
    ! rtph264pay config-interval=1 pt=96
    ! udpsink host={udp_host} port={udp_port}
"""
pipeline = Gst.parse_launch(pipeline_desc)
appsrc = pipeline.get_by_name("mysource")

# Запускаем GStreamer pipeline
pipeline.set_state(Gst.State.PLAYING)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Couldn't read frame")
        break

    # OpenCV читает кадр в BGR → конвертируем в I420
    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)

    # Преобразуем numpy array в Gst.Buffer
    buffer = Gst.Buffer.new_allocate(None, frame_yuv.nbytes, None)
    buffer.fill(0, frame_yuv.tobytes())
    buffer.pts = buffer.dts = Gst.util_uint64_scale(Gst.CLOCK_TIME_NONE, 1, 30)  # FPS = 30

    # Передаём в GStreamer
    appsrc.emit("push-buffer", buffer)

cap.release()
pipeline.set_state(Gst.State.NULL)
