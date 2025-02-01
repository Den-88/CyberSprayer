from pyfirmata2 import Arduino, util
import time

# board = Arduino("/dev/tty.usbserial-1421430")
board = Arduino("/dev/ttyUSB0")

led_pin = 13  # Пин для светодиода
relay_pin = 3  # Пин для светодиода

board.digital[led_pin].write(1)  # Включить
board.digital[relay_pin].write(1)  # Включить

time.sleep(1)
board.digital[led_pin].write(0)  # Выключить
board.digital[relay_pin].write(0)  # Выключить

time.sleep(1)
board.digital[led_pin].write(1)  # Включить
board.digital[relay_pin].write(1)  # Включить

time.sleep(1)
board.digital[led_pin].write(0)  # Выключить
board.digital[relay_pin].write(0)  # Выключить


board.exit()
