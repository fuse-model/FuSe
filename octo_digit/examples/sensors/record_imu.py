import adafruit_bno055

import serial

uart = serial.Serial("/dev/tty.usbserial-120")

sensor = adafruit_bno055.BNO055_UART(uart)

while True:
    print("Temperature: {} degrees C".format(sensor.temperature))
    print("Accelerometer (m/s^2): {}".format(sensor.acceleration))
    print("Magnetometer (microteslas): {}".format(sensor.magnetic))
    print("Gyroscope (rad/sec): {}".format(sensor.gyro))
    print()