import serial

ser = serial.Serial("COM3", baudrate = 9600, timeout = 1)

def spray():
    ser.write(b's')