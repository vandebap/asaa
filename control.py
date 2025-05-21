# control.py

import serial
from time import sleep

class MotorController:
    def __init__(self, port="/dev/ttyACM0", baudrate=57600):
        self.arduino = serial.Serial(port, baudrate, timeout=1)
        sleep(0.1)
        if self.arduino.isOpen():
            print(f"{port} connecté.")
        else:
            raise Exception("Échec de connexion à l'Arduino")

    def send_command(self, x, shoot):
        message = f"{x:03d}{shoot}"
        self.arduino.flush()
        self.arduino.flushOutput()
        self.arduino.write(message.encode())

    def close(self):
        self.arduino.close()

