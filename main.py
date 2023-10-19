import time
from playsound import playsound
import serial
import camera, multiprocessing as mp


def connect(port):
    """
    Used to connect to serial port

    Args:
        port ([string]): [port connected to arduino microprocesser]

    Returns:
        [Serial]: [Serial Port object]
    """
    try:
        arduino = serial.Serial(port, timeout=1, baudrate=9600)
    except:
        print("Check port")
    return arduino


def writeData(ard, data):
    """Write data to an serial monitor .

    Args:
        ard ([Serial]): [Serial object for microprocesser to be written data]
        data ([type]): [Data to be written]
    """
    ard.write(bytes(data, "utf-8"))


def readData(ard):
    """Read data from a serial monitor .

    Args:
        ard ([Serial]): [Serial object for microprocesser from which to read data]

    Returns:
        [String]: [Data that has been read, enclosed in b'']
    """
    data = ard.readline()
    return str(data)


def connection(inp, out):
    """Performs communication between microprocessers

    Args:
        inp ([Serial]): [Serial object of microprocesser to collect data from]
        out ([Serial]): [Serial object of microprocesser to show output]
    """
    p2 = mp.Process(target=startCam)

    while 1:

        data = readData(inp)

        # ----------- Manual Activations -------------------
        # data = str(input("Enter interaction number you want to see: "))
        # --------------------------------------------------
        if "1" in data:  # Ambient Light
            playsound("./Audio/AmbientLight.mp3")
        if "2" in data:  # Abnormal Temperature
            writeData(out, "4")
        if "3" in data:  # Touch Detected
            playsound("./Audio/TouchSensor.mp3")
        if "4" in data:  # Pressure Detected
            playsound("./Audio/PressureSensor.mp3")
        if "5" in data:  # Heart, SpO2 monitoring started
            writeData(out, "2")
            time.sleep(12)
        if "6" in data:
            writeData(out, "3")  # Irregular Heartbeat
        if "7" in data:
            writeData(out, "4")  # Irregular SpO2


# ----------- Manual Triggers -------------------
#  - MAX3010
# writeData(in,"1")
# p2.start()
# --------------------------------------------------


def startCam():
    """Start camera ."""
    camera.main()


if __name__ == "__main__":

    inp = connect("COM8")
    out = connect("COM7")
    connection(inp, out)