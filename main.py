import time, playsound
import serial
import Camera.camera, multiprocessing as mp


def connect(port):
    try:
        arduino = serial.Serial(port, timeout=1, baudrate=9600)
    except:
        print("Check port")
    return arduino


def writeData(ard, data):
    ard.write(bytes(data, "utf-8"))


def readData(ard):
    data = ard.readline()
    return str(data)


def connection(inp, out):

    p2 = mp.Process(target=startCam)
    

    while 1:

        camera = p2.start()
        data = readData(inp)        

        # ----------- Manual Activations -------------------
        # data = str(input("Enter interaction number you want to see: "))
        # --------------------------------------------------

        if 1 in data:  # Ambient Light
            playsound("/Audio/AmbientLight.mp3")
        if 2 in data:  # Abnormal Temperature
            writeData(out, "1")
        if 3 in data:  # Touch Detected
            playsound("/Audio/TouchSensor.mp3")
        if 4 in data:  # Pressure Detected
            playsound("/Audio/PressureSensor.mp3")
        if 5 in data:  # Heart, SpO2 monitoring started
            writeData(out, "2")
            time.sleep(12)
        if 6 in data:
            writeData(out, "3")  # Irregular Heartbeat
        if 7 in data:
            writeData(out, "4")  # Irregular SpO2

        if camera == 1:
            playsound("/Audio/FallDetection.mp3")
        elif camera == 2:
            playsound("/Audio/Happy.mp3")
        elif camera == 3:
            playsound("/Audio/Sad.mp3")


        p2.stop()


# ----------- Manual Triggers -------------------
#  - MAX3010
# writeData(in,"1")
# cam = p2.start()
# 
# --------------------------------------------------


def startCam():
    Camera.main()


if __name__ == "__main__":

    inp = connect("COM7")
    out = connect("COM5")

    p1 = mp.Process(target=connection, args=(inp, out))    
    p1.start()