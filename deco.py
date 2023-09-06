import serial


def cleanData(data):
    l = []
    temp = data[2:]
    l.append(temp[:-5])
    data = l[0]
    if (data != ''):
        return float(data)


try:
    arduino = serial.Serial("COM3", timeout=1, baudrate=9600)
except:
    print("Check port")

while 1:
    rawdata = str(arduino.readline())
    data = cleanData(rawdata)
    if (data != None):
        print(data)