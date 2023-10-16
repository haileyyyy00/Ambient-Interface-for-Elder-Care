import serial, time

flag = 0

def cleanData(data):
    l = []
    temp = data[2:]
    l.append(temp[:-5])
    data = l[0]
    if (data != ''):
        return float(data)

def writeData(data):
    arduino.write(data)
    flag = 1


try:
    arduino = serial.Serial("COM6", timeout=1, baudrate=9600)
except:
    print("Check port")

# while 1:
#     writeData(1)
#     rawdata = str(arduino.readline())
#     data = cleanData(rawdata)
#     if (data != None):
#         print(data)

def write_read(x):
    arduino.write(bytes(x,  'utf-8'))
    time.sleep(0.05)
    data = arduino.readline()
    return data


while True:
    num = input()
    value  = write_read(num)
    print(type(value))