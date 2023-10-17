import serial
import main


def connect(port):
    try:
        arduino = serial.Serial(port, timeout=1, baudrate=9600)
    except:
        print("Check port")
    return arduino


def writeData(ard, data):
    ard.write(bytes(data,  'utf-8'))


def readData(ard):
    data = ard.readline()
    # print(data)
    return (str(data))


if __name__ == "__main__":
    inp = connect("COM7")
    # out = connect("COM5")
    
    while 1:        
        data = readData(inp)
        print(data)
        if '10' in data:
            # writeData(out,"1")
            print("here")
            main.main()
