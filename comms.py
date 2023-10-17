import serial
import fall


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
    
    while 1:        
        writeData(inp,"2")
        data = readData(inp)
        if '10' in data:
            print(5)
        
