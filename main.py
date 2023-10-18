import serial
import camera, multiprocessing as mp


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
    return (str(data))

def test():
    while 1:        
        print(1)

def test2():
    camera.main()

if __name__ == "__main__":
    # inp = connect("COM7")
    # out = connect("COM5")

    p1 = mp.Process(target = test2)
    p2 = mp.Process(target = test)
    p1.start()
    p2.start()
    # while 1:        
    #     print(1)
        # data = readData(inp)
        # print(data)

        # if 1 in data: # Heat Detection
        #     writeData(out,"2")