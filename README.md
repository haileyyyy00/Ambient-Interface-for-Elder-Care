# <p align= "center">Ambient Interface for Elder Care 

## Project Description
The project focuses on creating an ambient interface to support the elderly without intruding on their daily lives. It consists of two components: an in-house tracking system for non-invasive data collection, and an external display for caretakers. By using Arduino-based wearables, IoT technology, and non-invasive audio-visual signals, we ensure seamless monitoring of elderly individuals' health while respecting their privacy. This innovative approach aims to improve elderly care without compromising their independence and comfort.

## Dependencies
* Playsound
* PySerial
* TensorFlow
* CV2
* Keras
* Numpy
* Mediapipe

## Table of Contents
* Sensors and Tools
* [Input](https://github.com/kushagra1912/Ambient-Interface-for-Elder-Care/blob/main/README.md#input-component)
* [Output](https://github.com/kushagra1912/Ambient-Interface-for-Elder-Care/edit/main/README.md#output-component)
* [Camera](https://github.com/kushagra1912/Ambient-Interface-for-Elder-Care/edit/main/README.md#camera-file)
* [Main](https://github.com/kushagra1912/Ambient-Interface-for-Elder-Care/edit/main/README.md#main)

## Input Component
In the directory Data You can access a file named [Data.ino](https://github.com/kushagra1912/Ambient-Interface-for-Elder-Care/blob/main/Data/Data.ino) which essentially is the input file for our project. Due to the conventions from Arduino IDE we had to name the file same as the directory name. <br>
- <details>
  <summary> Accessing and Running the File: </summary>
  - Data > Data.ino > Arduino IDE > Connect Arduino with laptop > Run the file <br>
</details>

- This code is designed for our Arduino Uno-based hardware project. Once uploaded to the Arduino using the Arduino IDE, the hardware collects data non-intrusively for processing. The code includes functions for checking temperature and measuring vital signs like SPO2 (blood oxygen saturation) and heart rate using a MAX30105 sensor, pressure sensor and a Photodiode for light. The sensor data is communicated over a serial connection, and the hardware can be triggered to measure vital signs by sending '1' through the serial monitor. The code is set up to provide accurate and non-invasive monitoring for a health-related project.

<details> 
  <summary> Here's a breakdown of the code and the process: </summary>
    1. In the <b> setup() function </b>, it sets up serial communication, pin modes for LEDs, and initializes the MAX30105 sensor with specific configurations. <br>
    2. In the <b> loop() function </b>, the code listens for input from the serial monitor. If it receives '1', it triggers the checkSPO2() function. <br>
    3. The <b> checkHeat() function </b> reads an analog temperature sensor (connected to A0), calculates the temperature in degrees Celsius, and prints '1' to the serial  monitor if the temperature exceeds 20°C. <br>
    4. The <b> checkSPO2() function </b> measures SPO2 and heart rate using the **MAX30105 sensor**. It collects data over 100 samples, calculates important metrics, and reports the results to the serial monitor. <br>
    5. The calculated SPO2 and heart rate values are then processed and displayed for monitoring and analysis.<br>
</details>

## Output Component
In the directory Data You can access a file named [Interactions.ino](https://github.com/kushagra1912/Ambient-Interface-for-Elder-Care/blob/main/Data/Interactions.ino) which essentially is the output file for our project. <br>
- <details>
  <summary> Accessing and Running the File: </summary>
  - Data > Interactions.ino > Arduino IDE > Connect Arduino with laptop > Run the file <br>
</details>

- This code serves as the output component for our hardware project based on the Arduino Uno. It is designed to assist caregivers in identifying abnormalities or issues with the person under their care, particularly elderly individuals. The code controls various devices connected to the Arduino. This code can be a valuable tool for caregivers, helping them identify and respond to various situations involving the person they are caring for, ultimately enhancing the quality of care.

<details> 
  <summary> Here's a breakdown of the code and the process: </summary>
    1. It defines pin assignments for high-power (hP), mist (mP), and LED strip (lP) components, along with a general variable (x). <br>
    2. The code initializes the Adafruit NeoPixel library for controlling an LED strip with 50 LEDs. <br>
    3. In the setup() function, it sets up serial communication, pin modes for mist and high-power devices, and initializes the LED strip to start as white. <br>
    4. The loop() function continuously monitors serial input and activates corresponding functions based on the input. <br>
    5. Depending on the input received (1, 2, 3, or 4), it triggers functions to start mist, heat, display red lights (indicating issues), or display a pattern of green and yellow lights (indicating abnormalities). <br>
    6. The code provides caregivers with a visual indication of potential problems, allowing them to take appropriate action.
</details>

## Camera File
In the directory Camera, You can access various files named [Camera](https://github.com/kushagra1912/Ambient-Interface-for-Elder-Care/tree/main/Camera) which essentially is the different python models for our project. <br>

- This "Camera Directory," is designed to leverage an external camera system for capturing real-time updates from elderly individuals. However, it's important to note that this system doesn't capture actual images or live photos of the person; instead, it employs a Convolutional Neural Network (CNN) model through a Python script to monitor and detect the movements of the elderly person. <br>
- The primary purpose of this system is to identify instances where an elderly person may have fallen down. When such a fall condition is detected, the system generates an output file to alert caregivers or relevant personnel. It's crucial to emphasize that this approach prioritizes non-invasive and privacy-conscious techniques for the purpose of caretaking. The system respects the individual's privacy and does not involve intrusive surveillance. This technology aims to provide an unobtrusive means of enhancing the safety and well-being of elderly individuals under care.

## Main
The [main.py](https://github.com/kushagra1912/Ambient-Interface-for-Elder-Care/blob/main/main.py) file is responsible for starting the application and handle the communication between the microprocessors used for input and output, along with the controlling the camera and sounds used by theis application. It has the following functions:

- connect(port) - This function is used to connect to the serial monitor of a microprocessor, it takes in the port as a parameter and returns the a Serial.serial object

- writeData(ard, data) - This function is used to write given data to the serial monitor, it takes the serial object created by connect() along with the data to be written and writes the data in a string format to the arduino

- readData(ard) - This function is used to read the serial data from the serial monitor, it takes the serial object created by connect() and returns the data as a string enclosed by b''

- connection(inp, out) - This function is the main function of the program and is responsible of all the functioning. It takes two serial objects as parameters the first one being responsible for input of data and the second one responsible for the output of the data.

- startCam() - This function is responsible for starting the camera
