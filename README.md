# <p align= "center">Ambient Interface for Elder Care 

## Project Description
The project focuses on creating an ambient interface to support the elderly without intruding on their daily lives. It consists of two components: an in-house tracking system for non-invasive data collection, and an external display for caretakers. By using Arduino-based wearables, IoT technology, and non-invasive audio-visual signals, we ensure seamless monitoring of elderly individuals' health while respecting their privacy. This innovative approach aims to improve elderly care without compromising their independence and comfort.

## Table of Contents
* Sensors and Tools Used
* [Input](https://github.com/Sanchitjain16/Ambient-Interface-for-Elder-Care/tree/main/input)
* [Output](https://github.com/Sanchitjain16/Ambient-Interface-for-Elder-Care/tree/main/output)
* [Camera]()
* [Main]()
* [Usage]()

## Input Component
This code is designed for our Arduino Uno-based hardware project. Once uploaded to the Arduino using the Arduino IDE, the hardware collects data non-intrusively for processing. The code includes functions for checking temperature and measuring vital signs like SPO2 (blood oxygen saturation) and heart rate using a MAX30105 sensor, pressure sensor and a Photodiode for light. The sensor data is communicated over a serial connection, and the hardware can be triggered to measure vital signs by sending '1' through the serial monitor. The code is set up to provide accurate and non-invasive monitoring for a health-related project.

<details> 
  <summary> Here's a breakdown of the code and the process: </summary>
  1. In the setup() function, it sets up serial communication, pin modes for LEDs, and initializes the MAX30105 sensor with specific configurations. <br>
  2. In the `loop() function`, the code listens for input from the serial monitor. If it receives '1', it triggers the checkSPO2() function. <br>
  3. The `checkHeat() function` reads an analog temperature sensor (connected to A0), calculates the temperature in degrees Celsius, and prints '1' to the serial monitor if the temperature exceeds 20°C. <br>
  4. The `checkSPO2() function` measures SPO2 and heart rate using the **MAX30105 sensor**. It collects data over 100 samples, calculates important metrics, and reports the results to the serial monitor. <br>
  5. The calculated SPO2 and heart rate values are then processed and displayed for monitoring and analysis.<br>
</details>
