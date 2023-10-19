#include <Wire.h>
#include "MAX30105.h"
#include "spo2_algorithm.h"

MAX30105 particleSensor;

#define MAX_BRIGHTNESS 255

#if defined(__AVR_ATmega328P__) || defined(__AVR_ATmega168__)
uint16_t irBuffer[100]; //infrared LED sensor data
uint16_t redBuffer[100];  //red LED sensor data
#else
uint32_t irBuffer[100]; //infrared LED sensor data
uint32_t redBuffer[100];  //red LED sensor data
#endif


// Pulse and SpO2
int32_t spo2; //SPO2 value
int32_t heartRate; //heart rate value
int pulseLED = 11; //Must be on PWM pin
int readLED = 13; //Blinks with each data read

// Touch Functionality
int touchSens = 2;
int inputState;

void setup(){
  
  Serial.begin(9600); // initialize serial communication at 115200 bits per second:

// Initialize MAX30102
  pinMode(pulseLED, OUTPUT);
  pinMode(readLED, OUTPUT);
  particleSensor.begin(Wire, I2C_SPEED_FAST);

  byte ledBrightness = 60; //Options: 0=Off to 255=50mA
  byte sampleAverage = 4; //Options: 1, 2, 4, 8, 16, 32
  byte ledMode = 2; //Options: 1 = Red only, 2 = Red + IR, 3 = Red + IR + Green
  byte sampleRate = 100; //Options: 50, 100, 200, 400, 800, 1000, 1600, 3200
  int pulseWidth = 411; //Options: 69, 118, 215, 411
  int adcRange = 4096; //Options: 2048, 4096, 8192, 16384

  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange); //Configure sensor with these settings

// Initialize touch sensor
  pinMode(touchSens, INPUT);
  
}


// the loop routine runs over and over again forever:
void loop() {

  int in = Serial.parseInt();
  if (in == 1){
    checkHO2();
  }
  
  checkHeat();
  checkLight();
  checkTouch();
  checkPress();
  
}

void checkLight(){
  int lightVal = analogRead(A0);
  if (lightVal <=10){
    Serial.println(1);
    }
  }
  
void checkHeat(){
  int tempVal = analogRead(A2);  
  float voltage = tempVal * (5.0 / 1023.0);
  float temperature = voltage * 100;
  if (temperature > 25){
    Serial.println(2);
  }  
}

void checkTouch(){
  inputState=digitalRead(touchSens);
  if(inputState== HIGH){
      Serial.println(3);
    }
}

void checkPress(){
  int force = analogRead(A1);
  if(force > 500){
    Serial.println(4);
    delay(1000);
  }  
}


void checkHO2(){

  Serial.println(5);
  int start = millis();
  int beats = 0;
  float red_dc = 0;
  float ir_dc = 0;
  int ir_val, red_val, ir_max, ir_min, red_max, red_min, red_ac, ir_ac;
  ir_max = 0;
  ir_min = 400;
  red_max = 0;
  red_min = 50000;

    for (byte i = 0 ; i < 100 ; i++)  {
      while (particleSensor.available() == false) //do we have new data?

        particleSensor.check(); //Check the sensor for new data
        digitalWrite(readLED, !digitalRead(readLED)); //Blink onboard LED with every data read
        
        red_val = particleSensor.getRed();
        ir_val = particleSensor.getIR();
        if(ir_val <40){
          
          red_dc += red_val;
          ir_dc += ir_val;
          ir_max = (ir_max > ir_val) ? ir_max : ir_val;
          ir_min = (ir_min < ir_val) ? ir_min : ir_val;
          red_max = (red_max > red_val) ? red_max : red_val;
          red_min = (red_min < red_val) ? red_min : red_val;
          red_ac = abs(red_max-red_min);
          ir_ac = abs(ir_max-ir_min);
          if ((ir_max - ir_min) > 20 & (ir_max - ir_min) < 1000){
            beats++;
          }
        }
        particleSensor.nextSample();
    }   

  int heartRate = (beats*6)/10;
  red_dc /= 100;
  ir_dc /=100;
  float r = (red_ac/red_dc)/(ir_ac/ir_dc);
  spo2 = 110 -25 * r;

  if (heartRate < 60 || heartRate > 105){
      Serial.println(6);
    }

  if (spo2 > 94){
      Serial.println(7);
    }
  
}
