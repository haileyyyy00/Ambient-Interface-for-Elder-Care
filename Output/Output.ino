int hP = 9;
int mP = 13;
int lP = 6;
int x = 0;


#include <Adafruit_NeoPixel.h>
#ifdef __AVR__
#include <avr/power.h>
#endif

Adafruit_NeoPixel strip = Adafruit_NeoPixel(150, lP, NEO_GRB + NEO_KHZ800);


void setup() {
  
  Serial.begin(9600);
  pinMode(mP,OUTPUT);
  pinMode(hP,OUTPUT);
  strip.begin();
  strip.show(); //starting white
  pinMode(10, OUTPUT);
  Serial.setTimeout(1);

}

void loop() {

  
  digitalWrite(mP,HIGH);
  
  
  colorWipe(strip.Color(127, 0, 255)); // green

  int in = Serial.parseInt();
  switch(in){
    case 1:
      startHeat();
      break;
    case 2:
      startMist();
      break;
    case 3:
      heartLights();
      break;
     case 4:
      irrLights();
      break;
      
    } 

  
  

}

void startHeat(){
  digitalWrite( hP, HIGH );
  delay(10000);
  digitalWrite( hP, LOW );
  }

void startMist(){
  digitalWrite(mP,LOW);
  delay(3000);
  digitalWrite(mP,HIGH);
  digitalWrite(mP,LOW);
  digitalWrite(mP,HIGH);
  digitalWrite(mP,LOW);
  digitalWrite(mP,HIGH);
  } 

void irrLights(){
  colorWipe(strip.Color(255, 0, 0)); // Red
  delay(500);
  colorWipe(strip.Color(0, 0, 0));
  delay(500);
  colorWipe(strip.Color(255, 0, 0)); // Red
  delay(500);
  colorWipe(strip.Color(0, 0, 0));
  delay(500);
  colorWipe(strip.Color(255, 0, 0)); // Red
  delay(500);
  colorWipe(strip.Color(0, 0, 0));
  }

void heartLights(){
  colorWipe(strip.Color(0, 0, 0));
  colorWipe(strip.Color(0, 255, 0)); // green
  colorWipe(strip.Color(255, 255, 0)); // yellow
  delay(500);
  colorWipe(strip.Color(0, 255, 0)); // green
  colorWipe(strip.Color(255, 255, 0)); // yellow
  delay(500);
  colorWipe(strip.Color(0, 255, 0)); // green
  colorWipe(strip.Color(255, 255, 0)); // yellow
  delay(500);
  colorWipe(strip.Color(0, 255, 0)); // green
  colorWipe(strip.Color(255, 255, 0)); // yellowss
  delay(500);
  colorWipe(strip.Color(0, 255, 0)); // green
  colorWipe(strip.Color(255, 255, 0)); // yellow
  delay(500);
  colorWipe(strip.Color(0, 255, 0)); // green
  colorWipe(strip.Color(255, 255, 0)); // yellow
  delay(500);
  colorWipe(strip.Color(0, 255, 0)); // green
  colorWipe(strip.Color(255, 255, 0)); // yellow
  delay(500);
  colorWipe(strip.Color(0, 255, 0)); // green
  colorWipe(strip.Color(255, 255, 0)); // yellow
  delay(500);
  colorWipe(strip.Color(0, 255, 0)); // green
  colorWipe(strip.Color(255, 255, 0)); // yellow    
  delay(500);
  colorWipe(strip.Color(0, 255, 0)); // green
  colorWipe(strip.Color(255, 255, 0)); // yellow
  delay(500);
  colorWipe(strip.Color(0, 0, 0));

  
  }
  



void colorWipe(uint32_t c) {
  for(uint16_t i=0; i<strip.numPixels(); i+=2) {
    if(i >= 3)
    {
      if(i <=26 || i >= 53) 
      {
        strip.setPixelColor(i, c);
        strip.setBrightness(200);
        strip.show();
      }
    }
  }
}
