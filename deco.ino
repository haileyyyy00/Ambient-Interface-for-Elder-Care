// the setup routine runs once when you press reset:
void setup() {
  // initialize serial communication at 9600 bits per second:
  Serial.begin(9600);
  pinMode(2,OUTPUT);
}

// the loop routine runs over and over again forever:
void loop() {
  // read the input on analog pin 0:
  int sensorValue = analogRead(A0);
  // Convert the analog reading (which goes from 0 - 1023) to a voltage (0 - 5V):
  float voltage = sensorValue * (5.0 / 1023.0);
  float temperature = voltage * 100;
//  Serial.println("Temp:");
  Serial.println(temperature);
//  if (temperature > 25 ){
//     digitalWrite(2,HIGH);
//    }else{
//      digitalWrite(2,LOW);
//      }
//  if (temperature > 25 ){
//     analogWrite(9,255);
//    }else if (temperature > 23){
//     analogWrite(9,150); 
//    }
//    else{
//      analogWrite(9,50); 
//      }
  Serial.flush();
  startFan(30,temperature);
  delay(1000);
  
}

void startFan(int threshold, int temp){
  int fanSpeed = 255 - (abs(threshold-temp)*10);
  analogWrite( 9, fanSpeed );
  }
