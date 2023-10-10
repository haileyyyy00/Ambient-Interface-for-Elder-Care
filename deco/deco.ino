// the setup routine runs once when you press reset:
void setup() {
  
  Serial.begin(9600);
  pinMode(2,OUTPUT);
}

// the loop routine runs over and over again forever:
void loop() {
  // read the input on analog pin 0:
  int sensorValue = analogRead(A0);
  
  float voltage = sensorValue * (5.0 / 1023.0);
  float temperature = voltage * 100;
  Serial.println(temperature);
  Serial.flush();
  startHeat(30,temperature);
  delay(1000);
  
}

void startHeat(int threshold, int temp){
  int heatLevel = 255 - (abs(threshold-temp)*10);
  analogWrite( 9, fanSpeed );
  }
