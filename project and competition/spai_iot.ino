///*
//  AnalogReadSerial
//
//  Reads an analog input on pin 0, prints the result to the Serial Monitor.
//  Graphical representation is available using Serial Plotter (Tools > Serial Plotter menu).
//  Attach the center pin of a potentiometer to pin A0, and the outside pins to +5V and ground.
//
//  This example code is in the public domain.
//
//  https://www.arduino.cc/en/Tutorial/BuiltInExamples/AnalogReadSerial
//*/
//
//// the setup routine runs once when you press reset:
//void setup() {
//  // initialize serial communication at 9600 bits per second:
//  Serial.begin(9600);
//}
//
//// the loop routine runs over and over again forever:
//void loop() {
//  // read the input on analog pin 0:
//  int sensorValue = analogRead(A0);
//  // print out the value you read:
//  Serial.println(sensorValue);
//  delay(1);        // delay in between reads for stability
//}
#include<WiFi.h>

#define ssid "WiFi_ThePine"
#define password ""

void initWiFi(){
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi ..");
  while (WiFi.status() != WL_CONNECTED) {
    Serial.println(WiFi.status());
    Serial.print('_');
    delay(1000);
  }
  Serial.println("Connected");
  Serial.println(WiFi.status());
  Serial.println(WiFi.localIP());
  Serial.print("RRSI: ");
  Serial.println(WiFi.RSSI());
}

void setup(){
  Serial.begin(115200);
  initWiFi();
}

void loop() {
  
}
