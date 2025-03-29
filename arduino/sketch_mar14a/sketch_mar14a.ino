#include <SPI.h>
#include <WiFiNINA.h>
#include <Vector.h>
Vector<int> process_text(char str[]) {
  Vector<int> arr;
  char *token = strtok(str, ",");
  while (token != NULL) {
      arr.push_back(atoi(token));
      token = strtok(NULL, ",");
  }
  return arr;
}
void setup() {
  Serial.begin(9600);
  WiFi.setLEDs(255, 0, 0);
  delay(500);
  WiFi.setLEDs(0, 255, 0);
  delay(1000);
}
void loop() {
  if (Serial.available()) {
    WiFi.setLEDs(0, 255, 0);
    String data = Serial.readStringUntil('\n');
    Serial.println(data);
  } else {
    WiFi.setLEDs(0, 0, 255);
    char test[] = "254,123";
    Serial.print(process_text(test)[0]);
    Serial.print(",");
    Serial.println(process_text(test)[1]);
  }
  delay(500);
}