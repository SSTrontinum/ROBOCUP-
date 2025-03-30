//////////////////////////
/// INCLUDED LIBRARIES ///
//////////////////////////
#include <WiFiNINA.h>
#include <SoftwareSerial.h>
#include <L298NX2.h>

////////////////////////
/// GLOBAL VARIABLES ///
////////////////////////
int leftMotorSpeed;
int rightMotorSpeed;
bool valid;
SoftwareSerial mySerial(5,11);
L298NX2 rightMotors(6, 0, 1, 3, 12, 13);
L298NX2 leftMotors(9, 2, 7, 10, 4, 8);

//////////////////////////////
/// USER DEFINED FUNCTIONS ///
//////////////////////////////
void parseSpeed(const char* input, int output[2]) {
    char buffer[20];  // Buffer to avoid modifying original string
    char *endptr; // For validations
    strncpy(buffer, input, sizeof(buffer));
    buffer[sizeof(buffer) - 1] = '\0';
    char* token = strtok(buffer, ",");
    output[0] = strtol(token, &endptr,10);
    if (*endptr != '\0' || endptr == input) {output[0] = -200;}
    token = strtok(NULL, ",");
    output[1] = strtol(token, &endptr,10);
    if (*endptr != '\0'|| endptr == input) {output[1] = -200;}
}

/////////////////////////
/// ARDUINO FUNCITONS ///
/////////////////////////
void setup() {
  WiFi.setLEDs(255, 0, 0);
  Serial.begin(9600);
  mySerial.begin(9600);
  delay(500);
  leftMotors.setSpeed(0);
  rightMotors.setSpeed(0);
  valid = false;
}

void loop() {
  mySerial.listen();
  if (mySerial.available()) { // If the pi sent data to the arduino
    /*
      Data sent will be in the format {leftMotorSpeed,rightMotorSpeed} where
      0 <= leftMotorSpeed <= 510, 0 <= rightMotorSpeed <= 510. Hence, the -255 converts it into a form
      that the motor drivers can use from -255 to 255
    */
    char data[20]; // Holds the data sent
    int result[2]; // After processing the speeds, left motor speed and right motor speed will pop up as result[0] and result[1]
    String inp = mySerial.readStringUntil('\n'); // Reads serial buffer
    inp.toCharArray(data, sizeof(data)); // Converts the string serial buffer to a char array
    parseSpeed(data, result); // splits the char array by ',' into result
    if (result[0] >= 0 && result[0] <= 510 && result[1] >= 0 && result[1] <= 510) {
      valid = true;
    } else {
      valid = false;
    }
    if (valid) {
      WiFi.setLEDs(0, 255, 0); // Set light to green
      // Updates left motor speed and right motor speed
      leftMotors.setSpeed(abs(result[0] - 255));
      rightMotors.setSpeed(abs(result[1] - 255));
      if (result[0] > 255) {
        leftMotors.forward();
      } else if (result[0] < 255) {
        leftMotors.backward();
      } else {
        leftMotors.stop();
      }
      if (result[1] > 255) {
        rightMotors.forward();
      } else if (result[1] < 255) {
        rightMotors.backward();
      } else {
        rightMotors.stop();
      }
    } else {
      // The serial output did not send the correct data
      WiFi.setLEDs(255, 0, 0); // Set light to red
      leftMotors.setSpeed(0);
      rightMotors.setSpeed(0);
      leftMotors.stop();
      rightMotors.stop();
    }
  } else {
    WiFi.setLEDs(0, 0, 255); // Set light to blue
  }
  delay(100); // Delay to improve performance
}