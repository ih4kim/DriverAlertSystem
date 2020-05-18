#include <Servo.h>
  

Servo servoleft;  // create servo object to control a servo
Servo servoright;

int posLeft = 0;    // variable to store the servo position
int posRight = 0;
bool reset = false;
char userInput;

void setup() {
  Serial.begin(9600);
  servoleft.attach(7);  // attaches the servo on pin 9 to the servo object
  servoright.attach(6);
}

void loop() {
  if (!reset){
    servoleft.write(90);
    servoright.write(0);
    reset = true;
  }
  if (Serial.available()>0){
    userInput = Serial.read();
    if (userInput == 's'){
      servoleft.write(0);
      servoright.write(90);
      delay(1000);
      reset = false;
    }
  }
}
