
# 2025 RoboCup Singapore Open
## Team SSTrontium

This is the repository for team SSTrontium of Robotics@Apex for the 2025 RoboCup Singapore Open.

## Premise

The Raspberry Pi has Wi-Fi, but transferring code to and from devices and the Raspberry Pi is time-consuming and hard to do. Hence this was set up.
## Demo

Insert gif or link to demo


## Photos
## Hardware

### Microcontrollers
- Raspberry Pi 4 Model B
- SSTuino Rev A
    
### Peripherals
- Raspberry Pi Cam
- 3× VL53L0X Time-of-Flight I²C Sensors
- GY-271 QMC5883L Magnetometer I²C Sensor

### Drivetrain
- 4× N20 100 RPM, 2.1 kg⋅cm Motor 
- 2× L298N Motor Driver

### Others
- PCA9548A I²C Multiplexer 
- 2× TCS34725 Colour I²C Sensor (Used as headlamps)

### Chassis
- Fully 3D printed

### Power 
- 2× 5V Power Banks
- 3× 18560 3.7V 4800 mAh Lithium-ion batteries
## Software
### Raspberry Pi 4 Model B (RPI)
![flowchart](img/flowchart.png)
- Does the bulk of logic
- Checks for obstacles, tracks line, follows green squares and sees red line
- Sends data over serial TX/RX to the SSTuino. Data sent is only motor speeds

### SSTuino Rev A (WiFi-enabled microcontroller)
- Controls the 2 Motor Drivers using the [L298N library by Andrea Lombardo](https://github.com/AndreaLombardo/L298N).
- Reads data sent by the RPI from serial TX/RX input and sets motor speeds accordingly.
## Documentation

[TDP](https://drive.google.com/---)


## Authors

- [@Bryan-Sng](https://www.github.com/SSTrontinum/ROBOCUP-)
- [@FCG-infinite](https://www.github.com/FCG-infinite)
- [@JKYS_11257](https://www.github.com/SSTrontinum/ROBOCUP-)
- [@Selvakumaran-Mugilan](https://www.github.com/SSTrontinum/ROBOCUP-)


## License

[MIT](https://choosealicense.com/licenses/mit/)


## Acknowledgements

 - [Robotics@Apex](https://github.com/roboapex)