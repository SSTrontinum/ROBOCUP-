
import time
import smbus
#import adafruit_tca9548a
import board
import adafruit_bno055 as bno055

"""
# TCA9548A multiplexer class
class TCA9548A(object):
    def __init__(self, bus, address=0x70):
        self.i2c_address = address
        self.bus = smbus.SMBus(bus)

    def setup(self, channel):
        try:
 
            self.bus.write_byte(self.i2c_address, channel)
            return 1, "Success"
        except Exception as msg:
            return 0, "Fail: {}".format(msg)

 
mux = TCA9548A(bus=1)

 
status, message = mux.setup(0x08)
 

 
# (rst=None assumes no hardware reset pin is used; adjust if ned)
bno = bno055.BNO055(busnum=1, rst=None)


if not bno.begin():
    raise RuntimeError("Failed to initialise BNO055!")


status, self_test, error = bno.get_system_status()
print("System status:", status)
print("Self test result (0x0F is normal): 0x{0:02X}".format(self_test))
if status == 0x01:
    print("System error:", error)

 
try:
    while True:
        heading, roll, pitch = bno.read_euler()
        print("Heading={0:0.2F}° Roll={1:0.2F}° Pitch={2:0.2F}°".format(heading, roll, pitch))


        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting program.
"""
bno = bno055.BNO055_I2C(board.I2C())
status, self_test, error = bno.get_system_status()
print("System status:", status)
print("Self test result (0x0F is normal): 0x{0:02X}".format(self_test))
if status == 0x01:
    print("System error:", error)

 
try:
    while True:
        heading, roll, pitch = bno.read_euler()
        print("Heading={0:0.2F}° Roll={1:0.2F}° Pitch={2:0.2F}°".format(heading, roll, pitch))


        time.sleep(1)
except KeyboardInterrupt:
    print("Exiting program.")