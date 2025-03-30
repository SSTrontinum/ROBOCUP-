import py_qmc5883l, time
sensor=py_qmc5883l.QMC5883L()
sensor.calibration = [[1.17476834930162, 0.018605719015194554, 1400.3312322052009], [0.01860571901519455, 1.0019807521296373, 3622.058744732488], [0.0, 0.0, 1.0]]
sensor.declination = 0.19

while True:
    print(sensor.get_bearing())