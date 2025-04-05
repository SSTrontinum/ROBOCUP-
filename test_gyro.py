import py_qmc5883l, time
sensor=py_qmc5883l.QMC5883L()
sensor.calibration = [[1.1734590532408868, 0.04732512632887426, 2214.3642882299787], [0.04732512632887431, 1.012911794110473, 4101.536882585206], [0.0, 0.0, 1.0]]
sensor.declination = 0.19

while True:
    print(sensor.get_bearing())