import py_qmc5883l, time
sensor=py_qmc5883l.QMC5883L()
sensor.calibration = [[1.0772152386855227, 0.07711129827072825, 1669.8477566463714], [0.07711129827072828, 1.077007497771448, 2400.475569758177], [0.0, 0.0, 1.0]]
sensor.declination = 0.19

while True:
    print(sensor.get_bearing())