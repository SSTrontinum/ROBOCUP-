import qmc5883l

compass = qmc5883l.QMC5883L()

x, y, z = compass.get_magnet()
print(f"X: {x}, Y: {y}, Z: {z}")
