import serial

ser = serial.Serial('/dev/ttyACM0', timeout=1)

with open("output/raw_output.ubx", "wb") as f:
    while True:
        data = ser.read(100)
        if data:
            f.write(data)

