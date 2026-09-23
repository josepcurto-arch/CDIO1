import serial
from pyubx2 import UBXMessage, SET, UBXReader

# Replace with your actual port
port = "/dev/ttyACM0"
baudrate = 9600

# Open serial connection
with serial.Serial(port, baudrate, timeout=2) as ser:

    # Create a UBX-CFG-MSG message to enable RXM-RAWX on UART1 (target=1)
    msg = UBXMessage("CFG", "CFG-MSG", SET,
                     msgClass=0x02,  # RXM
                     msgID=0x15,     # RAWX
                     rateDDC=0,
                     rateUART1=0,
                     rateUART2=0,
                     rateUSB=1,
                     rateSPI=0)

    # Send it
    ser.write(msg.serialize())

    # Optional: read and print response
    ubr = UBXReader(ser)
    print("Waiting for ACK...")
    for i in range(5):
        (_, parsed) = ubr.read()
        if parsed:
            print(parsed)
            break