import serial.tools.list_ports 
import silence_tensorflow.auto 
import local_img_classifier as lic
import serial

# Connect to the serial port
ports = list(serial.tools.list_ports.comports())

for p in ports:
    print(p)

user = serial.Serial('COM4', baudrate=9600, timeout=1)

output = lic.get_value()

match output:
    case "cardboard":
        user.write("1".encode())
    case "glass":
        user.write("2".encode())
    case "metal":
        user.write("3".encode())
    case "paper":
        user.write("4".encode())
    case "plastic":
        user.write("5".encode())

s