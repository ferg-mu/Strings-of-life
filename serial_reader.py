import serial

# Define serial port settings
SERIAL_PORT = 'COM10'  # Adjust the port as needed
BAUD_RATE = 9600  # Adjust the baud rate as needed

try:
    # Open serial port for reading
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
    print(f"Serial port {SERIAL_PORT} opened successfully.")

    # Continuously read data from serial port
    while True:
        data = ser.readline().strip()  # Read a line of data from the serial port
        if data:
            print("Received:", data.decode())  # Print the received data

except serial.SerialException as e:
    print(f"Failed to open serial port {SERIAL_PORT}: {e}")
