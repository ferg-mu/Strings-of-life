# import socket

# # Define server IP address and port
# SERVER_IP = '127.0.0.1'  # localhost
# SERVER_PORT = 12345

# # Create a socket object
# server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# # Bind the socket to the server address
# server_socket.bind((SERVER_IP, SERVER_PORT))

# # Publish "Hello, world!" to the network
# message = "Hello, world!"
# server_socket.sendto(message.encode(), (SERVER_IP, SERVER_PORT))

# # Close the socket
# server_socket.close()


import socket
import struct

# Define server IP address and port
SERVER_IP = '127.0.0.1'  # localhost
SERVER_PORT = 12345

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Bind the socket to the server address
server_socket.bind((SERVER_IP, SERVER_PORT))

# OSC message format: address pattern + type tag string + argument values
message = "/test"  # OSC address pattern
type_tags = ",s"   # Type tag string indicating a string argument
argument_values = ["Hello, world!"]  # Argument values (as strings)

# Pack OSC message
osc_message = struct.pack('>4s', b'#bundle')  # Bundle header
osc_message += struct.pack('>i', 1)            # Bundle timestamp (arbitrary, 1)
osc_message += _string_to_bytes(message)       # OSC address pattern
osc_message += _string_to_bytes(type_tags)     # Type tag string
for arg in argument_values:
    osc_message += _string_to_bytes(arg)       # Argument values

# Send OSC message to TouchDesigner
server_socket.sendto(osc_message, (SERVER_IP, SERVER_PORT))

# Close the socket
server_socket.close()

def _string_to_bytes(s):
    """Convert a string to bytes and add padding."""
    return bytes(s, 'utf-8') + b'\x00' * (4 - len(s) % 4)