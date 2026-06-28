import socket
import struct

def connect_to_monitor(host='127.0.0.1', port=60001):
    """Connect to the simulator's monitor port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    return sock

def send_command(sock, command_str):
    """Send an S-expression command to the simulator.
    
    Args:
        sock: Connected socket to monitor port
        command_str: S-expression as string, e.g., "(kickOff Left)"
    """
    msg_bytes = command_str.encode('ascii')
    # 4-byte length prefix in big-endian
    length_prefix = struct.pack('>I', len(msg_bytes))
    sock.sendall(length_prefix + msg_bytes)

# Example usage
sock = connect_to_monitor()
send_command(sock, "(kickOff Left)")
#send_command(sock, "(ball (pos 0 0 0.5) (vel 1 1 0))")
sock.close()