import socket
import struct
import threading
import time
from queue import Queue

def connect_to_monitor(host='localhost', port=60001):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    return sock

def send_command(sock, command_str):
    """Send a command to the simulator."""
    msg_bytes = command_str.encode('ascii')
    length_prefix = struct.pack('>I', len(msg_bytes))
    sock.sendall(length_prefix + msg_bytes)

def receive_message(sock):
    """Receive a single length-prefixed message."""
    try:
        # Read length prefix (4 bytes)
        length_data = sock.recv(4)
        if len(length_data) < 4:
            return None
        msg_len = int.from_bytes(length_data, byteorder='big')
        
        # Read message payload
        msg = sock.recv(msg_len)
        return msg.decode('utf-8')
    except Exception as e:
        print(f"Error receiving message: {e}")
        return None

def listen_for_state(sock, state_queue, stop_event):
    """Continuously listen for state updates from the server."""
    print("Listening for state updates...")
    while not stop_event.is_set():
        msg = receive_message(sock)
        if msg is None:
            break
        state_queue.put(msg)

def parse_state_message(msg):
    """Parse state message (S-expression format)."""
    # Simple parser for S-expressions like: (state (GS ...) (scene-graph ...))
    print(f"[State] {msg[:100]}...")  # Print first 100 chars
    return msg

def main():
    # Connect to monitor port
    print("Connecting to simulator...")
    sock = connect_to_monitor('localhost', port=60001)
    print("Connected!")
    
    # Create queue for state updates and stop event
    state_queue = Queue()
    stop_event = threading.Event()
    
    # Start listener thread
    listener_thread = threading.Thread(
        target=listen_for_state, 
        args=(sock, state_queue, stop_event),
        daemon=True
    )
    listener_thread.start()
    
    # Give listener thread time to start
    time.sleep(0.1)
    
    try:
        # Send initial commands
        print("\nSending commands...")
        send_command(sock, "(kickOff Left)")
        time.sleep(0.5)
        
        # Listen for state updates while allowing user input
        print("Receiving state updates (press Ctrl+C to stop)...\n")
        while True:
            # Check for state messages
            if not state_queue.empty():
                state = state_queue.get()
                parse_state_message(state)
            
            time.sleep(0.01)  # Small sleep to prevent busy-waiting
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        stop_event.set()
        listener_thread.join(timeout=1)
        sock.close()
        print("Disconnected")

if __name__ == '__main__':
    main()