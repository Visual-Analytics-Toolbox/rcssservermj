from types import FrameType
import socket
import signal
import threading
import logging
import re
from sexpdata import loads, dumps
# ---------- LOGGING CONFIG ----------
# console handler
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
ch.setLevel(logging.INFO)

# configure logging
logging.basicConfig(handlers=[ch], level=logging.DEBUG)
# ---------- LOGGING CONFIG ----------

logger = logging.getLogger(__name__)

class Client:
    def __init__(self, host: str, port: int, team: str, player_no: int, model_name: str | None = None):
        """
        Construct a new agent connecting to the given server.
        """

        self._host: str = host
        self._port: int = port

        self._model_name: str = 'ant' if model_name is None else model_name
        self._team: str = team
        self._player_no: int = player_no

        self._rcv_buffer_size = 1024
        self._rcv_buffer = bytearray(self._rcv_buffer_size)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self._has_beamed: bool = False

        # set TCP_NODELAY option to send messages immediately (without buffering)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    def run(self):
        """
        Run the simulation client.
        """

        # connect to server
        logger.info('Connecting to server at %s:%d...', self._host, self._port)
        try:
            self._sock.connect((self._host, self._port))
        except ConnectionRefusedError:
            logger.error('Connection refused. Make sure the server is running and listening on the specified interface.')  # noqa: TRY400
            return
        # logger.info('Server connection established.')

        # create client thread
        client_thread = threading.Thread(target=self._action_loop)
        client_thread.start()

        # wait for client thread to finish
        client_thread.join()

        # logger.info('Shutting down.')

        # close server connection
        self._sock.close()

    def shutdown(self) -> None:
        """
        Shutdown the client.
        """

        self._sock.shutdown(socket.SHUT_RDWR)
    
    def _action_loop(self):
        """
        Main loop of the agent.
        """

        logger.info('Initializing agent...')
        init_msg = f'(init {self._model_name} {self._team} {self._player_no})'
        self._send_message(init_msg.encode())

        logger.info('Running perception-action-loop.')
        while True:
            try:
                perception_msg = self._receive_message()

                perception_msg_str = perception_msg.decode()
                perception_data = self.parse_sensor_string(perception_msg_str)
                print(perception_data["GS"])
                print()
                quit()

            except Exception as e:
                logger.info('Server connection closed or client crashed.')
                logger.info('Exception details:', exc_info=e.__traceback__)
                break
    
    def _send_message(self, msg: bytes | bytearray) -> None:
        """
        Receive the next message from the TCP/IP socket.
        """

        self._sock.send((len(msg)).to_bytes(4, byteorder='big') + msg)

    def _receive_message(self) -> bytes | bytearray:
        """
        Receive the next message from the TCP/IP socket.
        """

        # receive message length information
        if self._sock.recv_into(self._rcv_buffer, nbytes=4, flags=socket.MSG_WAITALL) != 4:
            raise ConnectionResetError

        msg_size = int.from_bytes(self._rcv_buffer[:4], byteorder='big', signed=False)

        # ensure receive buffer is large enough to hold the message
        if msg_size > self._rcv_buffer_size:
            self._rcv_buffer_size = msg_size
            self._rcv_buffer = bytearray(self._rcv_buffer_size)

        # receive message with the specified length
        if self._sock.recv_into(self._rcv_buffer, nbytes=msg_size, flags=socket.MSG_WAITALL) != msg_size:
            raise ConnectionResetError

        return self._rcv_buffer[:msg_size]
    
    def parse_sensor_string(self, s: str) -> dict:
        """
        Parses a sensor data string of nested parenthesis groups into a structured dictionary.
        Repeated top-level tags are aggregated into lists.
        """
        result = {}
        # Top-level groups: (TAG ...content...)
        top_level_pattern = re.compile(r'\((\w+)((?:\s*\([^()]*\))*)\)')
        
        for tag, inner in top_level_pattern.findall(s):
            # Find inner key-value or key-list groups: (key val1 val2 ...)
            items = re.findall(r'\(\s*(\w+)((?:\s+[^()]+)+)\)', inner)
            group = {}
            for key, vals in items:
                tokens = vals.strip().split()
                parsed_vals = []
                for t in tokens:
                    try:
                        parsed_vals.append(float(t))
                    except ValueError:
                        parsed_vals.append(t)
                # Single value vs. list
                group[key] = parsed_vals[0] if len(parsed_vals) == 1 else parsed_vals
            
            # Merge into result, handling repeated tags as lists
            if tag in result:
                if isinstance(result[tag], list):
                    result[tag].append(group)
                else:
                    result[tag] = [result[tag], group]
            else:
                result[tag] = group
        
        return result


if __name__ == '__main__':


    # create client
    client = Client("localhost", 60000, "Berlin", "1", "T1")

    # register SIGINT handler
    def signal_handler(sig: int, frame: FrameType | int | signal.Handlers | None) -> None:
        del sig, frame  # signal unused parameter
        client.shutdown()

    signal.signal(signal.SIGINT, signal_handler)

    # run client
    client.run()
