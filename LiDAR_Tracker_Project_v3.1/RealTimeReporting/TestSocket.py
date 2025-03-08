import socket
import time
import threading



if __name__ == "__main__":
    sock = socket.socket(socket.AF_INET, # Internet
                    socket.SOCK_DGRAM) # UDP
    sock.bind(('', 2390))     
    while True:
        data,addr = sock.recvfrom(1206)
        # raw_data_queue.put_nowait((time.time(),data))
        print(data)