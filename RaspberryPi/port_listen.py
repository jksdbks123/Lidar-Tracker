import socket
import time
import dpkt

def reading_packets(sock,Packets_queue):
    
    while True:
        data,addr = sock.recvfrom(1206)
        Packets_queue.put_nowait(data)

if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, # Internet
                             socket.SOCK_DGRAM) # UDP
    sock.bind(('', 2368)) 
    while True:
        data,addr = sock.recvfrom(1206)
        print(addr,time.time())
        time.sleep(1)



       