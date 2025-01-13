import socket
HOST = "128.32.175.252"
PORT = 54321
socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket.bind((HOST, PORT))
socket.listen()
conn, addr = socket.accept() 
print(addr)

