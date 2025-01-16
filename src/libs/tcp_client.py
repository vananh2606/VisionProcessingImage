import socket

HOST = "127.0.0.1"
PORT = 8080

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

try:
    while True:
        message = input("Nhập tin nhắn gửi tới server ('exit' để thoát): ")
        if message.lower() == "exit":
            print("Đóng kết nối.")
            break

        client_socket.sendall(message.encode("utf-8"))
        data = client_socket.recv(1024)
        print(f"Phản hồi từ server: {data.decode('utf-8')}")
finally:
    client_socket.close()
