import socket

UDP_PORT = 2368

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', UDP_PORT))
print(f"Listening on UDP port {UDP_PORT}...")

while True:
    data, addr = sock.recvfrom(1500)
    print(f"Received {len(data)} bytes from {addr}")
    # 여기서 데이터 파싱 함수 호출 가능
