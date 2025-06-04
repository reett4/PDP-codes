import json
import socket


""" 
    Reads and prints data received over from a socket.
"""


HOST = '0.0.0.0'; PORT = 5050   # configuration-specific (works on MacOS)


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"listening on port {PORT}...")
    conn, addr = s.accept()
    with conn:
        buffer = ""
        while True:
            chunk = conn.recv(1024).decode('utf-8')
            if not chunk:
                break
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                data = json.loads(line)
                print("received:", data)