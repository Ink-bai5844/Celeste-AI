import socket

class SocketServer:
    def __init__(self, host="127.0.0.1", port=25001):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(1)

    def await_connection(self):
        return self.sock.accept()
