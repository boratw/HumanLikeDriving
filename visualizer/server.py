import socket
import threading
import os
import traceback

class VisualizeServer():
    def __init__(self, port=5454):
        self.handlers = {}
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(("", port))
            self.socket.listen()
            print("Successfully started server at port " + str(port))
        except:
            print("Failed to start server at port " + str(port))


    def Receive(self):
        self.client_socket, addr = self.socket.accept()
        try:
            str = self.client_socket.recv(4096).decode()
            request = (str.split("\n")[0]).split(" ")
            if request[0] == "GET":
                if request[1] == "/":
                    self.SendFile("visualizer/main.html")
                elif request[1][:3] == "/v/":
                    self.SendData(request[1])

                else:
                    self.SendFile("visualizer" + request[1])
        except UnicodeDecodeError:
            pass
        finally:
            self.client_socket.close()
        

    def SendFile(self, filename):
        try:
            ext = filename[(filename.rfind(".") + 1):]
            if ext == "html":
                content_type = "text/html"
                cached = False
            elif ext == "js":
                content_type = "text/javascript"
                cached = False
            elif ext == "png":
                content_type = "image/png"
                cached = True
            elif ext == "map":
                content_type = "text/map"
                cached = True
            else:
                raise IOError()
                

            file_size = os.path.getsize(filename) 
            with open(filename, "rb") as r:
                msg = \
                    "HTTP/1.1 200 OK\r\n" +\
                    "Access-Control-Allow-Origin: *\r\n" +\
                    "content-type: " + content_type + "\r\n" +\
                    "content-length: " + str(file_size) + "\r\n"
                
                if cached == True:
                    msg += \
                        "Cache-Control: public, max-age=31536000\r\n" +\
                        "Age: 0\r\n" +\
                        "Expires: Wed, 13 Mar 2033 22:03:00 GMT\r\n"
                else:
                    msg += \
                        "Cache-Control: no-cache\r\n" +\
                        "Age: 0\r\n"
                    
                msg += "Connection: close\r\n\r\n"
                self.client_socket.send(msg.encode())
                self.client_socket.send(r.read())
        except Exception as e:
            msg = \
                "HTTP/1.1 404 Not Found\r\n" +\
                "content-type: text/plain\r\n" +\
                "content-length: 9\r\n" +\
                "Connection: close\r\n\r\n" +\
                "Not found"
            self.client_socket.send(msg.encode())
            print(e)
            traceback.print_exc()
    
    def SendData(self, filename):
        try:
            requestparse = filename.split("/")
            data = self.handlers[requestparse[2]](requestparse[3:])
            msg = \
                "HTTP/1.1 200 OK\r\n" +\
                "Access-Control-Allow-Origin: *\r\n" +\
                "content-type: text/plain" + "\r\n" +\
                "content-length: " + str(len(data)) + "\r\n" +\
                "Connection: close\r\n\r\n"
        
            self.client_socket.send(msg.encode())
            self.client_socket.send(data.encode())
        except Exception as e:
            msg = \
                "HTTP/1.1 404 Not Found\r\n" +\
                "content-type: text/plain\r\n" +\
                "content-length: 9\r\n" +\
                "Connection: close\r\n\r\n" +\
                "Not found"
            self.client_socket.send(msg.encode())
            print(e)
            traceback.print_exc()

    
    def Destroy(self):
        self.socket.close()
