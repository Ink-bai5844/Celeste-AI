import struct

from .SessionData import SERIALIZED_VALUE_COUNT


class SocketInterface:

    def __init__(self, sock, agent, sessionParameters, preprocessor=lambda x: x, route_planner=None):
        self.sock = sock
        self.IODimensions = (SERIALIZED_VALUE_COUNT, 7)
        self.agent = agent
        self.preprocessor = preprocessor
        self.sessionParameters = sessionParameters
        self.route_planner = route_planner
        self.active_plan = None

    def agent_output_to_byte(self, floats):
        if len(floats) != 7:
            raise ValueError("Input list must contain 7 floats")

        byte = 0
        for i, f in enumerate(floats):
            if f > 0.5:
                byte |= 1 << i

        return bytes([byte])

    def recv_exact(self, byte_count):
        data = bytearray()
        while len(data) < byte_count:
            chunk = self.sock.recv(byte_count - len(data))
            if not chunk:
                raise ConnectionError("Socket closed while receiving data")
            data.extend(chunk)
        return bytes(data)

    def recv_frame_or_end(self):
        first_chunk = self.recv_exact(3)

        if first_chunk == b"END":
            return "END"

        if first_chunk == b"MAP":
            next_chunk = self.recv_exact(4)
            if next_chunk == b"INFO":
                return "MAPINFO"
            first_chunk = first_chunk + next_chunk

        expected_bytes = self.IODimensions[0] * 4
        data = bytearray(first_chunk)
        while len(data) < expected_bytes:
            chunk = self.sock.recv(expected_bytes - len(data))
            if not chunk:
                raise ConnectionError("Socket closed while receiving frame")
            data.extend(chunk)
        return bytes(data)

    def receive_map_info_and_send_plan(self):
        self.sock.sendall(bytes(1))
        length_data = self.recv_exact(4)
        payload_length = struct.unpack("<I", length_data)[0]
        if payload_length <= 0 or payload_length > 1024 * 1024:
            raise ValueError(f"Invalid MAPINFO payload length: {payload_length}")
        map_info = self.recv_exact(payload_length).decode("utf-8")

        if self.route_planner is None:
            plan_payload = "{}"
            self.active_plan = None
        else:
            self.active_plan = self.route_planner.plan_for_map(map_info)
            plan_payload = self.active_plan.to_json()

        encoded = plan_payload.encode("utf-8")
        self.sock.sendall(struct.pack("<I", len(encoded)))
        self.sock.sendall(encoded)

    def run(self):
        receivedData = self.sock.recv(1024)
        if receivedData.decode("ASCII") != "LOADINFO":
            print("Invalid data received : expected LOADINFO, got :")
            print(receivedData)
            raise Exception()
        self.sock.sendall(bytes(self.sessionParameters.SerializeToXML(), "UTF-8"))

        try:
            while True:
                receivedData = self.recv_frame_or_end()

                if receivedData == "END":
                    self.sock.sendall(bytes(1))
                    finalData = self.recv_exact(self.IODimensions[0] * 4)
                    finalPlayerState = struct.unpack(f"{str(self.IODimensions[0])}f", finalData)
                    self.sock.sendall(bytes(1))
                    return finalPlayerState

                if receivedData == "MAPINFO":
                    self.receive_map_info_and_send_plan()
                    continue

                deserialized = struct.unpack(f"{str(self.IODimensions[0])}f", receivedData)
                try:
                    processed_inputs = self.preprocessor(deserialized, self.active_plan)
                except TypeError:
                    processed_inputs = self.preprocessor(deserialized)
                self.sock.sendall(self.agent_output_to_byte(self.agent(processed_inputs)))

        except Exception as e:
            print(f"Client disconnected: {e}")
            print(receivedData)
            raise
