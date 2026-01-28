from opcua import Server, ua
from threading import Thread
import time


class OPCUAServer:
    def __init__(self, endpoint="opc.tcp://0.0.0.0:4840/freeopcua/server/"):
        self.server = Server()
        self.server.set_endpoint(endpoint)
        self.namespace = self.server.register_namespace("MyOPCUAServer")
        self.objects = self.server.get_objects_node()
        # enter security keys
        self.server.load_certificate("server_certificate.der")
        self.server.load_private_key("server_private_key.pem")
        self.server.set_security_policy([
            ua.SecurityPolicyType.NoSecurity,
            ua.SecurityPolicyType.Basic256Sha256_SignAndEncrypt,
            ua.SecurityPolicyType.Basic256Sha256_Sign,
        ])

        # Dictionary to hold variable nodes
        self.variables = {}

        # Server control
        self.running = False
        self.thread = None

    def create_variable(self, name, initial_value=0.0):
        """Create a new variable that can be updated later."""
        var = self.objects.add_variable(self.namespace, name, initial_value)
        var.set_writable()  # Important: allows clients to write too if needed
        self.variables[name] = var

    def update_variables(self, incoming_data):
        """Update the value of an existing variable."""
        for key in incoming_data.keys():
            print(key)
            if key in self.variables:
                self.variables[key].set_value(incoming_data[key])
            else:
                raise KeyError('key is not in set variables')

    def start(self):
        """Start the OPC UA server in a separate thread."""
        if not self.running:
            self.running = True

    def run_server(self):
        self.server.start()
        try:
            while self.running:
                time.sleep(0.1)
        finally:
            self.server.stop()

    def stop(self):
        """Stop the OPC UA server."""
        self.running = False
        if self.thread is not None:
            self.thread.join()

# Example usage:
if __name__ == "__main__":
    opcua_server = OPCUAServer()
    opcua_server.create_variable("Temperature", 25.0)
    opcua_server.create_variable("Pressure", 1.0)
    opcua_server.start()
    opcua_server.stop()

