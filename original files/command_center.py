import time
from camera_class import CameraClass
from frame_processor_class import FrameProcessor
from data_manager_class import DataManager
from gui_manager_class import GUIManager
# from opcua_server import OPCUAServer
import tkinter as tk
import cv2
import queue
import threading
from camera_parameters import monocle_parameters
import os


class MonocleCommandCenter:
    def __init__(self, camera_parameters):
        # Initialize components (CameraClass, FrameProcessor, etc.)
        self.camera_parameters = camera_parameters
        self.root = tk.Tk()
        self.camera = CameraClass(camera_parameters)
        self.frame_processor = FrameProcessor(camera_parameters)
        self.data_manager = DataManager(camera_parameters)

        self.gui = GUIManager(self.root, camera_parameters)
        # self.opcua_server = OPCUAServer()

        # Initialize Data Storage Components
        self.frame_queue = queue.Queue()
        self.camera_data = None
        self.frame_processor_data = None
        self.database_data = None
        self.gui_settings_data = None
        self.operator_command_data = None
        self.running_operation = False
        # Setup threading management
        self.threads = {}
        self.stop_event = threading.Event()
        self.iteration = 0
        self.camera_thread = None
        self.processor_thread = None
        self.test_mode = camera_parameters['general_params']['test_mode']
        self.data_generated = False
        self._watchdog_stop = threading.Event()
        self._watchdog_thread = None




    def initialize_CS_callbacks(self):
        # Camscan sends frame list to processor
        self.camera.establish_frame_to_processor_callback(self.frame_processor.queue_frame_processor)
        # Frame Processor sends output data to data manager

        # gui sends tuning parameter updates to the frame processor
        self.gui.initiate_tuning_parameter_callback(self.frame_processor.update_tuning_parameters)

        # GUI sends back settings data when 'Apply Settings' button is pressed on the GUI
        self.gui.callback_settings_params(self.frame_processor.receive_gui_settings_data)
        # Frame Processor sends back processed data and image, or error message when processing is performed
        self.frame_processor.establish_data_to_data_manager_callback(self.data_manager.receive_frame_processor_data)
        # self.frame_processor.establish_data_to_server_callback(self.opcua_server.update_variables)
        # Frame processor will send result frame to GUI
        self.frame_processor.establish_image_to_gui_callback(self.gui.receive_img)
        # GUI will send request to Data Manager
        self.gui.establish_hist_spinbox_callback(self.data_manager.send_requested_data_list)
        # Frame Processor will send error code to gui
        self.frame_processor.establish_gui_error_callback(self.gui.receive_error_callback)
        # Data Manager will return data list back to GUI
        # self.data_manager.establish_historian_list_callback(self.gui.receive_historian_data_list)
        # Data Manager sends data to GUI
        # start button will begin scan threads via callback
        self.gui.establish_start_button_callback(self.start_scan)
        self.gui.establish_stop_button_callback(self.stop_processing)


    def setup_server(self):
        item_list = self.camera_parameters['data_frame_parameters']['data_columns']
        for item in item_list:
            self.opcua_server.create_variable(item, None)

    def is_connected(self, host="8.8.8.8", port=53, timeout=3):
        import socket
        """
        Host: 8.8.8.8 (Google DNS)
        Port: 53 (DNS)
        """
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except socket.error:
            return False
    def _start_watchdog(self):
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        def _watch():
            while not self._watchdog_stop.is_set():
                try:
                    # Camera thread down? restart it.
                    if self.camera_thread and not self.camera_thread.is_alive():
                        self.gui.raise_error("Camera thread died — restarting.")
                        self.camera_thread = threading.Thread(
                            target=self.camera.run_CamScan, name="camera-thread", daemon=True
                        )
                        self.camera_thread.start()
                    # Processor thread down? restart it.
                    if self.processor_thread and not self.processor_thread.is_alive():
                        self.gui.raise_error("Processor thread died — restarting.")
                        self.processor_thread = threading.Thread(
                            target=self.frame_processor.run_frame_processor,
                            name="processor-thread",
                            daemon=True
                        )
                        self.processor_thread.start()
                except Exception as e:
                    # Never let the watchdog die
                    try:
                        self.gui.raise_error(f"Watchdog error: {e!r}")
                    except Exception:
                        pass
                time.sleep(2)
        self._watchdog_thread = threading.Thread(target=_watch, name="watchdog", daemon=True)
        self._watchdog_thread.start()
    def start_scan(self, boolean):
        self.data_generated = self.data_manager.setup_folder_structure()
        if self.data_generated:
            if not self.is_connected():
                self.gui.raise_error('NOT CONNECTED TO INTERNET')
                return
            if boolean and len(self.gui.variables) != 0:
                """Start the scanning process."""
                # self.setup_server()
                '''Set up holding folder for data'''

                if self.iteration == 0:
                    self.iteration += 1
                    self.camera_thread = threading.Thread(
                        target=self.camera.run_CamScan, name='camera-thread', daemon=True)
                    self.camera_thread.start()

                    # Thread for processing frames
                    self.processor_thread = threading.Thread(
                        target=self.frame_processor.run_frame_processor, name='processor-thread', daemon=True)
                    self.processor_thread.start()
                    self.gui.show_frame('home_frame')

                    # Thread for OPCUA Server
                    # self.server_thread = threading.Thread(
                    #     target=self.opcua_server.run_server()
                    # )
                    # self.server_thread.start()
                elif self.camera_parameters['general_params']['video_test']:
                    self.camera.cap = cv2.VideoCapture(self.camera_parameters['camera_params']['video_name'])
                    self.camera.start_operation.set()

                else:
                    pass
            else:
                # hola
                self.gui.raise_error("Enter in Settings Before Starting")
        else:
            self.gui.raise_error("Connect USB Device before proceeding")
        self._start_watchdog()

    def stop_processing(self, turnoff=True):
        """Stop all processing threads."""
        if self.data_generated:
            try:
                self.data_manager.send_excel_via_gmail(self.data_manager.excel_path,
                                                          self.camera_parameters['data_frame_parameters']
                                                          ['recipient_email_passcode'],
                                                          self.camera_parameters['data_frame_parameters']
                                                          ['sender_email_username'],
                                                          self.camera_parameters['data_frame_parameters']
                                                          ['sender_email_passcode'])
            except Exception as e:
                self.gui.raise_error(f"Email send error: {e!r}")

        self.camera.stop_event.set()

        self.frame_processor.stop_event.set()
        time.sleep(3)
        self.camera.close_cv2()
        if self.camera_thread is not None:
            self.camera_thread.join()
        if self.processor_thread is not None:
            self.processor_thread.join()
        # if self.server_thread is not None:
        #     self.server_thread.join()
        self._watchdog_stop.set()
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=2)
        time.sleep(2)

        self.gui.exit_gui()
        time.sleep(2)
        if turnoff and self.camera_parameters['general_params']['test_mode'] is False:
            os.system("sudo shutdown now")
        elif self.camera_parameters['general_params']['test_mode'] and turnoff:
            print('would have turned off if run on pi')


    def apply_mode(self, root, test_mode):
        if not test_mode:
            root.overrideredirect(True)
            root.attributes('-fullscreen', True)
        else:
            root.overrideredirect(False)
            root.attributes('-fullscreen', False)

    def run_gui(self):
        self.apply_mode(self.gui.root, self.test_mode)
        self.gui.setup_gui()




if __name__ == "__main__":
    command_center = MonocleCommandCenter(monocle_parameters)

    command_center.initialize_CS_callbacks()
    command_center.run_gui()
