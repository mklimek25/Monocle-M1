import cv2.aruco as aruco
import threading
import cv2
import datetime
from collections import Counter
import numpy as np
import time


class CameraClass:  # detects aruco_tags
    def __init__(self, input_params, aruco_dict=aruco.DICT_4X4_250):
        self.unit_test = False
        self.camera_parameters = input_params['camera_params']
        self.general_parameters = input_params['general_params']
        self.aruco_dict = aruco.getPredefinedDictionary(aruco_dict)
        self.camera_matrix = self.camera_parameters['camera_matrix']
        self.camera_distortion = self.camera_parameters['camera_distortion']
        self.aruco_parameters = aruco.DetectorParameters_create()
        # Adjust detection parameters
        # Reduce min window size for adaptive thresholding
        self.aruco_parameters.adaptiveThreshWinSizeMin = self.camera_parameters['adaptiveThreshWinSizeMin']
        # Allow for larger window size
        self.aruco_parameters.adaptiveThreshWinSizeMax = self.camera_parameters['adaptiveThreshWinSizeMax']
        # Step size for the adaptive thresholding
        self.aruco_parameters.adaptiveThreshWinSizeStep = self.camera_parameters['adaptiveThreshWinSizeStep']
        # Adjust constant subtracted from mean (increases tolerance to lighting)
        self.aruco_parameters.adaptiveThreshConstant = self.camera_parameters['adaptiveThreshConstant']
        # Lower to detect smaller markers
        self.aruco_parameters.minMarkerPerimeterRate = self.camera_parameters['minMarkerPerimeterRate']
        # Increase to allow for larger markers
        self.aruco_parameters.maxMarkerPerimeterRate = self.camera_parameters['maxMarkerPerimeterRate']
        # Adjust to allow imperfect marker shapes
        self.aruco_parameters.polygonalApproxAccuracyRate = self.camera_parameters['polygonalApproxAccuracyRate']
        # Decrease for markers that are close together
        self.aruco_parameters.minCornerDistanceRate = self.camera_parameters['minCornerDistanceRate']
        self.aruco_parameters.cornerRefinementMethod = aruco.CORNER_REFINE_NONE  # Enable corner refinement for more accurate result
        self.corners = None
        self.top_center = None
        self.stop_event = threading.Event()
        self.video_test = self.general_parameters['video_test']
        self.cap = None
        self.rotation_angle = self.camera_parameters['rotation_angle']  # going to hold constant - angle in degrees
        self.collection_count = self.camera_parameters['collection_count']
        self.collection_timespan = self.camera_parameters['collection_timespan']
        self.frame_to_processor_callback = None
        self.start_operation = threading.Event()
        self.prev_tc = None

    def establish_cap(self):
        if self.video_test:
            self.cap = cv2.VideoCapture(self.camera_parameters['video_name'])
            self.start_operation.wait(timeout=1)
        else:
            # noinspection PyUnresolvedReferences
            from picamera2 import Picamera2
            self.cap = Picamera2()
            self.cap.set_controls({'FrameRate': 15})  # Limit FPS to 15 to prevent overload
            camera_config = self.cap.create_video_configuration(main={"format": 'XRGB8888', "size": (1640,1232)})
            self.cap.configure(camera_config)
            if self.camera_parameters['set_controls']:
                self.cap.set_controls({'AeEnable': False})  # Autoexposure turned off
                self.cap.set_controls({'ExposureTimeRange': self.camera_parameters['ExposureTimeRange']})
                self.cap.set_controls({'AnalogueGainRange':self.camera_parameters['AnalogueGainRange']})
            # cap.set_controls({'ExposureTime': 10000, 'AnalogueGain': 3.5})  # manual exposure controld set
            self.cap.start()

    def establish_frame_to_processor_callback(self, callback):
        # ex: callback function may be FrameProcessor.process_frames
        # No Parenthesis
        self.frame_to_processor_callback = callback

    def crop_image(self, _frame, corners):
        if corners is not None:
            cutoff_point = int(corners[1][1])
        else:
            raise Exception
        warped_image = _frame[0:cutoff_point, :]

        return warped_image

    def secure_images(self, images):
        # import os
        pass

        """
       Saves a list of images in a new folder labeled with the current date and time.
       Each image is saved with a filename that includes the date, time, and a unique photo number.

       Args:
           images (list of numpy.ndarray): List of images to save.
       """
        # Generate the folder name using the current datetime
        # timestamp = datetime.datetime.now().strftime("%m-%d-%y-%H-%M-%S")
        # folder_name = self.general_parameters['directory'] + "../training_images/" + timestamp
        #
        # # Create the folder
        # os.makedirs(folder_name, exist_ok=True)
        #
        # # Save each image in the folder with a unique name
        # for i, image in enumerate(images, start=1):
        #     filename = f"{timestamp}-Photo-Number-{i}.png"
        #     filepath = os.path.join(folder_name, filename)
        #
        #     # Save the image using OpenCV
        #     cv2.imwrite(filepath, image)
        #
        # print(f"Images successfully saved in folder: {folder_name}")

    # Example usage:
    # Assuming 'images' is a list of OpenCV images (numpy arrays)
    # save_images_with_timestamp(images)
    def collect_frames(self, corners):
        cap = self.cap
        count = self.collection_count
        timespan = self.collection_timespan
        ret_list = []
        collection_list = []
        comp_time = datetime.datetime.now()
        capture_and_reset = datetime.timedelta(seconds=0 / count)
        ret = True

        while len(ret_list) < count and ret and not self.stop_event.is_set():

            if self.video_test is False:
                frame = cap.capture_array()

                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                time.sleep(0.01)  # Small delay to allow the camera buffer to clear


                ret = True
            else:
                ret, frame = cap.read()

                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


            if ret:

                rot_frame = self.crop_image(frame, corners)


                delta = datetime.datetime.now() - comp_time

                if delta > capture_and_reset:
                    ret_list.append(rot_frame)
                    collection_list.append(frame)
                    comp_time = datetime.datetime.now()

            else:
                break
        if self.camera_parameters['collect_images']:
            self.secure_images(collection_list)
        return ret_list

    def get_most_common_corners_and_top_center(self, corners_list):
        """
        Returns the most common top-left and top-right corners from a list of corners detected by cv2.aruco.detectMarkers,
        and the top-center point based on those corners.

        Args:
            corners_list (list): A list of corner arrays, where each entry is a NumPy array of shape (4, 1, 2)
                                 representing the four corners of a detected marker.

        Returns:
            tuple:
                - list: A list of the most common top-left and top-right corners as tuples [(x1, y1), (x2, y2)].
                - tuple: The top-center point of the ArUco tag as (x, y).
        """
        if not corners_list:
            print('ERROR - NO CORNER LIST ENTERED')
            return None, None

        # Store only the top-left (index 0) and top-right (index 1) corners
        top_left_corners = []
        top_right_corners = []

        for marker_corners in corners_list:
            marker_corners = marker_corners[0]
            corner_array = marker_corners.reshape(-1, 2)  # Convert to shape (4, 2)

            # Extract top-left and top-right corners
            top_left_corners.append((round(float(corner_array[0][0]), 2), round(float(corner_array[0][1]), 2)))
            top_right_corners.append((round(float(corner_array[1][0]), 2), round(float(corner_array[1][1]), 2)))

        # Count occurrences of the top-left and top-right corners
        top_left_counts = Counter(top_left_corners)
        top_right_counts = Counter(top_right_corners)

        # Find the most common top-left and top-right corners
        most_common_top_left = top_left_counts.most_common(1)[0][0]
        most_common_top_right = top_right_counts.most_common(1)[0][0]

        # Combine most common corners into a list
        most_common_corners = [most_common_top_left, most_common_top_right]

        # Calculate the top-center point (midpoint of top-left and top-right)
        top_center = (
            round((float(most_common_top_left[0]) + float(most_common_top_right[0])) / 2, 2),
            round((float(most_common_top_left[1]) + float(most_common_top_right[1])) / 2, 2),
        )

        return most_common_corners, top_center

    def run_CamScan(self):
        corner_list = []
        self.establish_cap()

        """
        Detects an Aruco marker indication by continuously analyzing video frames.
        Stops when the required indication criteria are met or the process is interrupted.

        Args:
            cap (cv2.VideoCapture): The video capture object.
            required_found_count (int): Number of consecutive frames where the marker must be detected.
            hidden_reset_count (int): Number of consecutive frames without a marker before resetting detection.

        Returns:
            bool: True if an indication is found, False otherwise.
        """
        # Validate the video capture object
        required_found_count = self.camera_parameters['required_found_count']
        hidden_reset_count = self.camera_parameters['hidden_reset_count']
        if self.video_test:
            if not self.cap.isOpened():
                raise ValueError("Video capture object is not open.")

        # Initialize detection state variables
        found_bank = 0
        hidden_bank = 0
        indication_ready = False
        try:
            while not self.stop_event.is_set():
                try:
                    if self.video_test:
                        self.start_operation.wait(timeout=1)
                        ret, frame = self.cap.read()
                        cv2.waitKey(99)
                        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame = self.cap.capture_array()
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                        ret = True

                    # Exit loop if the frame cannot be read
                    if not ret:
                        # print("Frame read failed. Exiting detection loop.")
                        self.start_operation.clear()
                        if self.unit_test:
                            exit()

                    else:

                        # Convert frame to grayscale for processing
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        gray_frame = gray_frame[950:1200, 600:900]
                        # Or use adaptive thresholding
                        blurred_gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
                        equalized_hist = cv2.equalizeHist(blurred_gray_frame)

                        cv2.imshow('thresh', equalized_hist)



                        # Detect Aruco markers

                        corners, ids, _ = aruco.detectMarkers(
                           equalized_hist,
                            self.aruco_dict,
                            self.camera_matrix,
                            self.camera_distortion,
                            parameters=self.aruco_parameters,
                        )
                        if len(corners) != 0:
                            adjusted_corners = []
                            for corner in corners:
                                # Adjust the x and y values by adding [600, 1000] to each corner
                                adjusted_corner = corner + np.array([[[600, 950]]], dtype=np.float32)
                                adjusted_corners.append(adjusted_corner.astype(np.float32))
                            corners = adjusted_corners


                        # Display detected markers for debugging/visualization
                        aruco.drawDetectedMarkers(frame, corners, ids)


                        # Process detection results
                        if ids is not None and 3 in ids:  # Check for specific marker ID
                            if len(corner_list) > 100:
                                corner_list = corner_list[-50:]  # Keep only the last 50 entries
                            corner_list.append(corners)
                            found_bank += 1
                            hidden_bank = 0  # Reset hidden_bank if a marker is found
                            if found_bank >= required_found_count and indication_ready:
                                if len(corner_list) > 0:
                                    corners, top_center = self.get_most_common_corners_and_top_center(corner_list)
                                    frame_list = self.collect_frames(corners)
                                    if not self.unit_test:
                                        self._safe_call(self.frame_to_processor_callback,
                                                        (frame_list, corners, top_center))  # was direct call
                                indication_ready = False

                        else:
                            corner_list = []
                            hidden_bank += 1
                            if hidden_bank >= hidden_reset_count:
                                found_bank = 0
                                indication_ready = True
                        cv2.imshow('frame', cv2.resize(frame, (640,480)))

                    # Handle user interruption
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Detection interrupted by user.")
                        break
                except Exception as e:
                    print(f"[CameraClass] Iteration error: {e!r}")
                    continue

        finally:
            # Ensure all resources are released
            if not self.video_test:
                self.cap.stop()  # Properly stop Picamera2
            if self.video_test and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
            print('exiting camscan')


    def draw_markers(self, frame, corners, ids):
        aruco.drawDetectedMarkers(frame, corners, ids)
        return frame

    def _safe_call(self, fn, *args, **kwargs):
        if callable(fn):
            try:
                fn(*args, **kwargs)
            except Exception as e:
                print(f"[CameraClass] Callback error: {e!r}")

    def close_cv2(self):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    from camera_parameters import monocle_parameters
    a = CameraClass(monocle_parameters)
    a.unit_test = True
    a.start_operation.set()
    a.run_CamScan()
