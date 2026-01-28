import cv2
import threading
import queue
# import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
import math

# CHANGED (Event Bus): FrameProcessor will publish results/images/errors instead of calling callbacks directly.
# NOTE: This is a *non-breaking migration step* — we keep the old callback attributes/methods too.
from event_bus import EventBus, Event, EventType
"""Frame processor takes input parameters and runs a series of algorithms to identify and measure the 
cut surface"""
class FrameProcessor:
    def __init__(self, input_params, event_bus: EventBus | None = None):
        self.general_params = input_params['general_params']
        self.frame_processor_params = input_params['frame_proccessor_params']

        # CHANGED (Event Bus): store the shared EventBus instance (injected by CommandCenter).
        # If None, the class will behave as before (callbacks only).
        self.event_bus = event_bus
        self.aruco_top_center = None  # needs to be set via callback
        self.database_callback = None
        self.gui_callback = None
        self.settings_params = {
            'email_alerts': None,
            'language_setting': None,
            'product': None,
            'minimum_height_value': None,
            'maximum_height_value': None,
            'minimum_density_value': None,
            'maximum_density_value': None,
            'minimum_surface_area_value': None,
            'maximum_surface_area_value': None
        }
        # BELOW ARE CALIBRATION PARAMETERS
        self.y_b = self.frame_processor_params['init_y_scalar']
        self.x_b = self.frame_processor_params['init_x_scalar']
        self.top_to_bottom_scalar = self.frame_processor_params['init_top_to_bottom_scalar']
        self.left_to_right_scalar = self.frame_processor_params['init_left_to_right_scalar']

        self.gui_image = None
        self.processed_img_list = None
        self.aruco_tc = None
        self.processing = None
        self.frame_processor_callback = None
        self.column_label_to_data_manager_callback = None
        self.data_to_data_manager_callback = None
        # self.data_to_server_callback = None
        self.gui_error_callback = None
        self.image_to_gui_callback = None
        self.most_recent_results = None
        self.activation_event = threading.Event()
        self.stop_event = threading.Event()
        self.queue = queue.Queue()
        self.unit_test = False
        self.height_coefficients = np.poly1d([237.15248044, -631.50139453, 502.08674595, -52.65718121])
        self.width_coefficients = np.poly1d([255.40114839, -437.50245826,  157.08603982,  102.35178353])
        self.rotation_angle = 1

    def update_tuning_parameters(self, dict):
        # updates tuning parameters as a callback function when prompted by the GUI
        print('received tuning parameters, updating now')
        for label in dict.keys():
            if hasattr(self, label):
                setattr(self, label, dict[label])
            else:
                raise ValueError(f'Error, {label} not a recognized attribute')



    def establish_gui_error_callback(self, callback):
        # INITIATES THE GUI ERROR CALL
        self.gui_error_callback = callback

    def establish_column_label_to_data_manager_callback(self, callback):
        # INITIATES CALLBACK TO DATA MANAGER FOR COLUMN OF DATA
        self.column_label_to_data_manager_callback = callback

    def establish_data_to_data_manager_callback(self, callback):
        # INITIATES CALLBACK TO DATA MANAGER FOR SENDING DATA OFF TO DATA MANAGER
        self.data_to_data_manager_callback = callback

    #def establish_data_to_server_callback(self, callback):
     #   self.data_to_server_callback = callback

    def establish_image_to_gui_callback(self, callback):
        self.image_to_gui_callback = callback

    def receive_gui_settings_data(self, data):
        keys = data.keys()
        for key in keys:
            value = data[key]
            if value is None:
                raise ValueError('value should not be None, but is')
            else:
                self.settings_params[key] = value

        self.gui_settings_data = data

    def frame_processor_callback(self, callback):
        self.frame_processor_callback = callback

    def collect_scalars(self, data):  # receiving a data callback from the GUI when entered
        x_tuning_variable, y_scalar, bot_to_top_scalar, left_to_right_scalar = data
        self.x_b = x_tuning_variable
        self.y_b = y_scalar
        self.bottom_to_top_scalar = bot_to_top_scalar
        self.left_to_right_scalar = left_to_right_scalar

    def determine_rotation(self, corners):
        point1, point2 = corners
        x1, y1 = point1
        x2, y2 = point2

        delta_x = x2 - x1
        delta_y = y2 - y1

        angle_rad = math.atan2(delta_y, delta_x)
        angle_deg = math.degrees(angle_rad)
        print(f'angle_deg: {angle_deg}')
        return angle_deg


    def preprocess(self, frame):
        '''creates a profile using a combination of the inRange function and the Sobel function.'''
        # this function applies fundamental image processing practices to make image readable by program
        ksize = self.frame_processor_params['sobel_kernel_size']
        blur_k = self.frame_processor_params['gauss_blur_kernel_size']
        closing_k = self.frame_processor_params['closing_kernel_size']
        subsection = self.frame_processor_params['check_points']
        cth = self.frame_processor_params['color_tolerance_high']
        ctl = self.frame_processor_params['color_tolerance_low']

        hframe = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_subsection = hframe[subsection[1][1]:subsection[0][1], subsection[0][0]:subsection[1][0]]


        mean_color = cv2.mean(frame_subsection)
        mean_blue, mean_green, mean_red = mean_color[:3]
        lower_range = np.array([int(max((mean_blue - ctl, 0))), int(max((mean_green - ctl, 0))), int(max((mean_red - ctl, 0)))])
        upper_range = np.array([int(min((mean_blue + cth, 255))), int(min((mean_green + cth, 255))), int(min((mean_red + cth, 255)))])

        mask = cv2.inRange(hframe, lower_range, upper_range)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        final_thresh_cutoff = self.frame_processor_params['final_threshold_param']
        gray_frame = cv2.blur(gray_frame, (blur_k, blur_k))  # blurring smoothens the image and improves tead accuracy
        # gray_frame = np.where(gray_frame == 255, 0, gray_frame)# In events of extreme brightness, turns 255 to 0
        gx = cv2.Sobel(gray_frame, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)  # line detection alg
        gy = cv2.Sobel(gray_frame, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)
        gx = cv2.convertScaleAbs(gx)
        gy = cv2.convertScaleAbs(gy)
        combined = cv2.addWeighted(gx, 0.8, gy, 0.8, 0)
        combined = cv2.erode(combined, (blur_k,blur_k), iterations=1)
        combined = cv2.dilate(combined, (blur_k,blur_k), iterations=1)
        kernel = np.ones((closing_k,closing_k))
        closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        ret, _thresh = cv2.threshold(closed, final_thresh_cutoff, 255, cv2.THRESH_BINARY_INV)
        result = cv2.bitwise_and(mask, _thresh)
        if self.unit_test:
            cv2.imshow('inRangeThresh', cv2.resize(mask, (640,480)))
            cv2.imshow('SobelThresh', cv2.resize(_thresh, (640,480)))
            cv2.imshow('combined', cv2.resize(result, (640,480)))
            cv2.waitKey(0)
        return result
    
    def passes_contour_criteria(self, contour):
        '''checks if all points in param list are inside of a given contour'''
        for point in self.frame_processor_params['check_points']:
            x, y = point
            # check to see if the point is inside the contour using the point polygon test
            result = cv2.pointPolygonTest(contour, (x,y), False)
            if result <= 0:  # -1 = outside
                return False
        return True

    def finding_contour_of_interest(self, thresh, is_simple, aruco_top_center):
        # this function finds the largest contour in the right section of the processed image
        # simple applies to the contour display algorithm I am using. I am using the faster method that loses
        # marginal accuracy, might be worth revisiting this choice later.

        if is_simple is True:
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        height = 0
        coi = None
        _bc = None
        # contours parsed for largest available (bun contour)
        for contour_ in contours:
            # parsing contours for
            if cv2.contourArea(contour_) > 300000 and self.passes_contour_criteria(contour_):

                x, y, w, h = cv2.boundingRect(contour_)

                if y+h > height:
                    height = thresh.shape[0] - (y + h)
                    coi = contour_
                    _bc = (x + w/2, y + h)

        if coi is not None:
            xo = _bc[0]
            yo = _bc[1]
            xi = aruco_top_center[0]
            yi = aruco_top_center[1]
            h_depth = yi - yo
            print(f'hdepth: {h_depth}')
            return coi, h_depth, _bc
        else:
            return None, None, None

    def refine_contour(self, contour):
        def rotate_contour(contour, center):
            x_min, y_min, width, height = cv2.boundingRect(contour)

            # Compute x positions at 40% and 60% of width
            x_40 = x_min + 0.4 * width
            x_60 = x_min + 0.6 * width


            # Extract contour points
            contour_points = contour[:, 0, :]  # Convert to (N,2) shape

            # Find closest y-values at x_40 and x_60
            mask_40 = np.abs(contour_points[:, 0] - x_40) < 3
            mask_60 = np.abs(contour_points[:, 0] - x_60) < 3

            if np.any(mask_40):
                y_40 = np.max(contour_points[mask_40, 1])
            else:
                y_40 = np.max(contour_points[:, 1])

            if np.any(mask_60):
                y_60 = np.max(contour_points[mask_60, 1])
            else:
                y_60 = np.max(contour_points[:, 1])

            # Compute rotation angle
            angle = np.arctan2(y_60 - y_40, x_60 - x_40) * 180 / np.pi  # Convert to degrees

            # Compute rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Rotate contour without explicitly creating homogeneous coordinates
            rotated_points = (contour_points @ rotation_matrix[:, :2].T) + rotation_matrix[:, 2]

            # Reshape and return as integer contour
            return rotated_points.reshape(-1, 1, 2).astype(np.int32)


    # def rotate_contour(contour, center):
        #     angle = self.rotation_angle + self.frame_processor_params['rotation_angle']
        #
        #
        #     # create a rotation matrix
        #     rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        #
        #     # rotate each point in contour
        #     contour_points = contour.reshape(-1, 2)  # convert contour to Nx2
        #     ones = np.ones((contour_points.shape[0], 1))  # add a column of ones for homogenous transform
        #     points_homogenous = np.hstack([contour_points, ones])  # Nx3 array
        #     rotated_points = points_homogenous @ rotation_matrix.T  # rotate points
        #     rotated_contour = rotated_points.reshape(-1, 1, 2).astype(np.int32)
        #     return rotated_contour

        def stretch_contour(contour, center):
            height_polynomial = np.poly1d(self.height_coefficients)
            width_polynomial = np.poly1d(self.width_coefficients)
            # inches get converted to pixels which are 1/100"
            mx = 100 * self.top_to_bottom_scalar
            bx = 100 * self.x_b
            my = 100 * self.left_to_right_scalar

            by = 100 * self.y_b
            zx, zy, width, height = cv2.boundingRect(contour)
            height_slope = my / width

            stretch_coordinates = []
            xmid, ymid = center
            for point in contour:

                x, y = point[0]
                dx = x - xmid
                if dx > 0:
                    x_int_var = 1
                else:
                    x_int_var = -1
                dy = y - ymid
                if dy > 0:
                    y_int_var = 1
                else:
                    y_int_var = -1

                xpercent = x / width
                ypercent = y / height
                tuning_height_expansion = height_polynomial(xpercent)
                tuning_width_expansion = width_polynomial(ypercent)
                left_to_right_expansion = xpercent * my

                top_to_bottom_expansion = ypercent * mx
                stretch_x = (xmid + dx) + x_int_var * (tuning_width_expansion + top_to_bottom_expansion + bx) // 2
                stretch_y = (ymid + dy) + y_int_var * (tuning_height_expansion + left_to_right_expansion + by) // 2
                stretch_coordinates.append([stretch_x, stretch_y])
            res_contour = np.array(stretch_coordinates, dtype=np.int32).reshape((-1, 1, 2))
            return res_contour

        def normalize_contour(contour):
            # Find the minimum x and y values in the contour
            min_x = np.min(contour[:, 0, 0])
            min_y = np.min(contour[:, 0, 1])

            # Subtract the minimum values to normalize the contour
            normalized_contour = contour - [min_x, min_y]

            return normalized_contour

        def iterate_slices(_image, direction):
            print('performing iteration')
            """
            Trim slices of an image starting from the middle, looking for necks in both directions.

            Parameters:
                _image (np.ndarray): Binary image representing the contour.
                direction (str): 'row' or 'column' to specify the axis for slicing.

            Returns:
                np.ndarray: Trimmed binary image, or the same image if no necks are found.
            """
            width_req = self.frame_processor_params.get('minimum_neck_thickness', 1)
            max_neck_length = self.frame_processor_params.get('maximum_neck_length', 10)

            axis = 0 if direction == 'row' else 1
            middle = _image.shape[axis] // 2
            neck_found_upper = False
            neck_found_lower = False

            # Initialize boundaries to the full range
            upper_boundary = 0
            lower_boundary = _image.shape[axis]

            for delta in range(1, middle + 1):
                # Check upper slice (towards the start of the axis)
                if not neck_found_upper and middle - delta >= 0:
                    upper_slice = np.sum(_image.take(indices=middle - delta, axis=axis))
                    if upper_slice < width_req:
                        upper_boundary = middle - delta
                        neck_found_upper = delta >= max_neck_length
                    else:
                        neck_found_upper = True  # Neck condition no longer satisfied

                # Check lower slice (towards the end of the axis)
                if not neck_found_lower and middle + delta < _image.shape[axis]:
                    lower_slice = np.sum(_image.take(indices=middle + delta, axis=axis))
                    if lower_slice < width_req:
                        lower_boundary = middle + delta
                        neck_found_lower = delta >= max_neck_length
                    else:
                        neck_found_lower = True  # Neck condition no longer satisfied

                # Stop iterating if both sides are processed
                if neck_found_upper and neck_found_lower:

                    break

            # Check if boundaries are unchanged (no necks found)
            if upper_boundary == 0 and lower_boundary == _image.shape[axis]:
                return _image  # Return the original image if no trimming is needed

            # Trim the image based on the identified boundaries
            if axis == 0:  # Trim rows
                _image = _image[upper_boundary:lower_boundary, :]
            else:  # Trim columns
                _image = _image[:, upper_boundary:lower_boundary]

            return _image

        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            raise ValueError('contour has 0 area, cannot calculate center')

        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        center = (cx, cy)
        rotated_contour = rotate_contour(contour, center)
        rotated_contour = normalize_contour(rotated_contour)
        stretched_contour = stretch_contour(rotated_contour, center)
        normalized_contour = normalize_contour(stretched_contour)
        x, y, w, h = cv2.boundingRect(normalized_contour)
        contour_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.fillPoly(contour_canvas, [normalized_contour], color=(255, 255, 255))

        if self.unit_test:
            cv2.imshow('rotated_contour', cv2.resize(contour_canvas, (640, 480)))
            cv2.waitKey(0)
        return contour_canvas

    def measure_contour(self, contour, Hdepth, xmid, ybot):

        def process_row(row):
            if not row.strip():
                print("Warning: Empty row detected.")
                return []

            row_params = row.split(',')
            processed_params = []

            for s in row_params:
                s = s.strip()  # Strip leading/trailing spaces
                try:
                    processed_params.append(float(s))
                except ValueError:
                    print(f"Warning: Could not convert '{s}' to float.")
                    processed_params.append(None)  # Or some default value

            return processed_params
        # step1: find parameters (above and below) that will be used
        calibration_measurement_params = []  # params will include params for distance 1 and params for distance 2.
        # params used in actual calibration will be a weighted average of these two param values
        new_contour = []
        # populating parameters with values based on calibration
        # do I need to be opening this over and over?
        with open(self.general_params['directory'] + 'parameters.csv', 'r') as f:
            _i = 0
            old_row = None
            for row in f:
                row_params = process_row(row)
                if None not in row_params:
                    # looking for the first row that exceeds Hdepth, if first row, only that one considered
                    if row_params[0] == Hdepth:
                        calibration_measurement_params.append(row_params)
                        calibration_measurement_params.append(None)
                        break
                    if row_params[0] > Hdepth:
                        if _i == 0:
                            calibration_measurement_params.append(row_params)
                            calibration_measurement_params.append(None)
                            composition = 1
                        else:
                            calibration_measurement_params.append(row_params)
                            calibration_measurement_params.append(old_row)
                            composition = (Hdepth - old_row[0])/(row_params[0] - old_row[0])
                        break
                else:
                    pass
                old_row = row_params
                _i += 1


        # step2: convert all contour points (xi = xi-xm, yi = y_bot-yi)
        if len(calibration_measurement_params) != 0:
            # going point by point in contour and dissecting position based on calibration
            # taking xi, yi on image and converting it to xo, yo in the real world positions are relative to fixed
            # point at the bottom center of the contour
            for c in contour:
                xi, yi = c[0]
                xi = xi - xmid
                yi = ybot - yi
                # step3: apply calibration to adjusted contour
                a1, b1, c1, d1, e1, f1, g1, h1 = calibration_measurement_params[0][1:]
                # apply changes to simplify and graph
                dxodxi1 = a1 * yi**3 + b1 * yi**2 + c1 * yi + d1
                # apply changes to simplify and graph
                xo1 = xi * dxodxi1

                yo1 = 0
                for j in range(int(yi)):
                    dyodyi1 = e1 * j**3 + f1 * j**2 + g1 * j + h1
                    yo1 += dyodyi1

                if calibration_measurement_params[1] is not None:  # should never be none, need to update when I recalibrate
                    a2, b2, c2, d2, e2, f2, g2, h2 = calibration_measurement_params[1][1:]
                    # apply changes to simplify and graph
                    dxodxi2 = a2 * yi**3 + b2 * yi**2 + c2 * yi + d2
                    # apply changes to simplify and graph
                    # dyodyi2 = e2 * yi**3 + f2 * yi**2 + g2 * yi + h2
                    xo2 = xi * dxodxi2

                    yo2 = 0
                    for j in range(int(yi)):
                        dyodyi2 = e2 * j**3 + f2 * j**2 + g2 * j + h2
                        yo2 += dyodyi2
                    xo = (xo1 * composition + xo2 * (1 - composition)) * 29 / 25.4
                    yo = (yo1 * composition + yo2 * (1 - composition)) * 29 / 25.4
                else:
                    xo = xo1
                    yo = yo1

                # yo += yo * left_to_right * (abs(xi)-820)/820 * xi/abs(xi)
                new_contour.append([round(xo*100), round(yo*100)])
            # Convert new_contour to a numpy array of the correct shape and type

            new_contour = np.array(new_contour, dtype=np.int32).reshape((-1, 1, 2))
            x, y, w, h = cv2.boundingRect(new_contour)

            for c_number in range(len(new_contour)):
                new_contour[c_number][0] = [new_contour[c_number][0][0]-x, h - new_contour[c_number][0][1]]
            # new_contour = self.rotate_contour(new_contour, self.frame_processor_params['rotation_angle'])
            # x, y, w, h = cv2.boundingRect(new_contour)
            
            return new_contour, w, h
            # step4: determine x,y range of new contour
        else:
            print('appears no params')
            return None, None, None

    def profile(self, black_image, num_height_sections, num_width_sections, shell_pixel_width=None,
                height_shell_cut_param=None, width_shell_cut_param=None):
        """
        Optimized contour profiling using vectorized NumPy operations.
        Ensures height results are ordered left-to-right and width results are ordered top-to-bottom.
        Positions middle sections near the center of the contour for each axis.
        """
        def calculate_bounds(start_val, end_val, num_sections, pixel_width):
            """Calculate evenly spaced shell bounds."""
            bounds = np.zeros((num_sections, 2), dtype=int)
            section_size = (end_val - start_val) // (num_sections - 1)

            for i in range(num_sections):
                start = start_val + i * section_size
                if i == num_sections - 1:  # Last section
                    start = end_val - pixel_width
                    end = end_val
                else:
                    end = start + pixel_width
                bounds[i] = [start, end]

            return bounds
        def profile_shell(black_image, bounds, axis):
            """Calculate profile information for given bounds using median of min_vals and max_vals."""
            result_info = []
            rect_info = []

            for bound in bounds:
                min_vals_list = []
                max_vals_list = []
                start, end = bound

                if axis == 'x':  # Profiling heights (left-to-right)
                    for bound_point in range(start, end):
                        slice_column = black_image[:, bound_point]
                        non_zero_indices = np.nonzero(slice_column)
                        if non_zero_indices[0].size == 0:  # Handle empty slices
                            min_vals_list.append(0)
                            max_vals_list.append(0)
                        else:
                            min_vals_list.append(np.min(non_zero_indices[0]))  # Along y-axis
                            max_vals_list.append(np.max(non_zero_indices[0]))
                else:  # axis == 'y', Profiling widths (top-to-bottom)
                    slice_row = black_image[start:end, :]
                    non_zero_indices = np.nonzero(slice_row)
                    if non_zero_indices[1].size == 0:  # Handle empty slices
                        min_vals_list.append(0)
                        max_vals_list.append(0)
                    else:
                        min_vals_list.append(np.min(non_zero_indices[1]))  # Along x-axis
                        max_vals_list.append(np.max(non_zero_indices[1]))

                # Calculate medians of min_vals and max_vals
                min_val = np.median(min_vals_list)
                max_val = np.median(max_vals_list)

                # Store rectangle information
                rect_info.append(((start, min_val), (end, max_val)))

                # Calculate result using the medians
                result = (max_val - min_val)
                result_info.append(result)

            return result_info, rect_info




    # Calculate contour area
        white_pixel_mask = np.all(black_image == [255, 255, 255], axis=-1)

        # Count the number of white pixels
        area = round(np.sum(white_pixel_mask)/10000, 2)

        contour_height, contour_width, _ = black_image.shape
        # Calculate height and width bounds
        if shell_pixel_width is None:
            shell_pixel_width = self.frame_processor_params['shell_pixel_width']
        if height_shell_cut_param is None:
            height_shell_cut_param = self.frame_processor_params['height_shell_cut_param']
        if width_shell_cut_param is None:
            width_shell_cut_param = self.frame_processor_params['width_shell_cut_param']

        height_bounds = calculate_bounds(int(height_shell_cut_param * contour_width),
                                         int((1-height_shell_cut_param) * contour_width),
                                         num_height_sections, shell_pixel_width)
        width_bounds = calculate_bounds(int(width_shell_cut_param * contour_height),
                                        int((1-width_shell_cut_param) * contour_height),
                                        num_width_sections, shell_pixel_width)
        # Sort bounds for proper order
        height_bounds = height_bounds[np.argsort(height_bounds[:, 0])]  # Sort by x-coordinate (left-to-right)
        width_bounds = width_bounds[np.argsort(width_bounds[:, 0])]    # Sort by y-coordinate (top-to-bottom)

        # Profile height and width
        height_results, height_rect_info = profile_shell(
            black_image, height_bounds, axis='x'
        )
        width_results, width_rect_info = profile_shell(
            black_image, width_bounds, axis='y'
        )

        return width_results, height_results, area, height_rect_info, width_rect_info

    def ransac_calculations(self, input_list, d):
        x_values = np.arange(len(input_list)).reshape(-1, 1)
        y_values = np.array(input_list).reshape(-1, 1)
        base_model = LinearRegression()
        ransac = RANSACRegressor(base_model, min_samples=2, residual_threshold=1.0, max_trials=100)
        ransac.fit(x_values, y_values)

        line_x = np.arange(len(y_values)).reshape(-1, 1)
        line_y_ransac = ransac.predict(line_x)
        residuals = np.abs(y_values - line_y_ransac)
        threshold = 0.1 * (np.max(line_x) - np.min(line_x))  # everything outside 5% will be cut off
        inline_mask = residuals < threshold

        inline_y_mean = np.mean(y_values[inline_mask])
        if d == 1:
            return round(inline_y_mean/100, 2)
        else:
            return round(inline_y_mean, 2), inline_mask

    def analyze_scan(self, data_list):
        def ret_scan_error(message):
            scan_error = True
            return None, None, None, None, None, None, None, None, scan_error, message
        # This function analyzes the scans results
        if len(data_list) < 3:
            print('error, less than 5 frames processed')
            return ret_scan_error('ERROR - LESS THAN 3 FRAMES PROCESSED')
        height_results = []
        width_results = []

        width_list = [x[0][0] for x in data_list]
        height_list = [x[1][0] for x in data_list]
        area_list = [x[2][0] for x in data_list]
        if None in width_list or None in height_list or None in area_list:
            print('error coming from None value in ret data')
            return ret_scan_error('ERROR - NULL VALUE IN DATA')

        for z in range(len(height_list[0])):  # parsing each height column
            ransac_height_data = [n[z] for n in height_list]  # gathering data across all scans
            average_height = self.ransac_calculations(ransac_height_data, 1)
            if average_height is None:
                print('error coming from ransac in heights')
                return ret_scan_error("RANSAC Error - Heights")
            height_results.append(round(average_height, 2))
        average_height = round(sum(height_results)/len(height_results), 2)

        for zo in range(len(width_list[0])):  # parsing each width row
            ransac_width_data = [n[zo] for n in width_list]  # gathering data across all scans
            rs_width_result = self.ransac_calculations(ransac_width_data, 1)
            if rs_width_result is None:
                print('error coming from ransac in widths')
                return ret_scan_error('RANSAC Error - Widths')
            width_results.append(round(rs_width_result, 2))
        average_width = round(sum(width_results)/len(width_results), 2)

        area_result, index_list = self.ransac_calculations(area_list, 2)
        if area_result is None:
            print('error coming from ransac in area')
            return ret_scan_error('RANSAC Error - Area')
        index = next((index for index, value in enumerate(index_list) if value), False)
        gui_image = data_list[index][3]
        height_rect_params = data_list[index][4]
        width_rect_params = data_list[index][5]
        return (height_results, average_height, width_results, average_width, area_result, gui_image,
                height_rect_params, width_rect_params, None, None)

    def process_frame(self, gray_frame, aruco_top_center):
        x_val_list = []
        y_val_list = []
        area_val_list = []
        # processing image to become readable by program
        threshed_img = self.preprocess(gray_frame)
        # Finding object (contour) that we wish to scan. Scanned object is given an associated depth from image
        # which is determined by measuring the distance from the bottom of the object to the aruco-tag fixed position
        coi, h_depth, bc = self.finding_contour_of_interest(threshed_img, False, aruco_top_center)
        hDepthRange = self.frame_processor_params['hDepthRange']
        # passing argument that helps
        if coi is not None and h_depth > hDepthRange[0] and h_depth < hDepthRange[1]:
            img_copy = gray_frame.copy()
            cv2.drawContours(img_copy, [coi], -1, (0, 255, 0), 3)
            x_mid, y_bot = bc

            # contour is measured through our calibration process with a checkerboard
            # image is remade with new dimensions
            contour_info = self.measure_contour(coi, h_depth, x_mid, y_bot)

            ret_contour, contour_width, contour_height = contour_info
            if ret_contour is not None:
                contour_canvas = np.zeros((contour_height, contour_width, 3), dtype=np.uint8)
                cv2.fillPoly(contour_canvas, [ret_contour], color=(255, 255, 255))
                contour_image = self.refine_contour(ret_contour)
                if ret_contour is None or contour_width is None or contour_height is None:
                    data = None
                    
                else:
                    x_vals, y_vals, area_val, height_shell_params, width_shell_params = self.profile(
                        black_image=contour_image, num_height_sections=5, num_width_sections=3)
                    x_val_list.append(x_vals)
                    y_val_list.append(y_vals)
                    area_val_list.append(area_val)
                    data = (x_val_list, y_val_list, area_val_list, contour_image, height_shell_params, width_shell_params)
            else:

                data = None
        else:
            data = None



        if self.unit_test:
            cv2.imshow('gray frame', cv2.resize(gray_frame, (640,480)))

            cv2.imshow('preprocessed', cv2.resize(threshed_img, (640,480)))

            coi_img = gray_frame.copy()
            if coi is not None:
                cv2.drawContours(coi_img, [coi], -1, (0, 255, 0), 3)
            cv2.imshow('coi', cv2.resize(coi_img, (640,480)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return data

    def queue_frame_processor(self, frame_list_data):
        # Entering into queue list of frames, cornerpoints found - most common, top-center
        self.queue.put(frame_list_data)
        self.activation_event.set()


    def run_frame_processor(self):
        """Continuously process frames from the queue."""
        if not self.gui_settings_data:
            return

        reading_number = 0
        

        # noinspection PyUnresolvedReferences
        product = self.gui_settings_data['product']
        self.processing = True
        while not self.stop_event.is_set():
            data_list = []
            self.activation_event.wait(timeout=1)
            if not self.queue.empty():
                self.activation_event.clear()
                print('starting frame processor')
                # Attempt to get a frame from the queue
                frame_list, corners, aruco_tc = self.queue.get(timeout=1)  # Wait for up to 1 second
                self.rotation_angle = self.determine_rotation(corners)
                timestamp = datetime.datetime.now().timestamp()
                for frame in frame_list:
                    if self.stop_event.is_set():
                        break
                    data = self.process_frame(frame, aruco_tc)
                    if data is not None:
                        data_list.append(data)

                if self.stop_event.is_set():
                    break
                (height_results, average_height, width_results,
                 average_width, area_result, contour_image, height_rect_params, width_rect_params,
                 scan_error, scan_error_message) = self.analyze_scan(data_list)
                if scan_error or area_result is None:
                    product_error = None
                    yield_result = None
                else:
                    product_error = self.check_for_product_error((height_results, width_results, area_result))
                    # noinspection PyUnresolvedReferences
                    yield_result = round(self.settings_params['minimum_surface_area_value'] / area_result, 2)

                self.most_recent_results =  {
                    'timestamp': timestamp,
                    'product': product,
                    'reading_number': reading_number,
                    'height_results': height_results,
                    'average_height': average_height,
                    'width_results': width_results,
                    'average_width': average_width,
                    'area_val': area_result,
                    'ret_yield': yield_result,
                    'scan_error': scan_error,
                    'product_error': product_error
                }

                callback_img = self.generate_gui_img(contour_image, height_results, width_results,
                                                     height_rect_params, width_rect_params, scan_error)
                # inside run_frame_processor, after you compute callback_img...
                if scan_error is None:
                    # CHANGED (Event Bus): publish results + image as events.
                    # This lets DataManager/GUI (and any future subscribers) react without FrameProcessor
                    # needing to know about them or be wired via callbacks.
                    if self.event_bus is not None:
                        self.event_bus.publish(
                            Event(
                                event_type=EventType.PROCESSING_COMPLETE,
                                source="frame_processor",
                                data=self.most_recent_results,
                            )
                        )
                        self.event_bus.publish(
                            Event(
                                event_type=EventType.IMAGE_READY,
                                source="frame_processor",
                                data=callback_img,
                            )
                        )

                    # Back-compat: keep the original callbacks during migration. Once event-bus is proven,
                    # you can delete these callback calls and the corresponding "establish_*" methods.
                    self._safe_call(self.data_to_data_manager_callback, self.most_recent_results)  # was direct call
                    # self._safe_call(self.data_to_server_callback, self.most_recent_results)        # was direct call
                    self._safe_call(self.image_to_gui_callback, callback_img)                      # was direct call
                else:
                    # CHANGED (Event Bus): publish scan errors for GUI to display (and for logging).
                    if self.event_bus is not None:
                        self.event_bus.publish(
                            Event(
                                event_type=EventType.SCAN_ERROR,
                                source="frame_processor",
                                data="SCAN ERROR: " + scan_error_message,
                            )
                        )

                    # Back-compat: original callback.
                    self._safe_call(self.gui_error_callback, 'SCAN ERROR: ' + scan_error_message)  # was direct call



                reading_number += 1
        print('exiting frame processor')


    def generate_gui_img(self, contour_image, height_results, width_results,  height_shell_params, width_shell_params, scan_error):

        def create_captioned_image(message: str):
            # Create a white image
            height, width = 1242, 1640  # OpenCV uses (height, width)
            image = np.ones((height, width, 3), dtype=np.uint8) * 255

            # Define font and text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            font_thickness = 3
            color = (0, 0, 255)  # Red color in BGR format

            # Get text size to calculate position
            text_size = cv2.getTextSize(message, font, font_scale, font_thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2

            # Put the text on the image
            cv2.putText(image, message, (text_x, text_y), font, font_scale, color, font_thickness, cv2.LINE_AA)

            return image

        def create_contour_image(contour_image, height_shell_params, width_shell_params):
            for height_shell in height_shell_params:
                cv2.rectangle(contour_image, (int(height_shell[0][0]), int(height_shell[0][1])),
                                              (int(height_shell[1][0]), int(height_shell[1][1])), (0, 255, 0), 20)
            for width_shell in width_shell_params:
                cv2.rectangle(contour_image, (int(width_shell[0][1]), int(width_shell[0][0])),
                              (int(width_shell[1][1]), int(width_shell[1][0])), (255, 0, 0), 20)
            return contour_image

        def append_results_to_image(image, height_results, width_results, height_rect_info, width_rect_info, wrow, wcolumn,
                                    w_column_end):
            """
            Appends the profiling results to an image by adding a white column and row, and placing results accordingly.
            The white column has a thickness of 100px and the white row has a thickness of 100px.
            """

            # Step 1: Create a larger image with an additional 100px white column and row
            original_height, original_width, _ = image.shape
            new_width = original_width + wcolumn + w_column_end# Add a white column on the left (100px wide)
            new_height = original_height + wrow  # Add a white row at the bottom (100px tall)
            expanded_image = np.ones((new_height, new_width, 3), dtype=np.uint8) * 255
            expanded_image[:new_height-wrow, wcolumn:original_width + wcolumn] = image
            # Step 2: Place height_results along the bottom row (white row)
            for i, (height, rect_info) in enumerate(zip(height_results, height_rect_info)):
                _, y_min = rect_info[0]  # Get the y-coordinate from height_rect_info

                y_coord = (rect_info[0][0] + rect_info[1][0]) // 2
                # Place text in the middle of the bottom white row
                cv2.putText(
                    expanded_image,
                    str(height),
                    (y_coord + wcolumn//2 + 100, new_height - wrow // 4),  # Text in the middle of the bottom white row
                    cv2.FONT_HERSHEY_SIMPLEX,
                    8,
                    (0, 0, 0),
                    20,
                    cv2.LINE_AA
                )

            # Step 3: Place width_results along the left column (white column)
            for i, (width, rect_info) in enumerate(zip(width_results, width_rect_info)):
                x_min, y_min = rect_info[0]  # Get the x-coordinate from width_rect_info
                x_coord = (rect_info[0][0] + rect_info[1][0]) // 2
                # Place text in the middle of the left white column
                cv2.putText(
                    expanded_image,
                    str(width),
                    (wcolumn // 4, x_coord + wrow // 4),  # Text in the middle of the left white column
                    cv2.FONT_HERSHEY_SIMPLEX,
                    8,
                    (0, 0, 0),
                    20,
                    cv2.LINE_AA
                )

            # Return the modified image
            return expanded_image

        if scan_error is not None:
            print('SCANNING ERROR DETECTED')
            error_msg = "Scanning Error Detected"
            ret_img = create_captioned_image(error_msg)
        else:
            ret_img = create_contour_image(contour_image, height_shell_params, width_shell_params)
            ret_img = append_results_to_image(ret_img, height_results, width_results,  height_shell_params, width_shell_params,
                                              300, 1000, w_column_end=100)
        return ret_img

    def check_for_product_error(self, processed_data):
        # check scanned data against standards for product-related issues
        ret_heights, ret_widths, ret_area = processed_data

        product_error_code = 'None'
        avg_heights = sum(ret_heights)/len(ret_heights)
        avg_widths = sum(ret_widths)/len(ret_widths)
        if (ret_heights[0] < self.gui_settings_data['minimum_height_value'] or
                ret_heights[len(ret_heights)-1] < self.gui_settings_data['minimum_height_value'] or
                ret_area < self.gui_settings_data['minimum_surface_area_value']):
            product_error_code = "Non-Compliant Product Detected"

        elif (avg_heights > self.gui_settings_data['maximum_height_value'] or
              ret_area > self.gui_settings_data['maximum_surface_area_value']):
            product_error_code = "Low Efficiency Detected"

        elif ((True in [q < 0.75 for q in [x / avg_widths for x in ret_widths]]) or
              (True in [q < 0.75 for q in [x / avg_heights for x in ret_heights]])):
            product_error_code = "Unexpected Shape Detected"

        elif abs(ret_heights[0] - ret_heights[len(ret_heights)-1]) > 1.5:
            product_error_code = "Height Imbalance Detected"
        return product_error_code

    def _safe_call(self, fn, *args, **kwargs):
        """Call a callback if it's callable; swallow exceptions to keep thread alive."""
        if callable(fn):
            try:
                fn(*args, **kwargs)
            except Exception as e:
                # Best-effort surface to GUI if available; else just print
                if callable(self.gui_error_callback):
                    try:
                        self.gui_error_callback(f"Callback error: {e!r}")
                    except Exception:
                        pass
                else:
                    print(f"[FrameProcessor] Callback error: {e!r}")


if __name__ == "__main__":
    pass
    # import os
    # from camera_parameters import monocle_parameters
    # from camera_class import CameraClass
    # import cv2.aruco as aruco
    # c = CameraClass(monocle_parameters)
    # f = FrameProcessor(monocle_parameters)
    # f.unit_test = True
    # images = []
    # #troubleshoot_image_path = '../training_images'
    # #for directory in os.listdir(troubleshoot_image_path):
    #     #if directory == '01-15-25-17-21-47':
    #         #dirpath = troubleshoot_image_path + '/' + directory
    #         #for filename in os.listdir(dirpath):
    #          #   filepath = dirpath + '/' + filename
    #             # Check if the file is an image
    #           #  if filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
    #                 # Read the image
    #            #     image = cv2.imread(filepath)
    #             #    if image is not None:  # Ensure the image was read successfully
    #              #       images.append(image)
    #                # else:
    #               #      print(f"Warning: Unable to read image {filepath}")
    # corner_list = []
    # for i in range(len(images)):
    #     image = images[i]
    #     gray_frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #
    #     # Detect Aruco markers
    #
    #     corners, ids, _ = aruco.detectMarkers(
    #         gray_frame,
    #         c.aruco_dict,
    #         c.camera_matrix,
    #         c.camera_distortion,
    #         parameters=c.aruco_parameters,
    #     )
    #     if ids is not None and 3 in ids:  # Check for specific marker ID
    #         corner_list.append(corners)
    #
    # corners, top_center = c.get_most_common_corners_and_top_center(corner_list)
    # for i in range(len(images)):
    #     images[i] = c.crop_image(images[i], corners)
    #     f.process_frame(images[i], top_center)






