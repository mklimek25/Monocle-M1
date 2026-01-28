import numpy as np
import os


def is_raspberry_pi():
    try:
        with open("/proc/cpuinfo", "r") as cpuinfo:
            for line in cpuinfo:
                if "Raspberry Pi" in line or "BCM" in line:
                    return True
    except FileNotFoundError:
        return False
    return False


if is_raspberry_pi():
    print("This system is running on a Raspberry Pi.")
    video_test = False
    gpio_pins = True
    directory = '/home/pi/monocle-official/src/prototype_files/'
    collection_count = 35
    hidden_reset_count = 50
    import RPi.GPIO as GPIO
    # Setup GPIO mode
    GPIO.setmode(GPIO.BCM)  # Use BCM pin numbering

    # Setup GPIO pin as output
    GPIO.setup(5, GPIO.OUT)  # Example: Pin 5 as output
    # Turn on the pin (high)
    GPIO.output(5, GPIO.HIGH)
    GPIO.setup(21, GPIO.OUT)  # Example: Pin 21 as output
    # Turn on the pin (high)
    GPIO.output(21, GPIO.HIGH)
    usb_dir_path = '/media/pi/69D5-7A09'
    test_mode = False
else:
    print("This system is NOT running on a Raspberry Pi.")
    video_test = True
    gpio_pins = False
    directory = '../prototype_files/'
    collection_count = 7
    hidden_reset_count = 7
    from pathlib import Path
    usb_dir_path = r"D:\\"
    test_mode = True


with open(directory + 'camera_cal.npy', 'rb') as f:
    camera_matrix = np.load(f)
    camera_distortion = np.load(f)

monocle_parameters = {
    # GENERAL PARAMS
    'general_params': {
        'directory': directory,
        'video_test': video_test,
        'gpio_pins': gpio_pins,
        'test_mode': test_mode
    },
    # CAMERA Unit Params:
    'camera_params': {
        'video_name': '../../Training_Film/1-27-height-imb-vid.avi',
        'collect_images': False,
        'camera_matrix': camera_matrix,  # camera calibration result
        'camera_distortion': camera_distortion,  # camera calibration result
        'required_found_count': collection_count,

        'hidden_reset_count': hidden_reset_count,
        'collection_count': 11,
        'collection_timespan': 0,
        # Aruco Detection Parameters
        'adaptiveThreshWinSizeMin': 3,    # Reduce min window size for adaptive thresholding
        'adaptiveThreshWinSizeMax': 23,   # Allow for larger window size
        'adaptiveThreshWinSizeStep': 10,  # Step size for the adaptive thresholding
        'adaptiveThreshConstant': 7,     # Adjust constant subtracted from mean (increases tolerance to lighting)
        'minMarkerPerimeterRate': 0.01,  # Lower to detect smaller markers
        'maxMarkerPerimeterRate': 4.0,   # Increase to allow for larger markers
        'polygonalApproxAccuracyRate': 0.03,  # Adjust to allow imperfect marker shapes
        'minCornerDistanceRate': 0.05,       # Decrease for markers that are close together
        'rotation_angle': 0,
        'set_controls': False,
        'AeEnable': True,
        'ExposureTimeRange': (20000, 1000000),
        'AnalogueGainRange': (1.0, 4.0)



    },
    # __________________________
    # Frame Processor Params:
    'frame_proccessor_params': {
        'check_points': [(500, 850), (1000, 750)],  # contour must contain these two points to be considered
        'color_tolerance_high': 60,  # color tolerance for inRange function on preprocessing
        'color_tolerance_low': 80,  # color tolerance for inRange function on preprocessing
        'investigate_processing': False,  # WILL initialize the image_processing_analysis when written
        'rotation_angle': -2.25,  # CHECK THIS, constant rotation images will take
        'hDepthRange': (25, 150),
        'minimum_neck_thickness': 200,  # minimum thickness to be defined as a potential neck - total guess
        'maximum_neck_length': 5,  # minimum length of neck before trim - total guess
        'sobel_kernel_size': -1,  # The kernel size input on the kernel function
        'gauss_blur_kernel_size': 5,  # the kernel size of the gaussian blur
        'closing_kernel_size': 3,
        'closing_iterations': 2,  # determines the number of morph_closing iterations the processing will take
        'final_threshold_param': 55,  # the cutoff point for thresholding the sobel output
        'contour_is_simple': False,  # determines whether is_simple will be used in contour formation
        'required_contour_area': 300000,  # surface area needed to be considered the contour
        # Measurement Scalars
        'init_x_scalar': 0,
        'init_y_scalar': 0,
        'init_top_to_bottom_scalar': 0,
        'init_left_to_right_scalar': 0.3,
        'numb_vertical_measurements': 5,
        'numb_horizontal_measurements': 3,
        'height_shell_cut_param': 0.04,
        'width_shell_cut_param': 0.1,
        'shell_pixel_width': 150,
        'image_results_params': (300, 200)

    },
    'data_frame_parameters': {
        'usb_dir_path': usb_dir_path,
        'output_folder': 'output_data',
        'df_name': 'test.db',
        'table_name': 'monocle_data',
        'data_columns': ["timestamp", "product", "reading_number",
                         "width_results", "average_width", "height_results", "average_height", "area_val",
                         "ret_yield", "scan_error", "product_error"],
        'analysis_args': {'reading_number': 'COUNT',
                          'average_width': 'AVERAGE',
                          'average_height': 'AVERAGE',
                          'area_val': 'AVERAGE',
                          'product_error': 'COUNTA'},




        'translation': {'Sides': ('height_results', ['START', 'END']),
                        'Center Height': ('height_results', ['MIDDLE']),
                        'Avg Heights': ('average_height', None),
                        'Avg Widths': ('average_width', None),
                        'Surface Area': ('area_val', None),
                        'Yield': ('ret_yield', None),
                        'Errors': ('product_error', None)
                        },
        'sender_email_username': 'kingspan.miami.vision.system@gmail.com',
        'sender_email_passcode': 'mujj keze islt kzol',
        'recipient_email_passcode': ['mark.d.klimek@gmail.com', 'pilar.perez@kingspan.com', 'rafael.alvarez@kingspan.com']
    },
    'security': {
        'exit_credentials': {
            'username': 'markklimek',
            'password': 'm345ur3m3nt'
        }
    },

    'gui_manager_parameters': {
        'exit_credentials': {'username': 'admin', 'password': 'password123'},
        'frame_names': [
            ("root", "error_frame", 4, 0, 4, 10),
            ("root", "home_frame", 2, 0, 8, 10),
            ("root", "density_frame", 2, 0, 8, 8),
            ("root", "historian_frame", 2, 0, 8, 8),
            ("root", "calibration_frame", 2, 0, 8, 8),
            ("root", "run_analysis_frame", 2, 0, 8, 8),
            ("root", "wifi_frame", 2, 0, 8, 8),
            ("root", "exit_frame", 2, 0, 8, 8),
            ("root", "settings_frame", 2, 0, 8, 8),
            ('home_frame', 'display_image_figure', 1, 1, 8,  8),
            ('historian_frame', 'center_height_figure', 1, 0, 8, 7),
            ('historian_frame', 'average_height_figure', 1, 0, 8, 7),
            ('historian_frame', 'average_width_figure', 1, 0, 8, 7),
            ('historian_frame', 'surface_area_figure', 1, 0, 8, 7),
            ('historian_frame', 'yield_figure', 1, 0, 8, 7),
            ('historian_frame', 'scan_error_figure', 1, 0, 8, 7),
            ('historian_frame', 'product_error_figure', 1, 0, 8, 7),
            ('historian_frame', 'side_height_figure', 1, 0, 8, 7),

        ],
        'label_names': {  # label_name, gridx, gridy, rowspan, columnspan, stringval, fontsize
            'root':[('monocle_title', 0, 0, 1, 8, 'MONOCLE', 'title_fontsize')],
            'exit_frame': [('exit_title_label', 0, 0, 1, 8, 'EXIT', 'title_fontsize')],
            'home_frame': [('home_title_label', 0, 0, 1, 10, 'HOME', 'title_fontsize'),
                           ('avgWidth_title', 1, 0, 1, 1, 'Average Width:', 'primary_fontsize'),
                           ('average_width', 2, 0, 1, 1, '', 'primary_fontsize'),
                           ('avgHeight_title', 3, 0, 1, 1, 'Average Height:', 'primary_fontsize'),
                           ('average_height', 4, 0, 1, 1, '', 'primary_fontsize'),
                           ('area_title', 5, 0, 1, 1, 'Area:', 'primary_fontsize'),
                           ('area_val', 6, 0, 1, 1, '', 'primary_fontsize'),
                           ('yield_title', 7, 0, 1, 1, 'Yield:', 'primary_fontsize'),
                           ('ret_yield', 8, 0, 1, 1, '', 'primary_fontsize')],
            'historian_frame': [('historian_title_label', 0, 0, 1, 8, 'HISTORIAN', 'title_fontsize')],
            'density_frame': [('density_title_label', 0, 0, 1, 8, 'DENSITY CALCULATION', 'title_fontsize'),
                              ('flow_rate_var_label', 2, 6, 1, 4, '', 'title_fontsize'),
                              ('line_speed_var_label', 4, 6, 1, 4, '', 'title_fontsize'),
                              ('density_result_label', 6, 6, 2, 4, '', 'title_fontsize')],
            'calibration_frame': [
                ('calibration_title_label', 0, 0, 1, 10, 'CALIBRATION', 'title_fontsize'),
                ('x_b', 1, 0, 2, 4, 'X BASE PARAM', 'title_fontsize'),
                ('y_b', 3, 0, 2, 4, 'Y BASE PARAM', 'title_fontsize'),
                ('left_to_right_scalar', 5, 0, 2, 4, 'HEIGHT SCALAR', 'title_fontsize'),
                ('top_to_bottom_scalar', 7, 0, 2, 4, 'WIDTH SCALAR', 'title_fontsize'),
            ],

            'settings_frame': [('settings_title_label', 0, 0, 1, 8, 'SETTINGS', 'title_fontsize'),
                               ('language_settings_label', 1, 0, 1, 7, 'LANGUAGE:', 'title_fontsize'),
                               ('product_type_label', 2, 0, 1, 7, 'PRODUCT TYPE:', 'title_fontsize'),
                               ('email_alerts_label', 3, 0, 1, 7, 'EMAIL ALERTS:', 'title_fontsize'),
                               ('minimum_height_label', 5, 0, 1, 7, 'MINIMUM HEIGHT:', 'primary_fontsize'),
                               ('minimum_height_value', 6, 0, 1, 7, '', 'primary_fontsize'),
                               ('maximum_height_label', 5, 7, 1, 1, 'MAXIMUM HEIGHT:', 'primary_fontsize'),
                               ('maximum_height_value', 6, 7, 1, 1, '', 'primary_fontsize'),
                               ('minimum_density_label', 7, 0, 1, 7, 'MINIMUM DENSITY:', 'primary_fontsize'),
                               ('minimum_density_value', 8, 0, 1, 7, '', 'primary_fontsize'),
                               ('maximum_density_label', 7, 7, 1, 1, 'MAXIMUM DENSITY:', 'primary_fontsize'),
                               ('maximum_density_value', 8, 7, 1, 1, '', 'primary_fontsize'),
                               ('minimum_surface_area_label', 9, 0, 1, 7, 'MINIMUM SURFACE AREA:', 'primary_fontsize'),
                               ('minimum_surface_area_value', 10, 0, 1, 7, '', 'primary_fontsize'),
                               ('maximum_surface_area_label', 9, 7, 1, 1, 'MAXIMUM SURFACE AREA:', 'primary_fontsize'),
                               ('maximum_surface_area_value', 10, 7, 1, 1, '', 'primary_fontsize'),
                               ],
            'wifi_frame': [('wifi_title_label', 0, 0, 1, 8, 'WIFI', 'title_fontsize')],
            'error_frame': [('error_frame', 0, 0, 8, 8, "", 'secondary_fontsize')]

        },

        'navigation_button_names': {'home_frame': ('HOME', 1, 0, 1, 2, 'title_fontsize'),
                                    'historian_frame': ("HISTORIAN", 1, 2, 1, 2, 'title_fontsize'),
                                    'density_frame': ('DENSITY', 1, 4, 1, 2, 'title_fontsize'),
                                    'settings_frame': ('SETTINGS', 1, 6, 1, 2, 'title_fontsize'),
                                    # 'calibration_frame': ('CALIBRATION', 11, 4, 1, 2, 'title_fontsize'),
                                    'exit_frame': ('EXIT', 11, 4, 1, 2, 'title_fontsize'),
                                    'wifi_frame': ('WIFI', 11, 2, 1, 2, 'title_fontsize')
                                    },
        'function_button_names': {
            'root': [
                ('call_start_button_callback','START SCAN', 11, 0, 1, 2, 'title_fontsize'),
                ('stop_scanning', 'TURN OFF', 11, 6, 1, 2, 'title_fontsize'),
            ],
            'home_frame': [],
            'historian_frame': [],
            'density_frame': [
                ('set_var_to_flowrate', 'FLOW RATE', 1, 6, 1, 2, 'primary_fontsize'),
                ('set_var_to_line_speed', 'LINE SPEED', 3, 6, 1, 2, 'primary_fontsize'),
                ('calculate_density', 'CALCULATE\nDENSITY', 5, 6, 1, 2, 'primary_fontsize'),
            ],
            'calibration_frame': [('apply_tuning_settings', 'APPLY SETTINGS', 9, 0, 1, 10, 'title_fontsize')],
            'settings_frame': [
                ('apply_settings', 'APPLY SETTINGS', 4, 0, 1, 8, 'title_fontsize')
            ],

        },
        'calculator_button_names': {
            'density_frame': [
                ("1", 1, 0, 2, 2),
                ("2", 1, 2, 2, 2), ("3", 1, 4, 2, 2),
                ("4", 3, 0, 2, 2), ("5", 3, 2, 2, 2),
                ("6", 3, 4, 2, 2), ("7", 5, 0, 2, 2),
                ("8", 5, 2, 2, 2), ("9", 5, 4, 2, 2),
                (".", 7, 0, 1, 2), ("0", 7, 2, 1, 2),
                ("X", 7, 4, 1, 2)
            ],
        },

        'selection_lists': {  # Shows every selection category along with its assosiated parameters
            # product parameters: min height, max height, min density, max density, min SA, max SA
            'product': {'ISO-C1 2.0 STANDARD WIDE': (25.75, 28.325, 2.04, 2.08, 1313.0, 1641.25),
                        'ISO-C1 2.0 30" HEIGHT': (30.00, 32.5, 2.04, 2.08, 1515.0, 1909.53),
                        'ISO-C1 2.5': (25.75, 28.325, 2.54, 2.58, 1313.0, 1641.25),
                        'ISO HT': (25.75, 28.325, 2.54, 2.58, 1313.0, 1641.25),
                        'ISO-C1 3.0': (18, 24, 3.1, 3.25, 1010, 1252.5),
                        'ISO-C1 4.0': (16, 20, 4.1, 4.25, 4808.0, 1010.0),
                        'ISO-C1 6.0': (12, 15, 6.1, 6.25, 606.0, 757.5),
                        },
        },
        'historian_display_list_data': {
            # 'Sides', 'Center Height', 'Avg Heights', 'Avg Widths', 'Surface Area', 'Yield', 'Errors'
            'Sides': {
                'figure': 'side_height_figure',
                'yaxis': ('Left Side', 'Right Side'),
                'db_tag': 'height_results',
                'index': ('START', 'END')
            },
            'Center Height': {
                'figure': 'center_height_figure',
                'yaxis': 'Center Height',
                'db_tag': 'height_results',
                'index': 'MIDDLE'
            },
            'Avg Heights': {
                'figure': 'average_height_figure',
                'yaxis': 'Avg Heights',
                'db_tag': 'average_height',
                'index': None
            },
            'Avg Widths': {
                'figure': 'average_width_figure',
                'yaxis': 'Avg Widths',
                'db_tag': 'average_width',
                'index': None
            },
            'Surface Area': {
                'figure': 'surface_area_figure',
                'yaxis': 'Surface Area',
                'db_tag': 'area_val',
                'index': None
            },
            'Yield': {'figure': 'yield_figure',
                      'yaxis': 'Yield',
                      'db_tag': 'ret_yield',
                      'index': None
                      },
            'Scanning Errors': {'figure': 'scan_error_figure',
                                'yaxis': 'Errors',
                                'db_tag': 'scan_error',
                                'index': None
                                },
            'Product Errors': {'figure': 'product_error_figure',
                               'yaxis': 'Errors',
                               'db_tag': 'product_error',
                               'index': None
                               }
        },
        'title_fontsize': 35,
        'primary_fontsize': 25,
        'secondary_fontsize': 20,
        'tertiary_fontsize': 15,
        'tune_param_fontsize': 40,
        'historian_param_fontsize': 40,
        'x_b': 0,
        'y_b': 0,
        'top_to_bottom_scalar': 0.00,
        'left_to_right_scalar': 0,
        'email_alerts_value': None,
        'language_setting_value': None,
        'minimum_height_value': None,
        'maximum_height_value': None,
        'minimum_density_value': None,
        'maximum_density_value': None,
        'minimum_surface_area_value': None,
        'maximum_surface_area_value': None,
        'historian_data_labels': ['Sides', 'Center Height', 'Avg Heights', 'Avg Widths', 'Surface Area', 'Yield',
                                  'Scanning Errors', 'Product Errors'],
        'language_setting': ['English', 'Spanish'],
        'email_alerts': ['True', 'False'],
        'spinbox_names': {
            'home_frame': [],
            'historian_frame': [('historian_data_labels', '_show_figure', 1, 7, 1, 1, 12, 8, 'historian_param_fontsize')],
            'density_frame': [],
            'calibration_frame': [('x_b', None, 1, 4, 2, 4, 25, 15, 'tune_param_fontsize'),
                                  ('y_b', None, 3, 4, 2, 4, 25, 15, 'tune_param_fontsize'),
                                  ('left_to_right_scalar', None, 5, 4, 2, 4, 25, 15, 'tune_param_fontsize'),
                                  ('top_to_bottom_scalar', None, 7, 4, 2, 4, 25, 15, 'tune_param_fontsize'),],
            'settings_frame': [('language_setting', None, 1, 7, 1, 3, 25, 15, 'title_fontsize'),
                               ('product', None, 2, 7, 1, 3, 25, 15, 'title_fontsize'),
                               ('email_alerts', None, 3, 7, 1, 3, 25, 15, 'title_fontsize')]

        },
        'translation': {'Sides': ('height_results', 'side_height_figure', ['START', 'END']),
                        'Center Height': ('height_results', 'center_height_figure', [('MIDDLE')]),
                        'Avg Heights': ('average_height', 'average_height_figure', None),
                        'Avg Widths': ('average_width', 'average_width_figure', None),
                        'Surface Area': ('area_val', 'surface_area_figure', None),
                        'Yield': ('ret_yield', 'yield_figure', None),
                        'Scanning Errors': ('scan_error', 'scan_error_figure', None),
                        'Product Errors': ('product_error', 'product_error_figure', None)
                        },
        'figure_names': {
            'display_image_figure': {
                'row': 0,
                'column': 0,
                'rowspan': 10,
                'columnspan': 10,
                'function': 'display_image'},
            'side_height_figure': {
                'row': 0,
                'column': 0,
                'rowspan': 8,
                'columnspan': 7,
                'y1_label': 'Left Side',
                'y2_label': 'Right Side',
                'df_column_name': 'height_results',
                'index': ('START', 'END'),
                'function': 'create_scatter_plot',
                'axis_title': 'Side Height [in]'},
            'center_height_figure': {
                'row': 0,
                'column': 0,
                'rowspan': 8,
                'columnspan': 7,
                'y1_label': 'Center Height',
                'y2_label': None,
                'df_column_name': 'height_results',
                'index': ('MIDDLE'),
                'function': 'create_scatter_plot',
                'axis_title': 'Center Height [in]'},
            'average_height_figure': {
                'row': 0,
                'column': 0,
                'rowspan': 8,
                'columnspan': 7,
                'y1_label': 'Average Height',
                'y2_label':  None,
                'df_column_name': 'average_height',
                'index': None,
                'function': 'create_scatter_plot',
                'axis_title': 'Average Height [in]'},
            'average_width_figure': {
                'row': 0,
                'column': 0,
                'rowspan': 8,
                'columnspan': 7,
                'y1_label': 'Average Width',  # not 'Average Height'
                'y2_label':  None,
                'df_column_name': 'average_width',
                'index': None,
                'function': 'create_scatter_plot',
                'axis_title': 'Average Width [in]'},
            'surface_area_figure': {
                'row': 0,
                'column': 0,
                'rowspan': 8,
                'columnspan': 7,
                'y1_label': 'Surface Area',
                'y2_label': None,
                'df_column_name': 'area_val',
                'index': None,
                'function': 'create_scatter_plot',
                'axis_title': 'Area [in^2]'},
            'yield_figure': {
                'row': 0,
                'column': 0,
                'rowspan': 8,
                'columnspan': 7,
                'y1_label': 'Yield',
                'y2_label': None,
                'df_column_name': 'ret_yield',
                'index': None,
                'function': 'create_scatter_plot',
                'axis_title': 'Yield [%]'},
            'scan_error_figure': {
                'row': 0,
                'column': 0,
                'rowspan': 8,
                'columnspan': 7,
                'y1_label': 'Scan Error',
                'y2_label': None,
                'df_column_name': 'scan_error',
                'index': None,
                'function': 'create_error_chart',
                'axis_title': 'Scanning Errors'},
            'product_error_figure': {
                'row': 0,
                'column': 0,
                'rowspan': 8,
                'columnspan': 7,
                'y1_label': 'Left Side',
                'y2_label': None,
                'df_column_name': 'product_error',
                'index': None,
                'function': 'create_error_chart',
                'axis_title': 'Product Errors'},
        },
        'listbox_names': {
            'home_frame': (),
            'historian_frame': ('historian_listbox', 2, 7, 6, 1),
            'density_frame': (),
            'calibration_frame': (),
            'settings_frame': ()
        },
    },
}
