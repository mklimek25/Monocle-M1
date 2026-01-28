import cv2
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import os
from frame_processor_class import FrameProcessor
from camera_class import CameraClass
from custom_spinbox_class import CustomSpinbox
from camera_parameters import monocle_parameters
troubleshoot_parameters = {
    'image_processing_list': ['image', 'preprocess', 'finding_contour_of_interest', 'measure_contour', 'refine_contour', 'profile', 'results'],
    'image_directory_path': 'training_images',
    'labels': ['x_vals', 'y_vals', 'area_val'],
    'figure_names': {
        'display_image_figure': {
            'row': 0,
            'column': 0,
            'rowspan': 10,
            'columnspan': 10,
            'function': 'display_image'},
}
}

class Troubleshooting():
    def __init__(self, params, frame_processor, camera_class):
        self.root = tk.Tk()
        self.params = params
        self.frames = {}
        self.buttons = {}
        self.labels = {}
        self.figures = {}
        self.spinboxes = {}
        self.directory_paths = None
        self.file_paths = None
        self.top_center = (777.31, 1074)
        self.frame_processor = frame_processor
        self.camera_class = camera_class
        self.info = None




    def collect_image_paths(self, filter_string):
        directory_paths = []
        filepaths = []
        troubleshoot_image_path = '../training_images/'
        for directory in os.listdir(troubleshoot_image_path):
            if filter_string is None or directory == filter_string:
  
                directory_paths.append(directory)
         
                        
                file_paths = []
                
                for filename in os.listdir(troubleshoot_image_path+'/'+directory):
                    # filepath = troubleshoot_image_path + directory + '/' + filename
                    file_paths.append(filename)
                    # filepath = os.path.join(troubleshoot_image_path, directory, filename)
                    # Check if the file is an image
                    # if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    #     # Read the image
                    #     image = cv2.imread(filepath)
                    #     if image is not None:  # Ensure the image was read successfully
                    #         images.append((filepath, image))
                    #     else:
                    #         print(f"Warning: Unable to read image {filepath}")
                filepaths = sorted(filepaths)
                filepaths.append(file_paths)

        return directory_paths, filepaths



    def display_image(self, canvas: FigureCanvasTkAgg, img=None):
        """
        Displays an image on a FigureCanvasTkAgg canvas.

        :param canvas: A FigureCanvasTkAgg object to display the image on.
        :param img: The image to be displayed (in BGR format, as used by OpenCV).
        """
        if img is None:
            return

        # Convert the image from BGR to RGB
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create a Matplotlib figure
        fig = canvas.figure
        fig.clf()  # Clear the figure to avoid overlap

        ax = fig.add_subplot(111)  # Add a subplot
        ax.imshow(img)  # Display the image
        ax.axis('off')  # Hide the axis
        canvas.draw()

    # Draw the updated figure on the canvas
    def setup_gui(self, no_of_images):
        for r in range(11):
            self.root.grid_rowconfigure(r, weight=1)
        for c in range(11):
            self.root.grid_columnconfigure(c, weight=1)
        j = 0
        for function in troubleshoot_parameters['image_processing_list']:
            frame = tk.Frame(self.root, bg="gray")
            self.frames[function] = frame
            frame.grid(row=2, column=0, rowspan=8, columnspan=8)
            if function == 'results':
                k = 0
                for label_name in troubleshoot_parameters['labels']:
                    label = tk.Label(master=self.frames['results'],
                                   text=label_name + ':',
                                   borderwidth=1,
                                   relief="solid",
                                   width=60,
                                   height=2,
                                   font=('Arial', 10),
                                   background='orange'
                                   )
                    label.grid(row=k*2, column=0, rowspan=2, columnspan=6)
                    self.labels[label_name] = label
                    k += 1

            button = tk.Button(master=self.root,
                               text=function,
                               command=lambda v=function: self._show_frame(v),
                               borderwidth=1,
                               relief="solid",
                               width=10,
                               height=2,
                               font=('Arial', 10),
                               background='orange'
                               )
            button.grid(row=j * 2, column=8, rowspan=2, columnspan=2)
            j += 1
            self.buttons[function] = button
        self._create_figures()
        directory_spinbox = CustomSpinbox(self.root, 0, [str(i) for i in range(len(self.directory_paths))],
                                          self.update_directory)

        filepath_spinbox = CustomSpinbox(self.root, 0, [str(i) for i in range(no_of_images)],
                                self.update_image)
        self.spinboxes['filepath_spinbox'] = filepath_spinbox
        directory_spinbox.grid(row=0, column=0, rowspan=1, columnspan=8)

        filepath_spinbox.grid(row=1, column=0, rowspan=1, columnspan=8)
        self.spinboxes['directory_spinbox'] = directory_spinbox


    def _create_figures(self):
        """
        Dynamically creates FigureCanvasTkAgg objects and embeds them into Tkinter frames.
        Uses parameters from self.params to configure the layout and properties.
        """
        figure_params = self.params["figure_names"]['display_image_figure']
        for figure_name in self.params['image_processing_list']:
            if figure_name != "results":
                print(figure_name)
                frame = self.frames[figure_name]
                figure_info = figure_params

                if len(figure_info) == 0:
                    pass
                else:
                    # Create a Matplotlib Figure
                    fig = Figure(figsize=(5, 4), dpi=100)
                    ax = fig.add_subplot(111)

                    # Call the provided plot function to populate the figure
                    plot_function = figure_info['function']
                    if hasattr(self, plot_function):
                        plot = getattr(self, plot_function)
                        if plot_function == 'display_image':
                            plot(ax, None)
                        else:
                            plot(ax, figure_name)
                    else:
                        pass

                    # Embed the figure into the Tkinter frame
                    canvas = FigureCanvasTkAgg(fig, master=frame)
                    canvas_widget = canvas.get_tk_widget()
                    canvas_widget.grid(row=figure_info['row'],
                                       column=figure_info['column'],
                                       rowspan=figure_info['rowspan'],
                                       columnspan=figure_info['columnspan'],
                                       padx=1, pady=1, sticky="nsew")

                    # Draw the figure on the canvas
                    canvas.draw()  # Ensure the figure is rendered

                    # Store the canvas object for later use
                    self.figures[figure_name] = canvas

    def update_directory(self, dir_var):
        var = int(dir_var)

        self.directory = self.directory_paths[var]
        self.frame_path = self.file_paths[var]
        if 'filepath_spinbox' in self.spinboxes.keys():
            self.update_image('0')

    def update_image(self, var):
        var = int(var)
        troubleshoot_image_path = '../training_images/'
        image_path = troubleshoot_image_path + self.directory + '/' + self.frame_path[var]
        print(image_path)
        image = cv2.imread(image_path)
        print(image.shape)

        print('updating image')

        print(f'var: {var}')
        
        self.display_image(self.figures['image'], image)
        # gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        preprocessed_img = self.frame_processor.preprocess(image)
        self.display_image(self.figures['preprocess'], preprocessed_img)
        contour_of_interest, h_depth, bc = self.frame_processor.finding_contour_of_interest(preprocessed_img,
                                                                               False, self.top_center)
        print(f'HDEPTH: {h_depth}')
        original_image_copy = image.copy()
        if contour_of_interest is not None:
            cv2.drawContours(original_image_copy, [contour_of_interest], -1, (0, 255, 0), 3)
            self.display_image(self.figures['finding_contour_of_interest'], original_image_copy)
            reshaped_contour, width, height = self.frame_processor.measure_contour(contour_of_interest, h_depth, bc[0], bc[1])
            reshaped_img = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.fillPoly(reshaped_img, [reshaped_contour], color=255)
            self.display_image(self.figures['measure_contour'], reshaped_img)
            refined_img = self.frame_processor.refine_contour(reshaped_contour)
            self.display_image(self.figures['refine_contour'], refined_img)

            print(f'width: {width}, height: {height}')
            x_vals, y_vals, area_val, height_shell_params, width_shell_params = self.frame_processor.profile(black_image=refined_img,
                                                                                        num_height_sections=5,
                                                                                        num_width_sections=3,
                                                                                        shell_pixel_width=100)
            x_vals = [round(x/100, 2) for x in x_vals]
            y_vals = [round(y/100, 2) for y in y_vals]
            generated_image = self.frame_processor.generate_gui_img(refined_img, y_vals, x_vals, height_shell_params, width_shell_params, None)
            self.display_image(self.figures['profile'], generated_image)
            self.labels['x_vals'].config(text='x_vals: ' + str(x_vals))
            self.labels['y_vals'].config(text='y_vals: ' + str(y_vals))
            self.labels['area_val'].config(text='area_val: ' + str(area_val))
        self.root.update()

    def _show_frame(self, frame_name):

        """Brings the specified frame to the front."""
        frame = self.frames.get(frame_name)
        if frame:
            frame.tkraise()
        else:
            print(f"Frame {frame_name} does not exist!")


    def run_troubleshooting_center(self):
        self.directory_paths, self.file_paths = self.collect_image_paths('01-26-25-18-04-57')
        self.setup_gui(10)
        self.root.mainloop()




if __name__ == "__main__":
    frame_processor = FrameProcessor(monocle_parameters)
    camera_class = CameraClass(monocle_parameters)
    troubleshooting_center = Troubleshooting(troubleshoot_parameters, frame_processor, camera_class)
    troubleshooting_center.run_troubleshooting_center()
