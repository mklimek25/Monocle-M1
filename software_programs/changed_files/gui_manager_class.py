import matplotlib.pyplot as plt
import tkinter as tk
from custom_spinbox_class import CustomSpinbox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime
from tkinter import font
import cv2
import subprocess
import shutil
from tkinter import ttk

# CHANGED (Event Bus): GUI subscribes to IMAGE_READY and SCAN_ERROR so FrameProcessor doesn't call GUI directly.
# IMPORTANT: Tkinter must be updated on the main thread, so handlers use `root.after(0, ...)`.
from event_bus import EventBus, Event, EventType


class GUIManager:
    def __init__(self, root, params, event_bus: EventBus | None = None):
        """
        Initialize the GUIManager.

        Args:
            root (tk.Tk): Root Tkinter window.
        """
        self.root = root
        self.params = params['gui_manager_parameters']

        # CHANGED (Event Bus): store bus + subscribe to events we care about.
        self.event_bus = event_bus
        if self.event_bus is not None:
            self.event_bus.subscribe(EventType.IMAGE_READY, self._on_image_ready)
            self.event_bus.subscribe(EventType.SCAN_ERROR, self._on_scan_error)
        self.exit_user_var = tk.StringVar()
        self.exit_pass_var = tk.StringVar()
        self.home_fig, self.home_ax = plt.subplots()
        self.scat_fig, self.scat_ax = plt.subplots()
        self.frames = {}
        self.labels = {}
        self.buttons = {}
        self.figures = {}
        self.spinboxes = {}
        self.variables = {}
        self.listboxes = {}
        self.density_append_val = 0
        self.density_frame_var = True
        self.flow_rate_label = ""
        self.line_speed_label = ""
        self.tuning_parameter = 1
        self.density_parameter = 1
        self.hist_spinbox_var = 'Sides'
        self.spinbox_callback = None
        self.stop_button_callback = None
        self.stop_operation = False
        self.wifi_supported = shutil.which("nmcli") is not None
        self.wifi_tree = None
        self.wifi_password_var = tk.StringVar()

    # CHANGED (Event Bus): Event handlers. These are small wrappers that jump back to Tk's thread.
    def _on_image_ready(self, event: Event) -> None:
        """
        Receive processed image from FrameProcessor.

        Why `root.after(0, ...)`:
        - FrameProcessor publishes from a background thread.
        - Tkinter widgets must only be touched on the main thread.
        """
        img = event.data
        try:
            self.root.after(0, self.receive_img, img)
        except Exception as e:
            # Fallback: print; avoid raising in event handlers
            print(f"[GUIManager] failed scheduling image update: {e!r}")

    def _on_scan_error(self, event: Event) -> None:
        """Receive scan error message from FrameProcessor and display it in the GUI."""
        msg = event.data
        try:
            self.root.after(0, self.raise_error, msg)
        except Exception as e:
            print(f"[GUIManager] failed scheduling error display: {e!r}")


    def initiate_tuning_parameter_callback(self, callback):
        self.tuning_parameter_callback = callback

    def receive_error_callback(self, error_message):
        self.raise_error(error_message)

    def establish_hist_spinbox_callback(self, spinbox_callback):
        self.hist_data_callback = spinbox_callback

    def establish_start_button_callback(self, start_button_callback):
        self.start_button_callback = start_button_callback

    def establish_stop_button_callback(self, stop_button_callback):
        self.stop_button_callback = stop_button_callback

    def call_start_button_callback(self):
        self.start_button_callback(True)

    def _initialize_root(self):
        """Initializes the root window."""
        self.root.title("Tkinter Grid Interface")
        w, h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+0+0")

    def _create_frames(self):
        """Creates all frames for the interface."""
        nav_frame_names = self.params["frame_names"]
        for master, name, row, col, rspan, cspan in nav_frame_names:
            if master == 'root':
                if name == 'error_frame':
                    frame = tk.Frame(self.root, bg="white")
                else:
                    frame = tk.Frame(self.root, bg="gray")
            else:
                frame = tk.Frame(self.frames[master], bg="gray")
            self.frames[name] = frame
            frame.grid(row=row, column=col, rowspan=rspan, columnspan=cspan, sticky="nsew")
            for i in range(rspan):
                frame.grid_rowconfigure(i, weight=1)
            for j in range(cspan):
                frame.grid_columnconfigure(j, weight=1)


    def _create_labels(self):
        label_info = self.params["label_names"]

        frame_names = [n[1] for n in self.params['frame_names']]

        label_keys = label_info.keys()
        for key in label_keys:
            if key == "root":
                frame = self.root
            elif key in frame_names:
                frame = self.frames[key]
            else:
                raise ValueError(f'key {key} not found in frame names')

            label_info = self.params['label_names'][key]

            for label_tag, row, col, rspan, cspan, label, fontsize in label_info:
                fontsize = self.params[fontsize]
                label_object = tk.Label(master=frame, text=label, borderwidth=1, relief="solid", width=10, height=2,
                                        font=('Arial', fontsize), background='gray')
                label_object.grid(row=row, column=col, rowspan=rspan,
                                  columnspan=cspan, padx=1, pady=1, sticky="nsew")
                self.labels[label_tag] = label_object

    def _create_directory_buttons(self):
        """
        Replaces the label creation logic with button creation.
        Each button will have an associated command.
        """
        nav_frame_names = self.params["frame_names"]  # Retrieve frame names
        names = [f[1] for f in nav_frame_names]  # Extract names from tuples or lists

        for name in names:
            if name not in self.params['navigation_button_names'].keys():
                pass
            else:
                button_info = self.params['navigation_button_names'][name]  # Retrieve button-specific info
                text, row, col, rspan, cspan, fontsize = button_info
                fontsize = self.params[fontsize]  # Get the font size from params
                button_object = tk.Button(
                    master=self.root,
                    text=text,
                    command=lambda v=name: self.show_frame(v),  # Pass the command directly
                    borderwidth=1,
                    relief="solid",
                    width=10,
                    height=2,
                    font=('Arial', fontsize),
                    background='orange'
                )
                button_object.grid(
                    row=row,
                    column=col,
                    rowspan=rspan,
                    columnspan=cspan,
                    padx=1,
                    pady=1,
                    sticky="nsew"
                )
                self.buttons[name] = button_object  # Store button in a dictionary for later use


    def _create_function_buttons(self):
        data = self.params['function_button_names']
        names =[f[1] for f in self.params['frame_names']]

        names.append('root')
        for name in names:
            if name in data.keys():
                if name == 'root':
                    master = self.root

                else:
                    master = self.frames[name]
                buttons = data[name]
                for button in buttons:
                    label, text, row, col, rspan, cspan, fontsize = button
                    fontsize = self.params[fontsize]  # Get the font size from params
                    if hasattr(self, label):
                        command = getattr(self, label)
                        button_object = tk.Button(
                            master=master,
                            text=text,
                            command=command,  # Pass the command directly
                            borderwidth=1,
                            relief="solid",
                            width=10,
                            height=2,
                            font=('Arial', fontsize),
                            background='orange'
                        )
                        button_object.grid(
                            row=row,
                            column=col,
                            rowspan=rspan,
                            columnspan=cspan,
                            padx=1,
                            pady=1,
                            sticky="nsew")
                    else:
                        pass

    def _create_spinbox_objects(self):
        frame_names = list(self.params['spinbox_names'].keys())
        for name in frame_names:
            master = self.frames[name]
            spinbox_data = self.params['spinbox_names'][name]
            if len(spinbox_data) > 0:
                for label, callback, row, col, rspan, cspan, entry_width, button_width, fontsize in spinbox_data:
                    fontsize = self.params[fontsize]
                    if label != 'product':
                        var = self.params[label]
                        options = var
                    else:
                        var = self.params['selection_lists'][label]
                        options = list(var.keys())
                    if callback:
                        if hasattr(self, callback):
                            callback = getattr(self, callback)
                        else:
                            callback = None

                    spinbox = CustomSpinbox(master, var, callback_var=callback, options=options, spinbox_width=entry_width,
                                            button_width=button_width, fontsize=fontsize)
                    spinbox.grid(row=row,
                                 column=col,
                                 rowspan=rspan,
                                 columnspan=cspan,
                                 padx=1,
                                 pady=1,
                                 sticky="nsew",)
                    self.spinboxes[label] = spinbox

    def _create_figures(self):
        """
        Dynamically creates FigureCanvasTkAgg objects and embeds them into Tkinter frames.
        Uses parameters from self.params to configure the layout and properties.
        """
        figure_params = self.params["figure_names"]
        for figure_name in figure_params.keys():
            frame = self.frames[figure_name]
            figure_info = figure_params[figure_name]

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
                if not self.stop_operation:
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



    def _create_listbox(self):
        nav_frame_names = self.params["frame_names"]
        names = [f[0] for f in nav_frame_names]
        for name in names:
            if name == "root":
                pass
            else:
                frame = self.frames[name]

                # Get figure information for the current frame
                figure_info = self.params['listbox_names'].get(name, [])
                if len(figure_info) > 0:
                    fig_tag, row, col, rspan, cspan = figure_info
                    custom_font = font.Font(family='Helvetica', size=25)

                    # Create the Listbox
                    listbox = tk.Listbox(frame, font=custom_font, width=10)
                    listbox.grid(row=row, column=col, rowspan=rspan, columnspan=cspan, padx=1, pady=1, sticky="nsew")

                    # Create the Scrollbar
                    scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=listbox.yview)
                    scrollbar.grid(row=row, column=col + cspan, rowspan=rspan, sticky="ns")

                    # Configure the Listbox to work with the Scrollbar
                    listbox.config(yscrollcommand=scrollbar.set)

                    # Store the Listbox in the dictionary
                    self.listboxes[fig_tag] = listbox

    def show_frame(self, frame_name):

        """Brings the specified frame to the front."""
        frame = self.frames.get(frame_name)
        if frame:
            frame.tkraise()
        else:
            print(f"Frame {frame_name} does not exist!")

    def _show_figure(self, scrollbox_name):
        if 'historian_listbox' in self.listboxes.keys():
            self.populate_listbox(self.listboxes['historian_listbox'], scrollbox_name)
        """
        Brings the specified figure to the front in its parent frame.
        : The key of the figure to bring to the front.
        """
        figure_key = self.params['translation'][scrollbox_name][1]
        if figure_key in self.figures:
            frame = self.frames.get(figure_key)
            if frame:
                frame.tkraise()
            else:
                print(f"Frame {figure_key} does not exist!")


        else:
            pass




    def receive_img(self, image):

        self.update_gui_components(image)

    def create_scatter_plot(self, ax, key):
        figure_params = self.params['figure_names'][key]

        # Pull out data from the dataframe
        x_vals = self.hist_data_callback('timestamp', None)
        y_vals = self.hist_data_callback(figure_params['df_column_name'], figure_params['index'])

        if len(x_vals) > 0 and len(y_vals) > 0:
            check_list = self.hist_data_callback('scan_error', None)[0]

            # Filter the data
            x_list_filtered = [x for x, check in zip(x_vals[0], check_list) if not check]
            y_list_filtered = [y for y, check in zip(y_vals[0], check_list) if not check]

            # Convert timestamps to datetime objects
            x_vals_datetime = [datetime.datetime.fromtimestamp(i) for i in x_list_filtered]

            # Plot the data
            ax.scatter(x_list_filtered, y_list_filtered, label=figure_params['y1_label'])

            if figure_params['y2_label'] is not None:
                y2_list_filtered = [y for y, check in zip(y_vals[1], check_list) if not check]
                ax.scatter(x_list_filtered, y2_list_filtered, label=figure_params['y2_label'])

            # Manually update x-ticks
            # Select a subset of x-values for readability (e.g., every nth point)
            tick_indices = range(0, len(x_vals_datetime), max(1, len(x_vals_datetime) // 10))  # Adjust tick frequency
            x_ticks = [x_list_filtered[i] for i in tick_indices]
            x_tick_labels = [x_vals_datetime[i].strftime('%H:%M') for i in tick_indices]

            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels, rotation=45, ha='right', fontsize=12)
            ax.set_xlabel("Time", fontsize=20)
        else:
            # Handle empty data
            ax.scatter([], [], label=figure_params['y1_label'])
            if figure_params['y2_label'] is not None:
                ax.scatter([], [], label=figure_params['y2_label'])

        # Set other axis labels and titles
        yaxis_title = figure_params['axis_title']
        ax.set_ylabel(yaxis_title, fontsize=20)
        ax.set_title(f"{yaxis_title} Chart", fontsize=30)
        ax.tick_params(axis='both', which='major', labelsize=14)



    def create_error_chart(self, ax, key):
        from matplotlib.patches import Patch
        color_map = {
            'scan_error_figure':{
                "Scanning Error": "red",
                "No Errors": "green"
            },
            'product_error_figure':{
                "Non-Compliant Product Detected": "red",
                "Height Imbalance Detected": "yellow",
                "Low Efficiency Detected": "orange",
                "No Errors": "green",
            }
        }

        legend_handles = [Patch(color=color, label=label) for label, color in color_map[key].items()]

        # Add the legend to the plot
        ax.legend(handles=legend_handles, fontsize=12, loc="best")
        figure_params = self.params['figure_names'][key]
        x_vals = self.hist_data_callback('timestamp', None)
        y_vals = self.hist_data_callback(figure_params['df_column_name'], figure_params['index'])

        if len(x_vals) > 0 and len(y_vals) > 0:
            # Convert timestamps to datetime objects
            x_vals = [datetime.datetime.fromtimestamp(i) for i in x_vals[0]]
            y_vals = y_vals[0]
            for i in range(len(y_vals)):
                if y_vals[i] is False or y_vals[i] is None or y_vals[i] == 'None':
                    y_vals[i] = "No Errors"

            y_v = [1 for _ in y_vals]
            plot_colors = [color_map[key][j] for j in y_vals]

            # Sequential reading numbers for x-ticks
            reading_numbers = list(range(1, len(x_vals) + 1))

            # Plot the bars using reading numbers as x-coordinates
            ax.bar(reading_numbers, y_v, color=plot_colors, width=0.5)

            # Manually replace x-ticks with time labels
            ax.set_xticks(reading_numbers)
            ax.set_yticks([])
            ax.set_xticklabels([dt.strftime('%H:%M') for dt in x_vals], rotation=45)

            # Add labels and title
            yaxis_title = figure_params['axis_title']
            ax.set_xlabel("Readings", fontsize=20)
            ax.set_ylabel(yaxis_title, fontsize=20)
            ax.set_title(yaxis_title + ' Chart', fontsize=30)
            ax.tick_params(axis='both', which='major', labelsize=14)
            # Get the current y-axis limits
            y_min, y_max = ax.get_ylim()

            # Double the upper limit to make room for the legend
            ax.set_ylim(y_min, y_max * 1.33)



            # Raise error if product error or scanning error is not None
            if y_vals[len(y_vals) - 1] != 'No Errors' or not y_vals[len(y_vals) - 1]:
                self.raise_error(f"Error Message: \n {y_vals[len(y_vals) - 1]}")


    def update_gui_components(self, image):
        # updating listbox
        if not self.stop_operation:
            self.populate_listbox(self.listboxes['historian_listbox'], self.spinboxes['historian_data_labels'].var)
            print(f'time at start: {datetime.datetime.now()}')
            display_information = self.params['figure_names']
            for key in display_information.keys():
                if not self.stop_operation:
                    display = display_information[key]
                    if key in self.figures.keys():
                        figure = self.figures[key]
                        fig = figure.figure
                        fig.clf()
                        ax = fig.add_subplot(111)
                        if hasattr(self, display['function']):
                            function = getattr(self, display['function'])
                            if not self.stop_operation:
                                if display['function'] == 'display_image':
                                    function(ax, image)
                                else:
                                    function(ax, key)
                            if not self.stop_operation:
                                figure.draw()
            self.update_labels()
        print(f'time at fin: {datetime.datetime.now()}')
        # Clear the current figure

    def update_labels(self):
        keys = [
            ('average_width', 'average_width'),
            ('average_height', 'average_height'),
            ('area_val', 'area_val'),      # was ('average_area', 'average_area')
            ('ret_yield', 'ret_yield'),    # was ('average_yield', 'average_yield')
        ]
        for key in keys:
            key, avg = key
            y_vals = self.hist_data_callback(key, None, datetime.datetime.now(), self.spinboxes['product'].var)
            y_vals = [y_v[0] for y_v in y_vals]
            y_val = y_vals[len(y_vals)-1]
            str_val = self.labels[key]
            str_val.config(text=str(y_val))
            avg_val = self.labels[avg]
            avg_val.config(text=str(round(sum(y_vals)/len(y_vals), 2)))


    def raise_error(self, text):
        colors = ["red", "white"]
        frame = self.frames['error_frame']
        if not frame:
            return

        label = self.labels['error_frame']
        label.config(text=text)
        frame.tkraise()
        self.root.update()

        # Define a helper function for the flashing logic
        def flash_color(index=0, remaining_time=4):
            if remaining_time > 0:
                # Change the background color
                label.config(bg=colors[index % len(colors)])

                # Schedule the next flash in 1 second
                self.root.after(1000, flash_color, index + 1, remaining_time - 1)
            else:
                # Lower the frame after 4 seconds
                frame.lower()
                self.root.update()

        # Start the flashing process
        flash_color()

    def set_var_to_flowrate(self):
        self.density_frame_var = True

    def set_var_to_line_speed(self):
        self.density_frame_var = False

    def calculate_density(self):
        flowrate_label = self.labels['flow_rate_var_label']
        line_speed_label = self.labels['line_speed_var_label']
        area_vals = self.hist_data_callback('area_val', None)
        if len(area_vals) == 0 or len(area_vals[0]) == 0:
            self.raise_error("No Surface Area Detected")
        else:
            area_vals = area_vals[0]
            try:
                flowrate = float(flowrate_label.cget('text'))
                line_speed = float(line_speed_label.cget('text'))
                surface_area = area_vals[len(area_vals) - 1]
                density_val = round(flowrate * 0.96 / line_speed / (surface_area/144), 2)
                density_val_label = self.labels['density_result_label']
                density_val_label.config(text=str(density_val))
            except ValueError:
                self.raise_error("Error Generating Density")



    def append_number(self, addition):
        if self.density_frame_var:
            label = self.labels['flow_rate_var_label']
        else:
            label = self.labels['line_speed_var_label']
        text = label.cget('text')
        if addition == "." and "." in text or len(str(text)) > 5:
            pass
        else:
            updated_text = text + addition
            label.config(text=updated_text)

    def remove_char(self):
        if self.density_frame_var:
            label = self.labels['flow_rate_var_label']
        else:
            label = self.labels['line_speed_var_label']
        text = label.cget('text')
        updated_text = text[:-1]
        label.config(text=updated_text)

    def _create_calculator_buttons(self):
        button_info = self.params['calculator_button_names']['density_frame']
        master = self.frames['density_frame']
        fontsize = self.params['title_fontsize']
        for button in button_info:
            text, row, col, rspan, cspan = button
            if text == "X":
                button_object = tk.Button(
                    master=master,
                    text=text,
                    command=self.remove_char,  # Pass the command directly
                    borderwidth=1,
                    relief="solid",
                    width=10,
                    height=2,
                    font=('Arial', fontsize),
                    background='orange'
                )
            else:
                button_object = tk.Button(
                    master=master,
                    text=text,
                    command=lambda x=text: self.append_number(x),  # Pass the command directly
                    borderwidth=1,
                    relief="solid",
                    width=10,
                    height=2,
                    font=('Arial', fontsize),
                    background='orange'
                )
            button_object.grid(
                row=row,
                column=col,
                rowspan=rspan,
                columnspan=cspan,
                padx=1,
                pady=1,
                sticky="nsew"
            )

    def apply_tuning_settings(self):
        to_processor_list = {}
        value_updates = ['x_b', 'y_b', 'left_to_right_scalar', 'top_to_bottom_scalar']
        for label in value_updates:
            new_value = float(self.spinboxes[label].var)
            to_processor_list[label] = new_value
            if new_value != self.params[label]:
                self.params[label] = new_value
        self.tuning_parameter_callback(to_processor_list)



    def apply_settings(self):
        email_result = self.spinboxes['email_alerts'].var
        self.variables['email_alerts'] = email_result

        self.variables['language_setting'] = self.spinboxes['language_setting'].var
        self.variables['product'] = self.spinboxes['product'].var

        value_updates = [('minimum_height_value', 0), ('maximum_height_value', 1), ('minimum_density_value', 2),
                         ('maximum_density_value', 3), ('minimum_surface_area_value', 4),
                         ('maximum_surface_area_value', 5)]
        product_type = self.spinboxes['product'].var
        product_data = self.params['selection_lists']['product'][product_type]
        for value, index in value_updates:
            self.labels[value].config(text=product_data[index])
            self.variables[value] = product_data[index]
        if self.settings_callback:
            self.settings_callback(self.variables)


    def callback_settings_params(self, callback):
        self.settings_callback = callback



    def email_data(self):
        print('emailing data')

    def stop_scanning(self, event=None):
        print('stopping scanning')
        self.stop_operation = True
        self.stop_button_callback()

    def display_image(self, canvas: FigureCanvasTkAgg, img=None):
        """
        Displays an image on a FigureCanvasTkAgg canvas.

        :param canvas: A FigureCanvasTkAgg object to display the image on.
        :param img: The image to be displayed (in BGR format, as used by OpenCV).
        """
        if img is None:
            return

        # Convert the image from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create a Matplotlib figure
        fig = canvas.figure
        fig.clf()  # Clear the figure to avoid overlap

        ax = fig.add_subplot(111)  # Add a subplot
        ax.imshow(img_rgb)  # Display the image
        ax.axis('off')  # Hide the axis

        # Draw the updated figure on the canvas

    def create_scrollbox(self, frame):
        custom_font = font.Font(family='Helvetica', size=25)
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set, font=custom_font)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Configure the scrollbar to control the Listbox
        scrollbar.config(command=self.listbox.yview)

    def populate_listbox(self, listbox, scrollbox_name):
        # Check if data is already in the listbox to avoid redundant updates
        existing_items = listbox.get(0, tk.END)
        if scrollbox_name in self.params['historian_display_list_data'].keys():
            listbox_instructions = self.params['historian_display_list_data'][scrollbox_name]

            ts = self.hist_data_callback('timestamp', None)
            if len(ts) == 0:
                return
            else:
                ts = ts[0]
                y_ret = self.hist_data_callback(listbox_instructions['db_tag'],listbox_instructions['index'])
                y_label = listbox_instructions['yaxis']
                if type(y_label) == tuple:
                    data1 = y_ret[0]
                    data2 = y_ret[1]
                    label1 = y_label[0]
                    label2 = y_label[1]
                else:
                    data1 = y_ret[0]
                    data2 = None
                    label1 = y_label
                    label2 = None


                listbox.delete(0, tk.END)  # Clear listbox only if necessary
                for i in range(len(ts)):
                    if data1[i] == 'n/a':
                        continue

                    time_display = datetime.datetime.fromtimestamp(ts[i]).strftime('%H:%M:%S')
                    listbox.insert(0, f'Time: \t {time_display}')
                    listbox.insert(0, f'{label1}: \t {data1[i]}')
                    if data2:
                        listbox.insert(0, f'{label2}: \t {data2[i]}')
                    listbox.insert(0, '-------------')
        else:
            raise ValueError(f'key {scrollbox_name} not found in figure instructions')
    """WIFI SETTINGS BELOW"""
    def _run_nmcli(self, *args):
        """Run nmcli and return (ok, stdout or error)."""
        if not self.wifi_supported:
            return False, "Wi-Fi control unavailable (nmcli not found)."
        try:
            cp = subprocess.run(["nmcli", *args], capture_output=True, text=True, timeout=8)
            if cp.returncode == 0:
                return True, cp.stdout.strip()
            return False, cp.stderr.strip() or cp.stdout.strip()
        except Exception as e:
            return False, str(e)

    def _build_wifi_ui(self):
        """Create the Wi-Fi list and controls inside wifi_frame."""
        frame = self.frames.get("wifi_frame")
        if not frame:
            return

        # Layout
        for r in range(10):
            frame.grid_rowconfigure(r, weight=1)
        for c in range(8):
            frame.grid_columnconfigure(c, weight=1)

        # Title (optional, you also have labels via params)
        # ttk.Label(frame, text="Wi-Fi", font=("Arial", self.params['title_fontsize'])).grid(row=0, column=0, columnspan=8)

        # Tree of networks
        cols = ("ssid", "signal", "security", "active")
        self.wifi_tree = ttk.Treeview(frame, columns=cols, show="headings", height=12)
        for col, hdr, w in [
            ("ssid", "SSID", 320),
            ("signal", "Signal", 100),
            ("security", "Security", 140),
            ("active", "Active", 90),
        ]:
            self.wifi_tree.heading(col, text=hdr)
            self.wifi_tree.column(col, width=w, anchor="w", stretch=True)
        self.wifi_tree.grid(row=1, column=0, rowspan=7, columnspan=6, sticky="nsew", padx=8, pady=8)

        sb = ttk.Scrollbar(frame, orient="vertical", command=self.wifi_tree.yview)
        sb.grid(row=1, column=6, rowspan=7, sticky="ns")
        self.wifi_tree.configure(yscrollcommand=sb.set)

        # Right-side controls
        controls = tk.Frame(frame, bg="gray")
        controls.grid(row=1, column=7, rowspan=7, sticky="nsew", padx=8, pady=8)
        for r in range(8):
            controls.grid_rowconfigure(r, weight=1)
        controls.grid_columnconfigure(0, weight=1)

        ttk.Button(controls, text="Refresh", command=self.scan_wifi).grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(controls, text="Connect", command=self.connect_selected).grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(controls, text="Disconnect", command=self.disconnect_wifi).grid(row=2, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(controls, text="Forget", command=self.forget_selected).grid(row=3, column=0, sticky="ew", padx=4, pady=4)

        ttk.Label(controls, text="Password").grid(row=4, column=0, sticky="w", padx=4)
        ttk.Entry(controls, textvariable=self.wifi_password_var, show="*").grid(row=5, column=0, sticky="ew", padx=4, pady=4)

        self.wifi_status_label = ttk.Label(controls, text="Status: —")
        self.wifi_status_label.grid(row=7, column=0, sticky="ew", padx=4, pady=4)

        # Initial populate
        self.scan_wifi()

    def scan_wifi(self):
        """Populate the list of nearby networks."""
        if not self.wifi_supported:
            self.wifi_status_label.config(text="Status: nmcli not available")
            return

        ok, out = self._run_nmcli("-t", "-f", "ACTIVE,SSID,SIGNAL,SECURITY", "dev", "wifi", "list")
        if not ok:
            self.raise_error(f"Wi-Fi scan failed:\n{out}")
            self.wifi_status_label.config(text=f"Status: {out[:60]}")
            return

        # Clear rows
        for row in self.wifi_tree.get_children():
            self.wifi_tree.delete(row)

        # Rows like: "yes:HomeWiFi:68:WPA2" or "no::--:--"
        for line in out.splitlines():
            if not line.strip():
                continue
            active, ssid, signal, security = (part if part else "—" for part in line.split(":", 3))
            self.wifi_tree.insert("", "end", values=(ssid, signal, security, "Yes" if active == "yes" else ""))

        # Current connection
        self.update_wifi_status()

    def update_wifi_status(self):
        ok, out = self._run_nmcli("-t", "-f", "GENERAL.STATE", "device", "show", "wlan0")
        if ok and out:
            # e.g. GENERAL.STATE:100 (connected)
            self.wifi_status_label.config(text=f"Status: {out.split(':',1)[-1]}")
        else:
            self.wifi_status_label.config(text=f"Status: {out if out else 'unknown'}")

    def _selected_ssid(self):
        sel = self.wifi_tree.selection()
        if not sel:
            self.raise_error("Select a network first.")
            return None
        ssid = self.wifi_tree.item(sel[0], "values")[0]
        if ssid in ("—", "", None):
            self.raise_error("Network has no SSID (hidden).")
            return None
        return ssid

    def connect_selected(self):
        ssid = self._selected_ssid()
        if not ssid:
            return

        # Determine if security requires a password (best-effort via scan row)
        item = self.wifi_tree.item(self.wifi_tree.selection()[0], "values")
        security = item[2]
        if "WPA" in security or "WEP" in security:
            pwd = self.wifi_password_var.get().strip()
            if not pwd:
                self.raise_error(f"Password required for {ssid}.")
                return
            ok, out = self._run_nmcli("dev", "wifi", "connect", ssid, "password", pwd, "ifname", "wlan0")
        else:
            ok, out = self._run_nmcli("dev", "wifi", "connect", ssid, "ifname", "wlan0")

        if not ok:
            self.raise_error(f"Connect failed:\n{out}")
        else:
            self.wifi_status_label.config(text=f"Connected: {ssid}")
            self.scan_wifi()

    def disconnect_wifi(self):
        ok, out = self._run_nmcli("device", "disconnect", "wlan0")
        if not ok:
            self.raise_error(f"Disconnect failed:\n{out}")
        self.scan_wifi()

    def forget_selected(self):
        ssid = self._selected_ssid()
        if not ssid:
            return
        # Find connection name that matches SSID, then delete it
        ok, out = self._run_nmcli("-t", "-f", "NAME,TYPE", "connection", "show")
        if not ok:
            self.raise_error(f"Could not list connections:\n{out}")
            return
        for line in out.splitlines():
            name, ctype = (line.split(":", 1) + [""])[:2]
            if ctype == "802-11-wireless" and name == ssid:
                ok2, out2 = self._run_nmcli("connection", "delete", "id", name)
                if not ok2:
                    self.raise_error(f"Forget failed:\n{out2}")
                break
        self.scan_wifi()

    def _configure_grids(self):
        """Configures row and column weights for the root and frames."""
        # Configure root grid
        for r in range(10):
            self.root.grid_rowconfigure(r, weight=1)
        for c in range(8):
            self.root.grid_columnconfigure(c, weight=1)

        # Configure grids for all frames
        for frame_name, frame in self.frames.items():
            rows, cols = self._get_frame_grid_settings(frame_name)
            for r in range(rows):
                frame.grid_rowconfigure(r, weight=1)
            for c in range(cols):
                frame.grid_columnconfigure(c, weight=1)

    def _get_frame_grid_settings(self, frame_name):
        """Returns grid configuration for each frame."""
        grid_settings = {
            "home_frame": (8, 8),
            "density_frame": (8, 8),
            "historian_frame": (8, 8),
            "error_frame": (8, 8),
            "tuning_frame": (10, 8),
            "settings_frame": (10, 8),
            "exit_frame": (8, 8),
        }
        return grid_settings.get(frame_name, (1, 1))
    def _build_exit_ui(self):
        """Create the username/password form inside exit_frame."""
        frame = self.frames.get("exit_frame")
        if not frame:
            return

        for r in range(8):
            frame.grid_rowconfigure(r, weight=1)
        for c in range(8):
            frame.grid_columnconfigure(c, weight=1)

        uname_lbl = tk.Label(frame, text="Username", font=("Arial", self.params['primary_fontsize']), bg="gray", relief="solid")
        uname_lbl.grid(row=2, column=2, columnspan=2, sticky="e", padx=8, pady=8)
        uname_ent = tk.Entry(frame, textvariable=self.exit_user_var, font=("Arial", self.params['primary_fontsize']))
        uname_ent.grid(row=2, column=4, columnspan=2, sticky="w", padx=8, pady=8)

        pwd_lbl = tk.Label(frame, text="Password", font=("Arial", self.params['primary_fontsize']), bg="gray", relief="solid")
        pwd_lbl.grid(row=3, column=2, columnspan=2, sticky="e", padx=8, pady=8)
        pwd_ent = tk.Entry(frame, textvariable=self.exit_pass_var, show="*", font=("Arial", self.params['primary_fontsize']))
        pwd_ent.grid(row=3, column=4, columnspan=2, sticky="w", padx=8, pady=8)

        enter_btn = tk.Button(frame, text="Enter", font=("Arial", self.params['title_fontsize']), bg="orange",
                              relief="solid", command=self._attempt_exit)
        enter_btn.grid(row=5, column=3, columnspan=2, sticky="nsew", padx=8, pady=8)

        # Submit on Enter
        frame.bind_all("<Return>", lambda _evt: self._attempt_exit())

    def _attempt_exit(self):
        """Validate credentials and request graceful exit without shutdown."""
        # Try both possible locations: top-level in params or nested under security
        creds = self.params.get('exit_credentials', None)
        if creds is None:
            creds = self.params.get('security', {}).get('exit_credentials', {})

        expected_user = str(creds.get('username', '')).strip()
        expected_pass = str(creds.get('password', '')).strip()

        typed_user = str(self.exit_user_var.get()).strip()
        typed_pass = str(self.exit_pass_var.get()).strip()

        if typed_user == expected_user and typed_pass == expected_pass:
            self.stop_operation = True
            if self.stop_button_callback:
                try:
                    self.stop_button_callback(False)  # call with turnoff=False
                except TypeError:
                    self.stop_button_callback()
        else:
            try:
                self.raise_error("Invalid credentials. Please try again.")
            except Exception:
                print("Invalid credentials. Please try again.")


    def setup_gui(self):
        """Sets up the entire GUI."""
        self._initialize_root()
        self._create_frames()
        self._configure_grids()
        self._create_directory_buttons()
        self._create_function_buttons()
        self._create_labels()
        self._create_calculator_buttons()
        self._create_spinbox_objects()
        self._create_figures()
        self._create_listbox()
        if "wifi_frame" in self.frames:
            self._build_wifi_ui()
        if "exit_frame" in self.frames:
            self._build_exit_ui()
        self.root.mainloop()

    def exit_gui(self):
        self.root.quit()
        self.root.destroy()
        print('be gone')



if __name__ == '__main__':
    from camera_parameters import monocle_parameters
    root = tk.Tk()
    a = GUIManager(root, monocle_parameters)
    a.setup_gui()