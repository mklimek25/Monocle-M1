import pandas as pd
import sqlite3
import json
import datetime
from pathlib import Path
import smtplib
from email.message import EmailMessage
import os


class DataManager:
    def __init__(self, params):
        """
        Initialize the DataManager with a params dictionary containing a list of column names.
        Args:
            params (dict): A dictionary containing a 'columns' key with a list of column names.
            db_name (str): The name of the SQLite database file.
            table_name (str): The name of the table to save data to.
        """
        self.params = params['data_frame_parameters']
        if 'data_columns' not in self.params or not isinstance(self.params['data_columns'], list):
            raise ValueError("Params must contain a key 'data_columns' with a list of column names.")

        # Create an empty DataFrame with specified columns
        db_name = self.params['df_name']

        self.df = pd.DataFrame(columns=self.params['data_columns'])
        self.db_name = db_name if db_name.endswith(".db") else db_name + ".db"
        self.table_name = self.params['table_name']
        self.gui_update_callback = None
        self.data_folder_path = self.params['usb_dir_path']
        self.excel_file_destination = None
        self.folder_name = None
        self.analysis_args = self.params['analysis_args']
        self.excel_path = None


    def establish_gui_update_callback(self, callback):
        self.gui_update_callback = callback

    # SET UP FILE SYSTEM IN USB DRIVE
    def setup_folder_structure(self):
        """
        Create a dated folder inside the internal_holdings_folder on the USB drive.
        Resulting path will look like:
        D:/internal_holdings_folder/08-04-2025-03/
        """
        try:
            base_path = Path(self.params['usb_dir_path'])  # Use pathlib for safety
            internal_folder = base_path / "internal_holdings_folder"
            internal_folder.mkdir(parents=True, exist_ok=True)

            today = datetime.datetime.now().strftime('%m-%d-%Y')
            existing_folders = [
                f.name for f in internal_folder.iterdir()
                if f.is_dir() and f.name.startswith(today)
            ]
            folder_suffix = f'{len(existing_folders)+1:02d}'
            new_folder_name = f'{today}-{folder_suffix}'

            self.folder_name = new_folder_name
            self.excel_file_destination = internal_folder / new_folder_name
            self.excel_file_destination.mkdir(parents=True, exist_ok=True)

            # Set final Excel path now
            self.excel_path = self.excel_file_destination / f'{new_folder_name}.xlsx'
            return True

        except FileNotFoundError:
            # If drive/path doesn't exist, run your custom error handler
            return False





    #  usb_base_path = 'D:/Monocle_Reports'  # Change this to your USB path

    def receive_frame_processor_data(self, row_data):
        print(f'data manager received data: {row_data}')
        """
        Append a new row to the DataFrame and save the updated data to SQL.

        Args:
            row_data (dict): A dictionary containing column names as keys and corresponding values.
             data_manager_input = (height_results, average_height, width_results, average_width,
                                      area_result, scan_error, product_error)
        """
        if not isinstance(row_data, dict):
            raise ValueError("Input data must be a dictionary with column names as keys.")

        row_data['height_results'] = json.dumps(row_data['height_results'])
        row_data['width_results'] = json.dumps(row_data['width_results'])
        # Append the new row to the DataFrame
        if len(row_data['height_results']) != 0:
            self.df = pd.concat([self.df, pd.DataFrame([row_data])], ignore_index=True)

        self._save_to_sql()


    def get_column(self, column_name):
        """
        Return a column of the DataFrame as a pandas Series.

        Args:
            column_name (str): The name of the column to return.

        Returns:
            list of pd.Series: The specified column.
        """

        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

        return self.df[column_name]



    def _process_results(self, df: pd.DataFrame, task_dict: dict, excel_path: str, sheet_name: str):
        """
        Process a DataFrame according to specified tasks on given columns,
        write results to a new Excel sheet, and return the results as a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to process.
            task_dict (dict): Dictionary in format {column_name: task}, where task is one of:
                              'SUM', 'AVERAGE', 'COUNT', 'COUNTA'.
            excel_path (str): Path to the existing Excel file to append to.
            sheet_name (str): Name of the sheet to create.

        Returns:
            pd.DataFrame: A DataFrame with columns ['Task Name', 'Value'].
        """

        rows = []

        for column, task in task_dict.items():
            if column not in df.columns:
                task_name = f"{task} of {column}"
                value = f"Column '{column}' not found."
                rows.append({'Task Name': task_name, 'Value': value})
                continue

            series = df[column]

            if task == 'SUM':
                task_name = f'Sum of {column}'
                value = series.dropna().sum()

            elif task == 'AVERAGE':
                task_name = f'Average of {column}'
                numeric_values = pd.to_numeric(series, errors='coerce')
                value = numeric_values.dropna().mean()

            elif task == 'COUNT':
                task_name = f'Count of {column}'
                value = series.count()

            elif task == 'COUNTA':
                task_name = f'Non-None Count of {column}'
                value = series[series != 'None'].count()

            else:
                task_name = f"{task} of {column}"
                value = f"Invalid task '{task}'."

            rows.append({'Task Name': task_name, 'Value': value})

        # Build the result DataFrame
        result_df = pd.DataFrame(rows)

        # Write to new sheet in existing Excel file
        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
                result_df.to_excel(writer, sheet_name=sheet_name, index=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"The Excel file '{excel_path}' does not exist.")
        except ValueError as e:
            raise ValueError(f"Failed to write to sheet '{sheet_name}': {e}")

        return result_df


    def _save_to_sql(self):
        # Save only the last row to SQL
        with sqlite3.connect(self.db_name) as conn:
            self.df.tail(1).to_sql(self.table_name, conn, if_exists="append", index=False)
            print(f"Final row saved to database: {self.db_name} (table: {self.table_name})")

        # Save to Excel
        try:
            print(f"Saving Excel to {self.excel_path}")
            self.df.to_excel(self.excel_path, sheet_name='Output', index=False)
            self._process_results(self.df, self.analysis_args, self.excel_path, sheet_name='Analysis')
            print(f"Excel file saved to: {self.excel_path}")
        except Exception as e:
            print(f"Error saving DataFrame to Excel: {e}")


    def send_requested_data_list(self, column_name, index, date_of_interest=None, product_of_interest=None):
        """
        Fetches and processes data from the DataFrame based on the requested column name.

        Args:
            column_name (str): The name of the column to process.

        Returns:
            None: The processed data is passed to the historian callback.
        """
        df = self.df.copy()
        if date_of_interest is not None:
            now = datetime.datetime.now()
            start_of_day = datetime.datetime.combine(now.date(), datetime.time.min).timestamp()
            end_of_day = datetime.datetime.combine(now.date(), datetime.time.max).timestamp()
            mask = (df['timestamp'] >= start_of_day) & (df['timestamp'] <= end_of_day)
            df = df[mask]


        return_list = []
        # Validate inputs
        if not isinstance(column_name, str) or len(df) == 0:
            return []

        # Handle column translation

        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the dataframe.")

        # Extract target column and indices
        elif index is not None:

            # Load JSON objects for complex column types
            list_of_interest = df[column_name].apply(json.loads).tolist()
            if list_of_interest is not None:
                # Extract specified indices for each list entry
                if type(index) == str:
                    index = [index]
                for item in index:

                    if item == "START":
                        idx = 0
                    elif item == "END":
                        idx = len(list_of_interest[0]) - 1
                    elif item == "MIDDLE":
                        idx = (len(list_of_interest[0]) - 1) // 2
                    else:
                        raise ValueError(f"Invalid index '{item}' in translation mapping.")

                    # Collect the values corresponding to the index
                    try:
                        return_list.append([entry[idx] for entry in list_of_interest])
                    except TypeError:
                        return_list.append(None)
            else:
                return_list = ['']
        else:
            # For simple columns, append the entire column's data
            return_list.append(df[column_name].tolist())

        # Pass the processed data to the historian list callback
        return return_list



    def send_excel_via_gmail(self, file_path, recipient_email, sender_email, app_password):
        if not os.path.isfile(file_path):
            print("❌ File does not exist:", file_path)
            return False

        # Build the email message
        msg = EmailMessage()
        msg['Subject'] = 'TESTING - Run results from ' + datetime.datetime.now().strftime("%d/%m/%Y")
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg.set_content("Please find the attached Excel file.")

        # Attach the file
        with open(file_path, 'rb') as f:
            file_data = f.read()
            file_name = os.path.basename(file_path)
            msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)

        # Send the email via Gmail SMTP
        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
                smtp.starttls()
                smtp.login(sender_email, app_password)
                smtp.send_message(msg)
            print("✅ Email sent successfully.")
            return True
        except Exception as e:
            print("❌ Failed to send email:", e)
            return False

    def establish_historian_list_callback(self, callback):
        self.historian_list_callback = callback

if __name__ == "__main__":
    from camera_parameters import monocle_parameters
    a = DataManager(monocle_parameters)
    dir_path = a.data_folder_path
    a.setup_folder_structure()
    a._save_to_sql()

    pass

