import sqlite3
import pandas as pd

def export_to_excel(db_name: str, table_name: str, output_file: str):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)

    # Read the table into a Pandas DataFrame
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)

    # Save DataFrame to an Excel file
    df.to_excel(output_file, index=False)

    # Close the database connection
    conn.close()

    print(f"Data exported successfully to {output_file}")

# Usage
export_to_excel('test.db', 'monocle_data', 'results.xlsx')
