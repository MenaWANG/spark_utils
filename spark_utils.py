from pyspark.sql import DataFrame
from pyspark.sql.functions import col


def shape(df: DataFrame):
    """
    Print the number of rows and columns in the DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame to get the shape of.

    Returns:
    None
    """
    num_rows = df.count()
    num_cols = len(df.columns)
    print(f"Number of rows: {num_rows:,}")
    print(f"Number of columns: {num_cols:,}")

def print_schema_alphabetically(df: DataFrame):
    """
    Prints the schema of the DataFrame with columns sorted alphabetically.
    
    Parameters:
    - df (DataFrame): The DataFrame whose schema is to be printed.
    
    Returns:
    None
    """
    sorted_columns = sorted(df.columns)
    sorted_df = df.select(sorted_columns)
    sorted_df.printSchema()

def is_primary_key(df: DataFrame, cols: list) -> bool:
    """
    Check if the combination of specified columns forms a primary key in the DataFrame.

    Args:
        df (DataFrame): The DataFrame to check.
        cols (list): A list of column names to check for forming a primary key.

    Returns:
        bool: True if the combination of columns forms a primary key, False otherwise.
    """
    # Check if the DataFrame is not empty
    if df.isEmpty():
        print("DataFrame is empty.")
        return False

    # Check if all columns exist in the DataFrame
    missing_cols = [col_name for col_name in cols if col_name not in df.columns]
    if missing_cols:
        print(f"Columns {', '.join(missing_cols)} do not exist in the DataFrame.")
        return False

    # Check for missing values in each specified column
    for col_name in cols:
        missing_rows_count = df.where(col(col_name).isNull()).count()
        if missing_rows_count > 0:
            print(f"There are {missing_rows_count:,} row(s) with missing values in column '{col_name}'.")

    # Filter out rows with missing values in any of the specified columns
    filtered_df = df.dropna(subset=cols)

    # Check if the combination of columns is unique after filtering out missing value rows
    unique_row_count = filtered_df.select(*cols).distinct().count()
    total_row_count = filtered_df.count()

    print(f"Total row count after filtering out missings: {total_row_count:,}")
    print(f"Unique row count after filtering out missings: {unique_row_count:,}")

    if unique_row_count == total_row_count:
        print(f"The column(s) {', '.join(cols)} forms a primary key.")
        return True
    else:
        print(f"The column(s) {', '.join(cols)} does not form a primary key.")
        return False

