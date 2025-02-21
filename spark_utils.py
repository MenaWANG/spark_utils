from pyspark.sql import DataFrame, Window
# from pyspark.sql.functions import col, count, round, format_string, lower, when, to_date, row_number, regexp_replace
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType
from functools import reduce
from typing import List
from pyspark.sql import SparkSession

def get_spark_session():
    """
    Get or create a Spark session using singleton pattern.
    """
    return SparkSession.builder.getOrCreate()


def round_numeric_cols(df, decimal_places=2):
    """
    Round all numeric columns (float, double, and decimal) in the DataFrame 
    to the specified number of decimal places. Integers are not rounded.

    Parameters:
        df (pyspark.sql.DataFrame): The input DataFrame to process.
        decimal_places (int, optional): The number of digits to the right of the decimal point, default is 2.

    Returns:
        pyspark.sql.DataFrame: A new DataFrame with numeric columns rounded to the specified decimal places.
    """
    for col_name, dtype in df.dtypes:
        if dtype in ["double", "float", "decimal"] or dtype.startswith("decimal"):
            df = df.withColumn(col_name, F.round(F.col(col_name), decimal_places))
        elif dtype in ["int", "bigint"]:
            df = df.withColumn(col_name, F.col(col_name))

    return df

def round_given_cols(df, cols, decimal_places=2):
    """
    Round given numeric columns in the DataFrame to the specified number of decimal places.

    Parameters:
        df (pyspark.sql.DataFrame): The input DataFrame to process.
        cols (list): The list of columns to round.
        decimal_places (int, optional): The number of digits to the right of the decimal point, default is 2.

    Returns:
        pyspark.sql.DataFrame: A new DataFrame with numeric columns rounded to the specified decimal places.
    """
    for col_name in cols:
        df = df.withColumn(col_name, F.round(F.col(col_name), decimal_places))
    return df


def shape(df: DataFrame, print_only: bool = True):
    """
    Get the number of rows and columns in the DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame to get the shape of.
    - print_only (bool): If True, only print out the shape. Default is True.

    Returns:
    - tuple or None: (num_rows, num_cols) if print_only is False, otherwise None
    """
    num_rows = df.count()
    num_cols = len(df.columns)
    print(f"Number of rows: {num_rows:,}")
    print(f"Number of columns: {num_cols:,}")
    if print_only:
        return None  
    else:
        return num_rows, num_cols

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

def is_primary_key(df: DataFrame, cols: List[str], verbose: bool = True) -> bool:
    """
    Check if the combination of specified columns forms 
    a primary key in the DataFrame.

    Parameters:
        df (DataFrame): The DataFrame to check.
        cols (list): A list of column names to check for forming a primary key.
        verbose (bool): If True, print detailed information. Default is True.

    Returns:
        bool: True if the combination of columns forms a primary key, False otherwise.
    """
    # Get SparkSession

    # Check if the DataFrame is not empty
    if df.isEmpty():
        if verbose:
            print("DataFrame is empty.")
        return False

    # Check if all columns exist in the DataFrame
    missing_cols = [col_name for col_name in cols if col_name not in df.columns]
    if missing_cols:
        if verbose:
            print(f"Column(s) {', '.join(missing_cols)} do not exist in the DataFrame.")
        return False

    # Check for missing values in each specified column
    for col_name in cols:
        missing_rows_count = df.where(F.col(col_name).isNull()).count()
        if missing_rows_count > 0:
            if verbose:
                print(f"There are {missing_rows_count:,} row(s) with missing values in column '{col_name}'.")

    # Filter out rows with missing values in any of the specified columns
    filtered_df = df.dropna(subset=cols)

    # Check if the combination of columns is unique after filtering out missing value rows
    unique_row_count = filtered_df.select(*cols).distinct().count()
    total_row_count = filtered_df.count()

    if verbose:
        print(f"Total row count after filtering out missings: {total_row_count:,}")
        print(f"Unique row count after filtering out missings: {unique_row_count:,}")

    if unique_row_count == total_row_count:
        if verbose:
            print(f"The column(s) {', '.join(cols)} form a primary key.")
        return True
    else:
        if verbose:
            print(f"The column(s) {', '.join(cols)} do not form a primary key.")
        return False

def find_duplicates(df: DataFrame, cols: List[str]) -> DataFrame:
    """
    Function to find duplicate rows based on specified columns.

    Parameters:
    - df (DataFrame): The DataFrame to check.
    - cols (list): List of column names to check for duplicates

    Returns:
    - duplicates (DataFrame): PySpark DataFrame containing duplicate rows based on the specified columns,
                  with the specified columns and the 'count' column as the first columns,
                  along with the rest of the columns from the original DataFrame
    """
    # Filter out rows with missing values in any of the specified columns
    for col_name in cols:
        df = df.filter(F.col(col_name).isNotNull())

    # Group by the specified columns and count the occurrences
    dup_counts = df.groupBy(*cols).count()
    
    # Filter to retain only the rows with count greater than 1
    duplicates = dup_counts.filter(F.col("count") > 1)
    
    # Join with the original DataFrame to include all columns
    duplicates = duplicates.join(df, cols, "inner")
    
    # Reorder columns with 'count' as the first column
    duplicate_cols = ['count'] + cols
    duplicates = duplicates.select(*duplicate_cols, *[c for c in df.columns if c not in cols])

    # Sort the result by the specified columns
    duplicates = duplicates.orderBy(*cols)
    
    return duplicates

def cols_responsible_for_id_dups(spark_df: DataFrame, id_list: List[str]) -> DataFrame:
    
    """
    This diagnostic function checks each column 
    for each unique id combinations to see whether there are differences, 
    then generates a summary table. 
    This can be used to identify columns responsible for most duplicates
    and help with troubleshooting.

    Parameters:
    - spark_df (DataFrame): The Spark DataFrame to analyze.
    - id_list (list): A list of column names representing the ID columns.

    Returns:
    - summary_table (DataFrame): A Spark DataFrame containing two columns 
      'col_name' and 'difference_counts'. 
      It represents the count of differing values for each column 
      across all unique ID column combinations.
    """
    # Get SparkSession
    spark = get_spark_session()
    
    # Filter out rows with missing values in any of the ID columns
    filtered_df = spark_df.na.drop(subset=id_list)
    
    # Define a function to count differences within a column for unique id_list combinations
    def count_differences(col_name):
        """
        Counts the number of differing values for each col_name.

        Parameters:
        - col_name (str): The name of the column to analyze.

        Returns:
        - count (int): The count of differing values.
        """
        # Count the number of distinct values for each combination of ID columns and current column
        distinct_count = filtered_df.groupBy(*id_list, col_name).count().groupBy(*id_list).count()
        return distinct_count.filter(F.col("count") > 1).count()
    
    # Get the column names excluding the ID columns
    value_cols = [col_name for col_name in spark_df.columns if col_name not in id_list]
    
    # Create a DataFrame to store the summary table
    summary_data = [(col_name, count_differences(col_name)) for col_name in value_cols]
    summary_table = spark.createDataFrame(summary_data, ["col_name", "difference_counts"])

    # Sort the summary_table by "difference_counts" from large to small
    summary_table = summary_table.orderBy(F.col("difference_counts").desc()) 
        
    return summary_table


def filter_df_by_strings(df:DataFrame, col_name:str, search_strings: List[str]) -> DataFrame:
    """
    Filter a DataFrame to find rows where the specified column contains 
    any of the given strings (case-insensitive).

    Parameters:
        df (DataFrame): The DataFrame to filter.
        col_name (str): The name of the column in which to search for the strings.
        search_strings (list of str): The list of strings to search for (case-insensitive).

    Returns:
        DataFrame: A new DataFrame containing only the rows where the specified column contains any of the search strings.
    """
    # Convert the search strings to lowercase
    search_strings_lower = [search_string.lower() for search_string in search_strings]
    
    # Construct the filter condition for each search string
    filter_conditions = [F.lower(F.col(col_name)).contains(search_string_lower) for search_string_lower in search_strings_lower]
    
    # Combine the filter conditions using OR
    combined_filter = reduce(lambda a, b: a | b, filter_conditions)
    
    # Filter the DataFrame
    filtered_df = df.filter(combined_filter)
    
    return filtered_df

def value_counts_with_pct(df:DataFrame, column_name:str) -> DataFrame:
    """
    Calculate the count and percentage of occurrences for each unique value in the specified column.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - column_name (str): The name of the column for which to calculate value counts.

    Returns:
    - DataFrame: A DataFrame containing two columns: the unique values in the specified column and their corresponding count and percentage.
    """
    counts = df.groupBy(column_name).agg(
        F.count("*").alias("count"),
        (F.count("*") / df.count() * 100).alias("pct")
    )

    counts = counts.withColumn("pct", F.round(F.col("pct"), 2))

    counts = counts.orderBy(F.col("count").desc())

    # Format count column with comma spacing for printing
    formatted_counts = counts.withColumn("count",  F.format_string("%,d", F.col("count")))
    formatted_counts.show()

    # Return counts DataFrame with raw numbers
    return counts

def transform_date_cols(df: DataFrame, date_cols: List[str], str_date_format: str = "ddMMMyyyy") -> DataFrame:
    """
    Transforms specified columns in a DataFrame to date format.
    
    Parameters:
        df (DataFrame): The input DataFrame.
        date_cols (List[str]): A list of column names to be transformed to dates.
        str_date_format (str, optional): The string format of the dates. Defaults to "ddMMMyyyy".

    Returns:
        DataFrame: The DataFrame with specified columns transformed to date format.
    """
    df_ = df
    for date_col in date_cols:
        # Check if the column is already of type DateType
        if dict(df.dtypes)[date_col] != 'date':
            df_ = df_.withColumn(date_col, F.to_date(F.col(date_col), str_date_format))

    return df_

def filter_by_date(df: DataFrame, date_col: str, min_date: str, max_date: str, original_date_format: str = "ddMMMyyyy") -> DataFrame:
    """
    Filter the DataFrame to include only rows where the specified date column is within the range [min_date, max_date].

    PySpark uses Java's date format, NOT Python's strftime:
        - dd → Day (2-digit, e.g., 02)
        - MMM → Abbreviated month (e.g., JAN)
        - yy → 2-digit year (e.g., 99 → 1999)
        - yyyy → 4-digit year (e.g., 2022)
        - HH → Hour (24-hour format, e.g., 23 for 11 PM)
        - mm → Minutes (e.g., 02)
        - ss → Seconds (e.g., 59)
        - S --> Milliseconds (e.g., 999)
        
    Parameters:
    - df (DataFrame): The DataFrame to filter.
    - date_col (str): The name of the date column to filter on.
    - min_date (str): The minimum date in yyyy-mm-dd format
    - max_date (str): The maximum date in yyyy-mm-dd format
    - original_date_format (str, optional): The format of the original date column. Defaults to "ddMMMyyyy".

    Returns:
    - DataFrame: The filtered DataFrame containing rows where the date column is within the specified range.
    """
    df_ = df.withColumn(date_col, 
                            F.when(F.col(date_col).cast("date").isNull(), F.to_date(F.col(date_col), original_date_format))
                            .otherwise(F.col(date_col)))
    
    filtered_df = df_.filter((F.col(date_col) >= min_date) & (F.col(date_col) <= max_date))
    
    return filtered_df

def get_distinct_values(df: DataFrame, column_name: str) -> list:
    """
    Function to get distinct values of a column in alphabetical order.

    Parameters:
    - df: PySpark DataFrame
    - column_name: Name of the column for which distinct values are to be retrieved

    Returns:
    - distinct_values_list: List of distinct values in alphabetical order
    """
    # Select the column, get distinct values, and order them
    distinct_values = df.select(F.col(column_name)).distinct().orderBy(column_name)
    # Collect the results to the driver and convert to a list
    distinct_values_list = [row[0] for row in distinct_values.collect()]
    return distinct_values_list

def top_rows_for_ids(df: DataFrame, id_list: list, value_field: str, ascending: bool = False) -> DataFrame:
    """
    Find all records for unique combination of id_list, then sort by a value field and take the top row for each id combination.

    Parameters:
        df (DataFrame): Input DataFrame.
        id_list (list): List of column names to use for identifying unique combinations.
        value_field (str): Field to sort by.
        ascending (bool, optional): Whether to sort in ascending order. Default is False (descending).

    Returns:
        DataFrame: DataFrame containing only the top row for each unique combination of id_list, sorted by value_field.
    """
    # Ensure id_list is not empty
    assert len(id_list) > 0, "id_list must not be empty"

    # Ensure value_field is present in the DataFrame
    assert value_field in df.columns, f"{value_field} not found in DataFrame columns"

    # Prepare window specification
    window_spec = Window.partitionBy(*id_list).orderBy(F.col(value_field).asc() if ascending else F.col(value_field).desc())

    # Add row number column to each partition
    df_ranked = df.withColumn("row_number", F.row_number().over(window_spec))

    # Filter DataFrame to keep only the top row for each id combination
    result_df = df_ranked.filter(F.col("row_number") == 1).drop("row_number")

    return result_df

def clean_dollar_cols(df: DataFrame, cols_to_clean: List[str]) -> DataFrame:
    """
    Clean specified columns of a Spark DataFrame by removing '$' symbols, commas, and converting to floating-point numbers.

    Parameters:
        df (DataFrame): The DataFrame to clean.
        cols_to_clean (List[str]): List of column names to clean.

    Returns:
        DataFrame: DataFrame with specified columns cleaned of '$' symbols and commas, and converted to floating-point numbers.
    """
    for col_name in cols_to_clean:
        df = df.withColumn(
            col_name, 
            F.when(
                F.col(col_name).isNotNull(), 
                F.regexp_replace(F.col(col_name), "[$,]", "").cast("float")
            ).otherwise(None)  # Keep NULLs unchanged
        )
    
    return df

