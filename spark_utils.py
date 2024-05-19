from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, round, format_string,  lower
from functools import reduce


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

def find_duplicates(df, cols):
    """
    Function to find duplicate rows in a Spark DataFrame based on specified columns.

    Args:
    - df: PySpark DataFrame
    - cols: List of column names to check for duplicates

    Returns:
    - duplicates: PySpark DataFrame containing duplicate rows based on the specified columns,
                  with the specified columns and the 'count' column as the first columns,
                  along with the rest of the columns from the original DataFrame
    """
    # Filter out rows with missing values in any of the specified columns
    for col_name in cols:
        df = df.filter(col(col_name).isNotNull())

    # Group by the specified columns and count the occurrences
    dup_counts = df.groupBy(*cols).count()
    
    # Filter to retain only the rows with count greater than 1
    duplicates = dup_counts.filter(col("count") > 1)
    
    # Join with the original DataFrame to include all columns
    duplicates = duplicates.join(df, cols, "inner")
    
    # Reorder columns with 'count' as the first column
    duplicate_cols = ['count'] + cols
    duplicates = duplicates.select(*duplicate_cols, *[c for c in df.columns if c not in cols])

    # Sort the result by the specified columns
    duplicates = duplicates.orderBy(*cols)
    
    return duplicates

def cols_responsible_for_id_dups(spark_df, id_list):
    
    """
    This diagnostic function checks each column 
    for each unique id combinations to see whether there are differences, 
    then generates a summary table. 
    This can be used to identify columns responsible for most duplicates
    and help with troubleshooting.

    Args:
    - spark_df (DataFrame): The Spark DataFrame to analyze.
    - id_list (list): A list of column names representing the ID columns.

    Returns:
    - summary_table (DataFrame): A Spark DataFrame containing two columns 
      'col_name' and 'difference_counts'. 
      It represents the count of differing values for each column 
      across all unique ID column combinations.
    """
    # Get or create the SparkSession
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    
    # Filter out rows with missing values in any of the ID columns
    filtered_df = spark_df.na.drop(subset=id_list)
    
    # Define a function to count differences within a column for unique id_list combinations
    def count_differences(col_name):
        """
        Counts the number of differing values for each col_name.

        Args:
        - col_name (str): The name of the column to analyze.

        Returns:
        - count (int): The count of differing values.
        """
        # Count the number of distinct values for each combination of ID columns and current column
        distinct_count = filtered_df.groupBy(*id_list, col_name).count().groupBy(*id_list).count()
        return distinct_count.filter(col("count") > 1).count()
    
    # Get the column names excluding the ID columns
    value_cols = [col_name for col_name in spark_df.columns if col_name not in id_list]
    
    # Create a DataFrame to store the summary table
    summary_data = [(col_name, count_differences(col_name)) for col_name in value_cols]
    summary_table = spark.createDataFrame(summary_data, ["col_name", "difference_counts"])

    # Sort the summary_table by "difference_counts" from large to small
    summary_table = summary_table.orderBy(col("difference_counts").desc()) 
        
    return summary_table


def filter_df_by_strings(df, col_name, search_strings):
    """
    Filter a DataFrame to find rows where the specified column contains 
    any of the given strings (case-insensitive).

    Args:
        df (DataFrame): The DataFrame to filter.
        col_name (str): The name of the column in which to search for the strings.
        search_strings (list of str): The list of strings to search for (case-insensitive).

    Returns:
        DataFrame: A new DataFrame containing only the rows where the specified column contains any of the search strings.
    """
    # Convert the search strings to lowercase
    search_strings_lower = [search_string.lower() for search_string in search_strings]
    
    # Construct the filter condition for each search string
    filter_conditions = [lower(col(col_name)).contains(search_string_lower) for search_string_lower in search_strings_lower]
    
    # Combine the filter conditions using OR
    combined_filter = reduce(lambda a, b: a | b, filter_conditions)
    
    # Filter the DataFrame
    filtered_df = df.filter(combined_filter)
    
    return filtered_df

def value_counts_with_pct(df, column_name):
    """
    Calculate the count and percentage of occurrences for each unique value in the specified column.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - column_name (str): The name of the column for which to calculate value counts.

    Returns:
    - DataFrame: A DataFrame containing two columns: the unique values in the specified column and their corresponding count and percentage.
    """
    counts = df.groupBy(column_name).agg(
        count("*").alias("count"),
        (count("*") / df.count() * 100).alias("pct")
    )

    counts = counts.withColumn("pct", round(col("pct"), 2))

    counts = counts.orderBy(col("count").desc())

    # Format count column with comma spacing for printing
    formatted_counts = counts.withColumn("count", format_string("%,d", col("count")))
    formatted_counts.show()

    # Return counts DataFrame with raw numbers
    return counts
