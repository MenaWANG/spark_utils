from pyspark.sql import DataFrame, Window

# from pyspark.sql.functions import col, count, round, format_string, lower, when, to_date, row_number, regexp_replace
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType
from functools import reduce
from typing import List, Union, Optional, Dict, Any
import sys
import os
import getpass

def get_spark_session():
    """
    Get or create a Spark session using singleton pattern.
    """
    try:
        return spark 
    except NameError:
        # Only create Spark sesssion if it is not already available
        from pyspark.sql import SparkSession
        return SparkSession.builder.getOrCreate()


def setup_pydantic_v2(custom_path: str = None) -> Optional[str]:
    """
    Setup Pydantic v2 in DBR environment from a custom installation path.  
    
    This function assumes Pydantic v2 has already been installed to a custom path.
    
    Installation Instructions (run these FIRST):
    --------------------------------------------------------------
    import getpass
    username = getpass.getuser()
    custom_path = f"/tmp/custom_packages_{username}"
    
    %pip install --target {custom_path} pydantic==2.11.7
    
    Then call this function:
    setup_pydantic_v2(custom_path)
    
    Parameters:
    -----------
    custom_path : str, optional
        Path where Pydantic v2 is installed. If None, will try to determine
        from username automatically.
        
    Returns:
    --------
    str or None
        Pydantic version if successful, None if failed
        
    Examples:
    ---------
    >>> # Auto-detect path based on username
    >>> setup_pydantic_v2()
    
    >>> # Explicit path
    >>> setup_pydantic_v2("/tmp/custom_packages_myusername")
    """
    # Auto-generate custom_path if not provided
    if custom_path is None:
        username = getpass.getuser()
        custom_path = f"/tmp/custom_packages_{username}"
        print(f"🔍 Using auto-detected path: {custom_path}")
    
    # Check if custom path exists
    if not os.path.exists(custom_path):
        print(f"❌ Custom path not found: {custom_path}")
        print("💡 Please install Pydantic first:")
        print(f"   %pip install --target {custom_path} pydantic==2.11.7")
        return None
    
    # Check if we already have the right version
    try:
        import pydantic
        if pydantic.__version__.startswith('2.'):
            print(f"✅ Pydantic {pydantic.__version__} already available")
            return pydantic.__version__
    except ImportError:
        pass
    
    # Clear any cached pydantic modules to ensure clean import
    pydantic_modules = [mod for mod in sys.modules.keys() if mod.startswith('pydantic')]
    if pydantic_modules:
        print(f"🧹 Clearing cached modules: {pydantic_modules}")
        for mod in pydantic_modules:
            del sys.modules[mod]
    
    # Add custom packages path to the beginning of sys.path
    if custom_path not in sys.path:
        sys.path.insert(0, custom_path)
        print(f"📁 Added to Python path: {custom_path}")
    
    # Verify installation
    try:
        import pydantic
        if pydantic.__version__.startswith('2.'):
            print(f"🚀 Pydantic {pydantic.__version__} ready!")
            return pydantic.__version__
        else:
            print(f"⚠️  Found Pydantic {pydantic.__version__}, but expected v2.x")
            return None
    except ImportError as e:
        print(f"❌ Failed to import Pydantic from {custom_path}: {e}")
        print(f"💡 Please ensure Pydantic v2 is installed at: {custom_path}")
        return None


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
    Prints the schema of the DataFrame with columns sorted alphabetically (case-insensitive).

    Parameters:
    - df (DataFrame): The DataFrame whose schema is to be printed.

    Returns:
    None
    """
    sorted_columns = sorted(df.columns, key=str.lower)
    sorted_df = df.select(sorted_columns)
    sorted_df.printSchema()


def is_primary_key(df: DataFrame, cols: Union[str, List[str]], verbose: bool = True) -> bool:
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
    spark = get_spark_session()
    
    # If cols is a single string, convert it to a list
    cols = [cols] if isinstance(cols, str) else cols

    # Check if the DataFrame is not empty
    if df.isEmpty():
        if verbose:
            print("DataFrame is empty.")
        return False

    # Check if all columns exist in the DataFrame
    missing_cols = [col_name for col_name in cols if col_name not in df.columns]
    if missing_cols:
        if verbose:
            print(f"❌ Column(s) {', '.join(missing_cols)} do not exist in the DataFrame.")
        return False

    # Filter out rows with missing values and cache for multiple operations
    filtered_df = df.dropna(subset=cols)
    filtered_df.cache()

    try:
        # Check for missing values in each specified column
        all_columns_complete = True
        for col_name in cols:
            missing_rows_count = df.where(F.col(col_name).isNull()).count()
            if missing_rows_count > 0:
                all_columns_complete = False
                if verbose:
                    print(f"⚠️ There are {missing_rows_count:,} row(s) with missing values in column '{col_name}'.")
        
        if verbose and all_columns_complete:
            print(f"✅ No missing values found in columns: {', '.join(cols)}")

        # Get counts for comparison
        total_row_count = filtered_df.count()

        # Cache the selected columns as they're used for distinct count
        selected_cols = filtered_df.select(*cols)
        selected_cols.cache()

        try:
            unique_row_count = selected_cols.distinct().count()

            if verbose:
                if not all_columns_complete:
                    print(
                        f"ℹ️ Total row count after filtering out missings: {total_row_count:,}"
                    )
                    print(
                        f"ℹ️ Unique row count after filtering out missings: {unique_row_count:,}"
                    )
                else:
                    print(f"ℹ️ Total row count: {total_row_count:,}")
                    print(f"ℹ️ Unique row count: {unique_row_count:,}")

            is_primary = unique_row_count == total_row_count

            if verbose:
                if is_primary:
                    print(f"🔑 The column(s) {', '.join(cols)} form a primary key.")
                else:
                    print(f"❌ The column(s) {', '.join(cols)} do not form a primary key.")

            return is_primary

        finally:
            selected_cols.unpersist()

    finally:
        filtered_df.unpersist()


def find_duplicates(df: DataFrame, cols: List[str]) -> DataFrame:
    """
    Function to find duplicate rows based on specified columns.

    Parameters:
    - df (DataFrame): The DataFrame to check.
    - cols (list): List of column names to check for duplicates

    Returns:
    - result (DataFrame): PySpark DataFrame containing duplicate rows based on the specified columns,
                  with the specified columns and the 'count' column as the first columns,
                  along with the rest of the columns from the original DataFrame,
                  ordered by the specified columns.
    """
    filtered_df = df
    for col_name in cols:
        filtered_df = filtered_df.filter(F.col(col_name).isNotNull())
    filtered_df.cache()

    try:
        # Group by and count
        dup_counts = filtered_df.groupBy(*cols).count()
        dup_counts.cache()

        try:
            duplicates = dup_counts.filter(F.col("count") > 1)
            # Join and reorder
            duplicate_cols = ["count"] + cols
            result = (
                duplicates.join(filtered_df, cols, "inner")
                .select(
                    *duplicate_cols, *[c for c in filtered_df.columns if c not in cols]
                )
                .orderBy(*cols)
            )
            return result
        finally:
            dup_counts.unpersist()
    finally:
        filtered_df.unpersist()


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

    # Cache the filtered DataFrame as it will be used multiple times
    filtered_df = spark_df.na.drop(subset=id_list)
    filtered_df.cache()

    try:

        def count_differences(col_name):
            distinct_count = filtered_df.groupBy(*id_list, col_name).count()
            distinct_count.cache()  # Cache as it's used twice

            try:
                result = (
                    distinct_count.groupBy(*id_list)
                    .count()
                    .filter(F.col("count") > 1)
                    .count()
                )
            finally:
                distinct_count.unpersist()

            return result

        value_cols = [
            col_name for col_name in spark_df.columns if col_name not in id_list
        ]

        # Create and cache summary table
        summary_data = [
            (col_name, count_differences(col_name)) for col_name in value_cols
        ]
        summary_table = spark.createDataFrame(
            summary_data, ["col_name", "difference_counts"]
        ).orderBy(F.col("difference_counts").desc())

    finally:
        filtered_df.unpersist()

    return summary_table

def deduplicate_by_rank(
    df: DataFrame,
    id_cols: Union[str, List[str]],
    ranking_col: str,
    ascending: bool = False,
    tiebreaker_col: Optional[str] = None,
    verbose: bool = False,
) -> DataFrame:
    """
    Deduplicate rows by keeping the best-ranked row per group of id_cols,
    optionally breaking ties by preferring non-missing tiebreaker_col.

    Parameters
    ----------
    df : DataFrame
        The PySpark DataFrame to deduplicate.
    id_cols : Union[str, List[str]]
        Column(s) defining the unique entity (e.g., customer_id, product_id).
    ranking_col : str
        The column to rank within each group (e.g., 'date', 'score', 'priority').
    ascending : bool, default=False
        Sort order for ranking_col:
        - True: smallest value kept (e.g., earliest date)
        - False: largest value kept (e.g., most recent date, highest score)
    tiebreaker_col : Optional[str], default=None
        Column where non-missing values are preferred in case of ties.
        Useful when ranking_col has identical values.
    verbose : bool, default=False
        If True, print information about the deduplication process.

    Returns
    -------
    DataFrame
        Deduplicated PySpark DataFrame with one row per unique combination of id_cols.

    Examples
    --------
    >>> from pyspark.sql import SparkSession
    >>> spark = SparkSession.builder.appName("test").getOrCreate()
    >>> df = spark.createDataFrame([
    ...     ('C001', '2024-01-01', 100, 'old@email.com'),
    ...     ('C001', '2024-01-15', 200, 'new@email.com'),
    ...     ('C002', '2024-01-05', 150, None),
    ...     ('C002', '2024-01-10', 150, 'email@test.com'),
    ...     ('C003', '2024-01-20', 300, 'test@email.com')
    ... ], ['customer_id', 'transaction_date', 'amount', 'email'])

    >>> # Keep most recent transaction per customer
    >>> result = deduplicate_by_rank(df, 'customer_id', 'transaction_date', ascending=False)
    >>> result.show()

    >>> # Keep highest amount, break ties by preferring non-null email
    >>> result = deduplicate_by_rank(df, 'customer_id', 'amount', ascending=False, tiebreaker_col='email')
    >>> result.show()
    """
    # Handle empty DataFrame
    if df.count() == 0:
        if verbose:
            print("⚠️ Input DataFrame is empty. Returning empty DataFrame.")
        return df

    # Normalize id_cols to list
    if isinstance(id_cols, str):
        id_cols = [id_cols]

    # Validate that all columns exist
    df_columns = df.columns
    missing_cols = [col for col in id_cols + [ranking_col] if col not in df_columns]
    if tiebreaker_col and tiebreaker_col not in df_columns:
        missing_cols.append(tiebreaker_col)

    if missing_cols:
        raise ValueError(f"Column(s) {missing_cols} not found in DataFrame")

    if verbose:
        initial_count = df.count()
        unique_groups = df.select(*id_cols).distinct().count()
        print(f"🔄 Deduplicating {initial_count} rows by {id_cols}")
        print(f"ℹ️ Found {unique_groups} unique groups")

    # Define window specification for ranking
    window_spec = Window.partitionBy(*id_cols)
    
    # Build order by columns for the window
    order_cols = []
    
    # Add ranking column with appropriate sort order
    if ascending:
        order_cols.append(F.col(ranking_col).asc())
    else:
        order_cols.append(F.col(ranking_col).desc())
    
    # Add tiebreaker logic if specified
    if tiebreaker_col:
        # Prefer non-null values (nulls last)
        order_cols.append(F.col(tiebreaker_col).asc_nulls_last())
    
    # Apply ordering to window
    window_spec = window_spec.orderBy(*order_cols)
    
    # Add row number based on ranking
    df_with_rank = df.withColumn("_row_number", F.row_number().over(window_spec))
    
    # Keep only the first-ranked row per group
    dedup_df = df_with_rank.filter(F.col("_row_number") == 1).drop("_row_number")

    if verbose:
        final_count = dedup_df.count()
        removed_count = initial_count - final_count
        print(f"✅ Removed {removed_count} duplicate rows")
        print(f"📊 Final dataset: {final_count} rows")

    return dedup_df

def filter_df_by_strings(
    df: DataFrame, col_name: str, search_strings: List[str]
) -> DataFrame:
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
    filter_conditions = [
        F.lower(F.col(col_name)).contains(search_string_lower)
        for search_string_lower in search_strings_lower
    ]

    # Combine the filter conditions using OR
    combined_filter = reduce(lambda a, b: a | b, filter_conditions)

    # Filter the DataFrame
    filtered_df = df.filter(combined_filter)

    return filtered_df


def value_counts_with_pct(df: DataFrame, column_name: str) -> DataFrame:
    """
    Calculate the count and percentage of occurrences for each unique value in the specified column.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - column_name (str): The name of the column for which to calculate value counts.

    Returns:
    - DataFrame: A DataFrame containing two columns: the unique values in the specified column and their corresponding count and percentage.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")

    counts = df.groupBy(column_name).agg(
        F.count("*").alias("count"), (F.count("*") / df.count() * 100).alias("pct")
    )

    counts = counts.withColumn("pct", F.round(F.col("pct"), 2))

    counts = counts.orderBy(F.col("count").desc())

    # Format count column with comma spacing for printing
    formatted_counts = counts.withColumn(
        "count", F.format_string("%,d", F.col("count"))
    )
    formatted_counts.show()

    # Return counts DataFrame with raw numbers
    return counts


def transform_date_cols(
    df: DataFrame, date_cols: List[str], str_date_format: str = "ddMMMyyyy"
) -> DataFrame:
    """
    Transforms specified columns in a DataFrame to date format.

    Parameters:
        df (DataFrame): The input DataFrame.
        date_cols (List[str]): A list of column names to be transformed to dates.
        str_date_format (str, optional): The string format of the dates. Defaults to "ddMMMyyyy".

    Returns:
        DataFrame: The DataFrame with specified columns transformed to date format.
    """
    if not date_cols:
        raise ValueError("date_cols list cannot be empty")

    df_ = df
    for date_col in date_cols:
        # Check if the column is already of type DateType
        if dict(df.dtypes)[date_col] != "date":
            df_ = df_.withColumn(date_col, F.to_date(F.col(date_col), str_date_format))

    return df_


def filter_by_date(
    df: DataFrame,
    date_col: str,
    min_date: str,
    max_date: str,
    original_date_format: str = "ddMMMyyyy",
) -> DataFrame:
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
    df_ = df.withColumn(
        date_col,
        F.when(
            F.col(date_col).cast("date").isNull(),
            F.to_date(F.col(date_col), original_date_format),
        ).otherwise(F.col(date_col)),
    )

    filtered_df = df_.filter(
        (F.col(date_col) >= min_date) & (F.col(date_col) <= max_date)
    )

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


def top_rows_for_ids(
    df: DataFrame, id_list: list, value_field: str, ascending: bool = False
) -> DataFrame:
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
    window_spec = Window.partitionBy(*id_list).orderBy(
        F.col(value_field).asc() if ascending else F.col(value_field).desc()
    )

    # Add row number column to each partition
    df_ranked = df.withColumn("row_number", F.row_number().over(window_spec))

    # Filter DataFrame to keep only the top row for each id combination
    result_df = df_ranked.filter(F.col("row_number") == 1).drop("row_number")

    return result_df


def clean_dollar_cols(df: DataFrame, cols_to_clean: List[str]) -> DataFrame:
    """
    Clean specified columns of a Spark DataFrame by removing '$' symbols, commas,
    and converting to floating-point numbers.

    Parameters:
        df (DataFrame): The DataFrame to clean.
        cols_to_clean (List[str]): List of column names to clean.

    Returns:
        DataFrame: DataFrame with specified columns cleaned of '$' symbols and commas,
                   and converted to floating-point numbers.
    """
    df_ = df

    for col_name in cols_to_clean:
        df_ = df_.withColumn(
            col_name,
            F.when(
                F.col(col_name).isNotNull(),
                F.regexp_replace(
                    F.regexp_replace(F.col(col_name), r"^\$", ""),  # Remove $ at start
                    ",",
                    "",  # Remove commas
                ).cast("float"),
            ).otherwise(None),
        )

    return df_

def union_tables_by_prefix(
    table_prefix: str,
    output_table_name: str,
    output_schema: Optional[str] = None,
    source_schema: Optional[str] = None,
    order_by_col: Optional[Union[str, List[str]]] = None,
    drop_duplicates: bool = False
) -> None:
    """
    Safely combine multiple tables with the same prefix into a single table.
    Ensure the target table doesn't exist or is OK to overwrite before running.
    Handles column ordering differences by standardizing column order before union.
    
    Args:
        table_prefix: Prefix of the tables to combine (e.g., 'cleaned_transcript_oi_batch_')
        output_table_name: Name of the final combined table
        output_schema: Schema for the output table (optional)
        source_schema: Schema where the batch tables are located (optional)
        order_by_col: Column(s) to order the final table by
        drop_duplicates: Whether to remove duplicate rows
    """
    spark = get_spark_session()
   
    # Get list of all tables with the specified prefix
    if source_schema:
        tables = spark.sql(f"SHOW TABLES IN {source_schema}").collect()
        matching_tables = [
            row.tableName for row in tables 
            if row.tableName.startswith(table_prefix)
        ]
    else:
        tables = spark.sql("SHOW TABLES").collect()
        matching_tables = [
            row.tableName for row in tables 
            if row.tableName.startswith(table_prefix)
        ]
    
    if not matching_tables:
        raise ValueError(f"No tables found with prefix '{table_prefix}'")
    
    print(f"🔍 Found {len(matching_tables)} tables to combine:")
    for table in sorted(matching_tables):
        print(f"   - {table}")
    
    # Read first table to establish the standard column schema
    first_table = sorted(matching_tables)[0]
    full_first_table_name = f"{source_schema}.{first_table}" if source_schema else first_table
    first_df = spark.table(full_first_table_name)
    standard_columns = first_df.columns
    
    print(f"📋 Using column order from {first_table}:")
    print(f"   Columns: {standard_columns}")
    
    # Verify all tables have the same columns (but potentially different order)
    print("🔍 Verifying column consistency across all tables...")
    for table_name in matching_tables:
        full_table_name = f"{source_schema}.{table_name}" if source_schema else table_name
        table_df = spark.table(full_table_name)
        table_columns = set(table_df.columns)
        standard_columns_set = set(standard_columns)
        
        if table_columns != standard_columns_set:
            missing_cols = standard_columns_set - table_columns
            extra_cols = table_columns - standard_columns_set
            error_msg = f"Schema mismatch in table {table_name}:\n"
            if missing_cols:
                error_msg += f"  Missing columns: {missing_cols}\n"
            if extra_cols:
                error_msg += f"  Extra columns: {extra_cols}\n"
            raise ValueError(error_msg)
        
        # Check if column order is different
        if table_df.columns != standard_columns:
            print(f"   ⚠️  {table_name} has different column order - will standardize")
        else:
            print(f"   ✅ {table_name} has matching column order")
    
    # Read and union all tables with standardized column order
    combined_df = None
    total_rows = 0
    
    for i, table_name in enumerate(sorted(matching_tables)):
        full_table_name = f"{source_schema}.{table_name}" if source_schema else table_name
        
        print(f"📖 Reading table {i+1}/{len(matching_tables)}: {full_table_name}")
        
        table_df = spark.table(full_table_name)
        
        # Standardize column order by selecting columns in the standard order
        # This ensures union operations are safe regardless of original column order
        standardized_df = table_df.select(*standard_columns)
        
        table_row_count = standardized_df.count()
        total_rows += table_row_count
        
        print(f"   Rows: {table_row_count:,}")
        
        if combined_df is None:
            combined_df = standardized_df
        else:
            # Now safe to union since all DataFrames have identical column order
            combined_df = combined_df.union(standardized_df)
    
    print(f"\n📊 Total rows before processing: {total_rows:,}")
    
    # Drop duplicates if requested
    if drop_duplicates:
        print("🔄 Removing duplicates...")
        before_count = combined_df.count()
        combined_df = combined_df.dropDuplicates()
        after_count = combined_df.count()
        print(f"   Rows after deduplication: {after_count:,} (removed {before_count - after_count:,})")
    
    # Order the final table if specified
    if order_by_col:
        print(f"🔄 Ordering by: {order_by_col}")
        if isinstance(order_by_col, str):
            combined_df = combined_df.orderBy(F.col(order_by_col))
        else:
            combined_df = combined_df.orderBy(*[F.col(c) for c in order_by_col])
    
    # Write the combined table
    output_full_name = f"{output_schema}.{output_table_name}" if output_schema else output_table_name
    
    print(f"💾 Writing combined table: {output_full_name}")
    combined_df.write.mode("overwrite").saveAsTable(output_full_name)
    
    # Verify the final table
    final_count = spark.table(output_full_name).count()
    print(f"✅ Successfully created {output_full_name} with {final_count:,} rows")
    print(f"📋 Final table column order: {spark.table(output_full_name).columns}")

def union_with_historical_data(
    table_new: DataFrame, 
    table_historical: DataFrame, 
    join_keys: List[str] = ["id1", "id2"],
    date_col: str = "date",
    sequence_col: str = "record_sequence",
    only_with_history: bool = False
) -> DataFrame:
    """
    Union table_new with historical data from table_historical and add a sequence column to track 
    chronological order. Table_new records get sequence=1, and historical records get 
    sequence values 2, 3, etc. based on chronological order (most recent historical = 2).
    
    Smart Filtering Logic:
    1. Only historical records from table_historical that have matching join_keys in table_new are included
    2. Only historical records where date_col < corresponding table_new date are included
    3. This ensures we get truly historical data that's both relevant and chronologically valid
    
    Safe Schema Handling:
    - Missing columns are filled with null values to enable clean union operations
    - Column ordering is standardized before union to prevent schema mismatch errors
        
    Parameters:
        table_new (DataFrame): The primary DataFrame (current/latest records)
        table_historical (DataFrame): The DataFrame containing historical data
        join_keys (List[str]): List of column names to group by. Default is ["id1", "id2"]
        date_col (str): The date column name to order by. Default is "date"
        sequence_col (str): Name for the sequence column. Default is "record_sequence"
        only_with_history (bool): If True, only keep records from table_new that have 
                                 matching historical records. If False, keep all records 
                                 from table_new. Default is False.
    
    Returns:
        DataFrame: Combined DataFrame with sequence column showing chronological order,
                  containing records based on only_with_history parameter,
                  ordered by join_keys then sequence_col for easy timeline viewing        

    """
    # Validate inputs
    if not isinstance(join_keys, list):
        raise ValueError("join_keys must be a list of column names")
    
    # Check if required columns exist
    missing_cols_new = [col for col in join_keys + [date_col] if col not in table_new.columns]
    missing_cols_hist = [col for col in join_keys + [date_col] if col not in table_historical.columns]
    
    if missing_cols_new:
        raise ValueError(f"Missing columns in table_new: {missing_cols_new}")
    if missing_cols_hist:
        raise ValueError(f"Missing columns in table_historical: {missing_cols_hist}")
    
    # Check if sequence column name conflicts with existing columns
    if sequence_col in table_new.columns or sequence_col in table_historical.columns:
        raise ValueError(f"Sequence column name '{sequence_col}' already exists in one of the tables")
    
    # Get all columns from both tables in a deterministic order
    # Use dict.fromkeys() to maintain order of first appearance while removing duplicates
    all_columns = list(dict.fromkeys(table_new.columns + table_historical.columns))
    
    # Get table_new with only the join_keys and date for comparison
    table_new_dates = table_new.select(*join_keys, date_col).alias("tn")
    
    # Join table_historical with table_new_dates to get matching keys and compare dates
    table_historical_aliased = table_historical.alias("th")
    
    # Join on the key columns and filter where table_historical date < table_new date
    table_historical_filtered = table_historical_aliased.join(
        table_new_dates, 
        join_keys, 
        "inner"
    ).filter(
        F.col(f"th.{date_col}") < F.col(f"tn.{date_col}")
    ).select("th.*")
    
    # Apply filtering logic based on only_with_history parameter
    if only_with_history:
        # Only keep records from table_new that have matching historical records
        keys_with_history = table_historical_filtered.select(*join_keys).distinct()
        table_new_filtered = table_new.join(keys_with_history, join_keys, "inner")
    else:
        # Keep all records from table_new (default behavior)
        table_new_filtered = table_new
    
    # Add missing columns to each table with null values
    table_new_normalized = table_new_filtered
    table_historical_normalized = table_historical_filtered
    
    for col in all_columns:
        if col not in table_new_filtered.columns:
            table_new_normalized = table_new_normalized.withColumn(col, F.lit(None))
        if col not in table_historical_filtered.columns:
            table_historical_normalized = table_historical_normalized.withColumn(col, F.lit(None))
    
    # Ensure both tables have the same column order
    table_new_normalized = table_new_normalized.select(*all_columns)
    table_historical_normalized = table_historical_normalized.select(*all_columns)
    
    # Add source identifier to track origin
    table_new_with_source = table_new_normalized.withColumn("_source", F.lit("current"))
    table_historical_with_source = table_historical_normalized.withColumn("_source", F.lit("historical"))
    
    # Union the tables
    combined_df = table_new_with_source.union(table_historical_with_source)
    
    # Create window specification for ranking by date within each group
    window_spec = Window.partitionBy(*join_keys).orderBy(F.desc(date_col))
    
    # Add sequence number using row_number
    result = combined_df.withColumn(
        sequence_col, 
        F.row_number().over(window_spec)
    ).drop("_source")
    
    # Order the result by join_keys and then sequence for intuitive timeline view
    result = result.orderBy(*join_keys, sequence_col)
    
    return result


def count_delimited_items(col_name: str, delimiter: str = ",", distinct: bool = False):
    """
    Count number of items in delimited string column.
    
    Parameters:
        col_name (str): Name of the column containing delimited string
        delimiter (str, optional): Delimiter to split on. Default is ",".
        distinct (bool, optional): If True, count only unique items. Default is False.
    
    Returns:
        pyspark.sql.Column: New column with count of delimited items
    
    Examples:
        >>> # Sample data with comma-delimited codes
        >>> data = [
        ...     ("user1", "A001,B002,C003"),        # 3 items
        ...     ("user2", "X001"),                  # 1 item
        ...     ("user3", ""),                      # 1 item (empty string becomes [""])
        ...     ("user4", "A001,B002,A001,C003"),   # 4 items (3 distinct)
        ... ]
        >>> df = spark.createDataFrame(data, ["user", "codes"])
        
        >>> # Using default comma delimiter (total count)
        >>> df.select("*", count_delimited_items("codes")).show()
        >>> # +-----+-------------------+------------+
        >>> # | user|              codes| codes_count|
        >>> # +-----+-------------------+------------+
        >>> # |user1|   A001,B002,C003|          3|
        >>> # |user2|               X001|          1|
        >>> # |user3|                   |          1|
        >>> # |user4| A001,B002,A001,C003|          4|
        >>> # +-----+-------------------+------------+
        
        >>> # Using distinct count
        >>> df.select("*", count_delimited_items("codes", distinct=True)).show()
        >>> # +-----+-------------------+------------+
        >>> # | user|              codes| codes_count|
        >>> # +-----+-------------------+------------+
        >>> # |user1|   A001,B002,C003|          3|
        >>> # |user2|               X001|          1|
        >>> # |user3|                   |          1|
        >>> # |user4| A001,B002,A001,C003|          3|  # Duplicates removed
        >>> # +-----+-------------------+------------+
        
        >>> # Using semicolon delimiter with distinct count
        >>> data_semicolon = [("user1", "A001;B002;A001;C003")]
        >>> df_semi = spark.createDataFrame(data_semicolon, ["user", "codes"])
        >>> df_semi.select("*", count_delimited_items("codes", ";", distinct=True)).show()
        >>> # +-----+-------------------+------------+
        >>> # | user|              codes| codes_count|
        >>> # +-----+-------------------+------------+
        >>> # |user1| A001;B002;A001;C003|          3|  # A001 appears twice, counted once
        >>> # +-----+-------------------+------------+
    """
    if distinct:
        # Split, convert to set to remove duplicates, then count
        return F.size(F.array_distinct(F.split(F.col(col_name), delimiter))).alias(f"{col_name}_count")
    else:
        # Original behavior - count all items including duplicates
        return F.size(F.split(F.col(col_name), delimiter)).alias(f"{col_name}_count")


def add_delimited_codes_descriptions(
    df: DataFrame, 
    col_name: str, 
    dim_df: DataFrame, 
    code_col: str, 
    desc_col: str, 
    delimiter: str = ",",
    distinct: bool = False,
    output_col_name: Optional[str] = None
) -> DataFrame:
    """
    Add a column with descriptions for delimited codes using a dimension table.
    
    Parameters:
        df (DataFrame): The main DataFrame containing delimited codes
        col_name (str): Name of the column containing delimited codes
        dim_df (DataFrame): Dimension table containing code-to-description mappings
        code_col (str): Column name in dim_df containing the codes
        desc_col (str): Column name in dim_df containing the descriptions
        delimiter (str, optional): Delimiter used in the codes column. Default is ",".
        distinct (bool, optional): If True, remove duplicate descriptions. Default is False.
        output_col_name (str, optional): Name for the new descriptions column. 
                                       If None, uses "{col_name}_desc".
    
    Returns:
        DataFrame: Original DataFrame with added descriptions column
    
    Examples:
        >>> # Sample main data with delimited codes
        >>> main_data = [
        ...     ("user1", "A001,B002,C003"),
        ...     ("user2", "X001"),
        ...     ("user3", "A001,B002,A001"),  # Duplicate codes
        ...     ("user4", "Z999,B002"),       # Z999 not in dim table
        ... ]
        >>> df = spark.createDataFrame(main_data, ["user", "codes"])
        
        >>> # Dimension table with code mappings
        >>> dim_data = [
        ...     ("A001", "Product Alpha"),
        ...     ("B002", "Product Beta"),
        ...     ("C003", "Product Gamma"),
        ...     ("X001", "Product X"),
        ... ]
        >>> dim_df = spark.createDataFrame(dim_data, ["code", "description"])
        
        >>> # Map codes to descriptions (with duplicates)
        >>> result = add_delimited_codes_descriptions(
        ...     df, "codes", dim_df, "code", "description"
        ... )
        >>> result.show(truncate=False)
        >>> # +-----+-------------------+------------------------------------------+
        >>> # | user|              codes|                            codes_desc|
        >>> # +-----+-------------------+------------------------------------------+
        >>> # |user1|   A001,B002,C003|Product Alpha,Product Beta,Product Gamma|
        >>> # |user2|               X001|                             Product X|
        >>> # |user3|     A001,B002,A001|  Product Alpha,Product Beta,Product Alpha|
        >>> # |user4|         Z999,B002|                        null,Product Beta|
        >>> # +-----+-------------------+------------------------------------------+
        
        >>> # Map codes to descriptions (distinct only)
        >>> result_distinct = add_delimited_codes_descriptions(
        ...     df, "codes", dim_df, "code", "description", distinct=True
        ... )
        >>> result_distinct.show(truncate=False)
        >>> # +-----+-------------------+--------------------------------+
        >>> # | user|              codes|                     codes_desc|
        >>> # +-----+-------------------+--------------------------------+
        >>> # |user1|   A001,B002,C003|Product Alpha,Product Beta,Product Gamma|
        >>> # |user2|               X001|                       Product X|
        >>> # |user3|     A001,B002,A001|      Product Alpha,Product Beta|  # Duplicates removed
        >>> # |user4|         Z999,B002|                  Product Beta|  # null filtered out
        >>> # +-----+-------------------+--------------------------------+
        
        >>> # Custom output column name with semicolon delimiter
        >>> result_custom = add_delimited_codes_descriptions(
        ...     df, "codes", dim_df, "code", "description", 
        ...     delimiter=";", output_col_name="product_names"
        ... )
    """
    # Validate inputs
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in main DataFrame")
    
    if code_col not in dim_df.columns:
        raise ValueError(f"Column '{code_col}' not found in dimension DataFrame")
    
    if desc_col not in dim_df.columns:
        raise ValueError(f"Column '{desc_col}' not found in dimension DataFrame")
    
    # Set output column name
    if output_col_name is None:
        output_col_name = f"{col_name}_desc"
    
    if output_col_name in df.columns:
        raise ValueError(f"Output column '{output_col_name}' already exists in DataFrame")
    
    # Create unique row identifier for grouping back
    df_with_id = df.withColumn("_row_id", F.monotonically_increasing_id())
    
    # Split delimited codes into individual rows
    df_exploded = df_with_id.withColumn(
        "_individual_code", 
        F.explode(F.split(F.trim(F.col(col_name)), delimiter))
    )
    
    # Trim whitespace from individual codes
    df_exploded = df_exploded.withColumn(
        "_individual_code", 
        F.trim(F.col("_individual_code"))
    )
    
    # Join with dimension table to get descriptions
    df_mapped = df_exploded.join(
        dim_df, 
        df_exploded._individual_code == dim_df[code_col], 
        "left"
    )
    
    # Group back by original rows and concatenate descriptions
    if distinct:
        # Remove nulls and duplicates, then concatenate
        agg_expr = F.concat_ws(
            delimiter, 
            F.array_distinct(
                F.filter(
                    F.collect_list(desc_col), 
                    lambda x: x.isNotNull()
                )
            )
        )
    else:
        # Keep all descriptions including nulls, then concatenate
        agg_expr = F.concat_ws(delimiter, F.collect_list(desc_col))
    
    # Get all original columns except the row_id
    original_cols = [c for c in df.columns]
    
    df_result = df_mapped.groupBy("_row_id", *original_cols) \
                        .agg(agg_expr.alias(output_col_name)) \
                        .drop("_row_id")
    
    return df_result


def cleanup_tables_by_prefix(
    schema_name: str,
    table_prefix: str,
    catalog_name: Optional[str] = None,
    dry_run: bool = True,
    confirm_deletion: bool = True
) -> Dict[str, Any]:
    """
    Clean up all tables that start with a specific prefix in a Unity Catalog schema.
    
    Parameters:
    -----------
    schema_name : str
        Schema name in Unity Catalog
    table_prefix : str
        Prefix to match table names (e.g., "batch_", "temp_", "processed_")
    catalog_name : str, optional
        Catalog name. If None, uses current catalog
    dry_run : bool, default=True
        If True, only shows what would be deleted without actually deleting
    confirm_deletion : bool, default=True
        If True, asks for confirmation before deleting (only when dry_run=False)
        
    Returns:
    --------
    Dict[str, Any]
        Summary of cleanup operation
        
    Examples:
    ---------
    >>> # Dry run to see what would be deleted
    >>> cleanup_tables_by_prefix("my_schema", "batch_", dry_run=True)
    
    >>> # Actually delete tables with confirmation
    >>> cleanup_tables_by_prefix("my_schema", "batch_", dry_run=False)
    
    >>> # Delete without confirmation (use with caution)
    >>> cleanup_tables_by_prefix("my_schema", "temp_", dry_run=False, confirm_deletion=False)
    """    
    spark = get_spark_session()
   
    # Construct schema reference
    if catalog_name:
        schema_ref = f"{catalog_name}.{schema_name}"
    else:
        schema_ref = schema_name
    
    print(f"🔍 Searching for tables with prefix '{table_prefix}' in schema: {schema_ref}")
    
    try:
        # Get all tables in the schema
        tables_df = spark.sql(f"SHOW TABLES IN {schema_ref}")
        all_tables = [row['tableName'] for row in tables_df.collect()]
        
        # Filter tables by prefix
        matching_tables = [table for table in all_tables if table.startswith(table_prefix)]
        
        print(f"📊 Found {len(matching_tables)} tables matching prefix '{table_prefix}':")
        for table in matching_tables:
            print(f"   - {table}")
        
        if not matching_tables:
            print("✅ No tables found to clean up.")
            return {
                'total_tables_found': 0,
                'tables_deleted': 0,
                'tables_failed': 0,
                'dry_run': dry_run
            }
        
        # Dry run - just show what would be deleted
        if dry_run:
            print("\n🔍 DRY RUN MODE - No tables will be deleted")
            print("   To actually delete these tables, set dry_run=False")
            return {
                'total_tables_found': len(matching_tables),
                'tables_to_delete': matching_tables,
                'tables_deleted': 0,
                'tables_failed': 0,
                'dry_run': True
            }
        
        # Confirmation prompt
        if confirm_deletion:
            print(f"\n⚠️  WARNING: About to delete {len(matching_tables)} tables!")
            response = input("Type 'DELETE' to confirm deletion: ")
            if response != 'DELETE':
                print("❌ Deletion cancelled.")
                return {
                    'total_tables_found': len(matching_tables),
                    'tables_deleted': 0,
                    'tables_failed': 0,
                    'dry_run': False,
                    'cancelled': True
                }
        
        # Actually delete tables
        print(f"\n🗑️  Deleting {len(matching_tables)} tables...")
        deleted_tables = []
        failed_tables = []
        
        for table in matching_tables:
            try:
                full_table_name = f"{schema_ref}.{table}"
                spark.sql(f"DROP TABLE IF EXISTS {full_table_name}")
                deleted_tables.append(table)
                print(f"   ✅ Deleted: {table}")
                
            except Exception as e:
                failed_tables.append({'table': table, 'error': str(e)})
                print(f"   ❌ Failed to delete {table}: {str(e)}")
        
        
        # Summary
        print("\n📊 CLEANUP SUMMARY:")
        print(f"   Schema: {schema_ref}")
        print(f"   Prefix: {table_prefix}")
        print(f"   Tables found: {len(matching_tables)}")
        print(f"   Successfully deleted: {len(deleted_tables)}")
        print(f"   Failed deletions: {len(failed_tables)}")
        
        if failed_tables:
            print("\n❌ Failed deletions:")
            for failure in failed_tables:
                print(f"   - {failure['table']}: {failure['error']}")
        
        return {
            'total_tables_found': len(matching_tables),
            'tables_deleted': len(deleted_tables),
            'tables_failed': len(failed_tables),
            'deleted_tables': deleted_tables,
            'failed_tables': failed_tables,
            'dry_run': False
        }
        
    except Exception as e:
        error_msg = f"Error during cleanup operation: {str(e)}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg) from e

