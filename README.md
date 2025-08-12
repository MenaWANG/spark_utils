# spark_utils

Welcome to `spark_utils`, a collection of utility functions for making PySpark development easier, faster, and cleaner! 🚀

## Why Spark Utils?

`spark_utils` is a dedicated module to house all those handy functions that you find yourself writing over and over again. By centralizing these utilities, you can:

- **Avoid Namespace Pollution**: Keep your PySpark functions separate from your general Python functions, avoiding any pesky name collisions.
- **Enhance Readability**: Clearly distinguish between your PySpark-specific logic and the rest of your codebase.
- **Boost Reusability**: Write once, use anywhere. Reuse common data transformations, I/O operations, and more across your projects.
- **Simplify Maintenance**: Update and manage your utilities in one place, making your life as a developer a whole lot easier.
- **Improve Performance**: Apply best practices consistently, optimizing your Spark jobs without breaking a sweat.

## What's Inside?

The `spark_utils` module is packed with useful functions for common PySpark tasks. Here's a sneak peek at some of the goodies you'll find:    

- [**Data Quality Checks**](data_quality_check_demo.ipynb): Handy tools to help you explore your data quickly with a focus on quality check.
- [**Data Clean & Filter**](data_cleaning_demo.ipynb): Handy functions to clean messy date and number fields, and to filter with complex logic.
- **Environment Setup**: Helper functions to set up your Spark environment:
  - `setup_pydantic_v2`: Configure a particular package version in Databricks Runtime (DBR) environments with custom installation paths when necessary.




## Installation

To get started, simply clone the repository and import `spark_utils` into your PySpark project:

```sh
git clone https://github.com/MenaWANG/spark_utils.git
```

## Contributing
We love contributions! If you have a utility function that you think would be a great addition, feel free to open a pull request. Let's make PySpark development smoother together! 🤗

## License
This project is licensed under the MIT License.


