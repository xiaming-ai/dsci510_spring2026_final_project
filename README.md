# Transportation Data Pipeline

## Overview
This project contains two Python scripts designed to retrieve and process transportation-related data:

* **`api_access.py`:** Fetches data from an external API and saves it locally.
* **`clean_nhts_data.py`:** Cleans and processes the retrieved dataset (likely NHTS data) for further analysis.

These scripts are intended to work together in a simple data pipeline: first download the data, then clean it.

---

## File Descriptions

### 1. `api_access.py`
* **Purpose:** Connects to an external API, sends requests to retrieve data, and saves the response (likely JSON or CSV) to a local file.
* **Key Libraries Used:**
    * `requests` – for API calls
    * `pandas` – for handling tabular data
    * `json`, `os`, `sys` – for file handling and system operations
* **Output:** Raw dataset file (e.g., JSON or CSV)

### 2. `clean_nhts_data.py`
* **Purpose:** Loads the raw dataset, cleans and preprocesses the information, and outputs a refined dataset ready for analysis.
* **Key Libraries Used:**
    * `pandas` – for data manipulation
    * `os` – for file operations
* **Typical Operations:**
    * Removing missing or invalid values
    * Renaming columns
    * Filtering relevant records
    * Formatting data types
* **Output:** Cleaned dataset file (e.g., CSV)

---

## Requirements
Make sure you have **Python 3** installed. You will also need to install the required dependencies using pip:

```bash
pip install pandas requests
