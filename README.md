# Predicting Subscription Churn Using Behavioral Micro-Moments

## Overview

This project analyzes user behavioral data to predict subscription churn for a subscription service.  The analysis focuses on identifying specific in-app actions ("micro-moments") that strongly correlate with churn.  By pinpointing these critical actions, we aim to develop targeted interventions to improve customer retention rates. The analysis involves data cleaning, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

## Technologies Used

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## How to Run

1. **Install Dependencies:**  Ensure you have Python 3.x installed.  Then, install the required Python libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Analysis:** Execute the main script:

   ```bash
   python main.py
   ```

## Example Output

The script will print key findings from the analysis to the console, including summary statistics, model performance metrics (e.g., accuracy, precision, recall, F1-score), and insights derived from the feature importance analysis.  Additionally, the script will generate several visualization plots (e.g., churn rate over time, feature importance bar chart) saved as PNG files in the `output` directory.  These plots provide a visual representation of the data and the model's findings.  The exact output and plots will vary depending on the input data.


## Data
The project requires a dataset containing user behavioral data, including features that represent in-app actions and a target variable indicating whether the user churned.  This dataset is not included in this repository for privacy reasons, but the code is structured to work with a similarly formatted CSV file.  Please replace `data.csv` with your own data file.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request.


## License

[Specify your license here, e.g., MIT License]