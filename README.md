# Predicting Home Prices in California

## Overview

This project aims to predict the cost of homes in California using machine learning techniques. The dataset contains
various features related to housing attributes such as location, age, income, and proximity to the ocean. The primary
objective is to build predictive models that can accurately estimate the median house value based on these features.

## Features

The dataset consists of the following feature variables:

1. **Longitude**: A measure of how far west a house is.
2. **Latitude**: A measure of how far north a house is.
3. **Housing Median Age**: Median age of a house within a block.
4. **Total Rooms**: Total number of rooms within a block.
5. **Total Bedrooms**: Total number of bedrooms within a block.
6. **Population**: Total number of people residing within a block.
7. **Households**: Total number of households within a block.
8. **Median Income**: Median income for households within a block.
9. **Ocean Proximity**: Location of the house in proximity to the ocean.

## Target Variable

- **Median House Value**: Median house value for households within a block.

## Methodology

The project follows the following steps:

1. **Data Acquisition & Preprocessing**: Obtain the dataset and preprocess it by handling missing values and performing
   feature engineering.

2. **Exploratory Data Analysis (EDA)**: Visualize the data distribution and analyze correlations between features.

3. **Model Training**: Train various machine learning models including Linear Regression, Random Forest Regression, and
   XGBoost Regression using the preprocessed data.

4. **Model Evaluation**: Evaluate the trained models using metrics such as R-squared, Root Mean Squared Error (RMSE),
   and Mean Absolute Error (MAE).

5. **Cross Validation & Hyperparameter Tuning**: Validate model performance using cross-validation and fine-tune
   hyperparameters to improve model accuracy.

## Libraries Used

- Pandas: Data manipulation and preprocessing.
- Matplotlib & Seaborn: Data visualization.
- Scikit-learn: Model training, evaluation, and preprocessing.
- NumPy: Numerical computing.
- XGBoost: Implementation of the gradient boosting algorithm.

## Instructions

1. Ensure you have Python installed along with necessary libraries mentioned in the project.
2. Download the dataset (housing.csv) and place it in the project directory.
3. Open the Jupyter notebook (Predict_Home_Prices_California.ipynb) to explore the code and execute the cells.
4. Follow the instructions and comments within the notebook to understand each step of the project.

## Results

The project achieves a predictive model capable of estimating home prices in California with a certain degree of
accuracy. Various machine learning algorithms are evaluated, and the best-performing model is selected based on
evaluation metrics.

## Credits

This project was developed by [Your Name] as part of [Any relevant course or organization].


------------

**Running streamlit:**

To launch streamlit, use the following command:

```bash
python -m streamlit.cli run app.py
```

----------------


**Running Jupyter Notebook:**

To launch the Jupyter Notebook server, use the following command:

```bash
jupyter notebook
```

(Note: Use Control-C to stop the server)

---

**Installing Dependencies:**

Ensure that the required dependencies are installed by running the following commands:

```bash
pip install -r requirements.txt
python -m pip install jupyter
```

---

**Memory Profiling:**

To profile the memory usage, decorate your Python script with `@memory_profiler.profile` and run the following command:

```bash
python -m memory_profiler main.py
```

---

**Line Profiling:**

For line-level profiling, use the `line_profiler_pycharm` package. Decorate your Python script with `@profile` and
execute the following command:

```bash
python -m line_profiler main.py.lprof > results.txt
```

Make sure to replace `main.py` with the appropriate filename.