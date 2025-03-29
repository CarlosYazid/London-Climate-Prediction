# London Climate Prediction

![ML Framework](https://img.shields.io/badge/scikit--learn?style=flat&logo=scikit-learn&labelColor=1.4.1&color=orange)

## 1. Project Name
**London Climate Prediction** - Machine learning models for predicting London's average temperature based on historical weather data.

## 2. Brief Description
This project implements various regression models to predict London's mean temperature using historical weather data. The solution leverages scikit-learn for machine learning and MLflow for experiment tracking, model management, and deployment. The system evaluates multiple algorithms (Linear Regression, Decision Trees, Random Forests) with different hyperparameters to determine the most accurate temperature prediction model.

## 3. Main Features
- **Data Processing Pipeline**: Handles missing values, feature scaling, and data type optimization
- **Multiple Model Comparison**: Evaluates Linear Regression, Decision Tree, and Random Forest models
- **Experiment Tracking**: Uses MLflow to log parameters, metrics, and models
- **Feature Engineering**: Extracts temporal features (month, year) from date data
- **Performance Metrics**: Calculates RMSE for model evaluation
- **Reproducible Experiments**: Tracks all experiment parameters and data transformations
- **Model Signatures**: Defines explicit input/output schemas for deployment

## 4. Prerequisites
Before running this project, ensure you have the following installed:

### System Requirements
- Python 3.8+
- pip package manager

### Python Packages (see full list in requirements.txt)
Core dependencies:
- pandas (>=2.2.3)
- numpy (>=1.26.4)
- scikit-learn (>=1.4.1)
- mlflow (>=2.20.1)
- matplotlib (>=3.10.0)
- seaborn (>=0.13.2)

## 5. Installation

### Option 1: Using pip
```bash
# Clone the repository
git clone https://github.com/CarlosYazid/London-Climate-Prediction.git
cd London-Climate-Prediction

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install requirements
pip install -r requirements.txt
```

### Option 2: Using conda
```bash
# Create conda environment
conda create -n london-climate python=3.10
conda activate london-climate

# Install core packages
conda install -c conda-forge pandas numpy scikit-learn mlflow matplotlib seaborn
```

## 6. Usage

### Running the Prediction Pipeline
1. Ensure you have the data file `london_weather.csv` in the project directory
2. Start MLflow tracking server:
```bash
mlflow ui
```
3. Run the main notebook/script:
```bash
jupyter notebook
```
Then open and run the notebook cells sequentially.

### Key Functions
- Data loading and preprocessing:
```python
weather = pd.read_csv("london_weather.csv", parse_dates=[0], date_format="%Y%m%d")
weather['month'] = weather['date'].dt.month
weather['year'] = weather['date'].dt.year
```

- Model training and evaluation:
```python
with mlflow.start_run(run_name=run_name):
    model = RandomForestRegressor(max_depth=depth).fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    mlflow.log_metric("rmse", rmse)
```

## 7. Examples

### Example 1: Basic Training Run
```python
# After data preparation
with mlflow.start_run():
    model = DecisionTreeRegressor(max_depth=5)
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "model")
```

### Example 2: Loading a Saved Model
```python
loaded_model = mlflow.sklearn.load_model("runs:/<RUN_ID>/model")
predictions = loaded_model.predict(new_data)
```

## 8. Project Structure
```
London-Climate-Prediction/
├── .gitignore            - Specifies intentionally untracked files
├── london_weather.csv    - Primary dataset (not included in repo)
├── requirements.txt      - Full list of Python dependencies
├── tower_bridge.jpeg     - Sample image for documentation
└── notebook.ipynb        - Main Jupyter notebook with all code
```

## 9. API Reference

### Data Processing Functions
- `change_datatype(X_train, X_test, y_train, y_test)`  
  Converts data types for memory optimization

### Model Training
- All scikit-learn regression model interfaces are supported
- MLflow tracking automatically captures:
  - Parameters (`mlflow.log_param()`)
  - Metrics (`mlflow.log_metric()`)
  - Models (`mlflow.sklearn.log_model()`)

## 10. How to Contribute

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guide
- Include tests for new features
- Update documentation accordingly
- Use descriptive commit messages

## 11. Troubleshooting

### Common Issues
**Issue:** Missing data file  
**Solution:** Ensure `london_weather.csv` is in the project root

**Issue:** Package version conflicts  
**Solution:** Create a fresh virtual environment and install exact versions from requirements.txt

**Issue:** MLflow server not starting  
**Solution:** Check if port 5000 is available or specify another port:
```bash
mlflow ui --port 5001
```

## 12. Changelog

### [1.0.0] - 2025-03-29
- Initial release
- Implemented Linear Regression, Decision Tree, and Random Forest models
- Added MLflow experiment tracking
- Completed data preprocessing pipeline

## 13. License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 14. Contact
For questions or support, please contact:

**Project Maintainer**: Carlos Yazid <br>
**Email**: contact@carlospadilla.co  
**GitHub Issues**: [Issues](https://github.com/CarlosYazid/London-Climate-Prediction/issues)