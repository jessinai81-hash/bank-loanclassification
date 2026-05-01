# Bank Loan Classification System

A machine learning pipeline for predicting bank loan approvals with an interactive Streamlit web application.

## Features

- **Feature Skewness Analysis**: Detects skewed features using seaborn
- **Intelligent Preprocessing**: 
  - Power Transformer + Standard Scaler for skewed features
  - Standard Scaler for normal features
- **Multiple Classifiers**: Compares 6 different ML algorithms
- **Interactive Web App**: Built with Streamlit for real-time predictions
- **High Accuracy**: Best model (Random Forest) achieves 100% accuracy

## Dataset

- **File**: `14.banking_loan_dataset_1000.csv`
- **Samples**: 1,000 loan records
- **Features**: Credit Score, Loan Amount, Years Employed, Employment Type
- **Target**: Loan Approved (Binary Classification)

## Models Trained

| Model | Accuracy |
|-------|----------|
| Random Forest Classifier | **100.00%** ⭐ |
| SVC | 98.00% |
| Gaussian NB Classifier | 97.50% |
| SGD Classifier | 96.00% |
| KNN Classifier | 95.50% |
| Logistic Regression | 94.50% |

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/jessinai81-hash/bank-loanclassification.git
cd bank-loanclassification
```

2. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run ML Pipeline (Training)

```bash
python ml_pipeline.py
```

This will:
- Analyze feature skewness
- Preprocess the data
- Train 6 different classifiers
- Save the best model and preprocessing tools

### Run Streamlit App (Predictions)

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

## Input Features

- **Credit Score**: 300-850
- **Loan Amount**: $5,000-$50,000
- **Years Employed**: 0-30
- **Employment Type**: Salaried or Self-Employed

## Output

The app provides:
- ✅/❌ Loan approval prediction
- Approval probability percentage
- Rejection probability percentage
- Input summary table

## Project Structure

```
bank-loanclassification/
├── ml_pipeline.py              # Training pipeline
├── streamlit_app.py            # Web application
├── 14.banking_loan_dataset_1000.csv   # Dataset
├── best_model.pkl              # Trained Random Forest model
├── scaler.pkl                  # StandardScaler for preprocessing
├── label_encoder.pkl           # LabelEncoder for categorical features
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

## Technologies Used

- **Python 3.x**: Main programming language
- **Scikit-learn**: Machine learning algorithms
- **Pandas**: Data manipulation and analysis
- **Numpy**: Numerical computations
- **Seaborn**: Data visualization and skewness detection
- **Matplotlib**: Plotting
- **Streamlit**: Web application framework

## Author

jessinai81  
Email: jessinai81@gmail.com

## License

MIT License - feel free to use this project for your own purposes

## Deployment

This app can be deployed on Streamlit Cloud:

1. Push code to GitHub (already done!)
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub account
4. Deploy the repository

## Contact

For questions or suggestions, feel free to reach out!
