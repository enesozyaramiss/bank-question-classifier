# Banking Customer Question Classification System

##  Overview

This project implements an automated system for classifying Bulgarian banking customer questions into appropriate departments. The system combines traditional machine learning with modern deep learning approaches to achieve **88.6% accuracy** on real-world data.

##  Key Results

- **Traditional Model (Logistic Regression):** 69.6% test accuracy
- **Deep Learning Model (MBERT):** 72.4% test accuracy  
- **Production Accuracy:** 88.6% on full dataset
- **13 Department Categories** with automated routing capability

##  Project Structure

```
BANK-QUESTION-CLASSIFIER/
├── data/
│   ├── processed/           # Clean train/test datasets
│   │   ├── train_data.csv   # 1,565 training samples
│   │   ├── test_data.csv    # 392 test samples
│   │   └── class_mapping.csv
│   └── raw/
│       └── Коментари за сортиране.xlsx  # Original dataset
├── models/
│   ├── analysis/            # Performance visualizations
│   ├── best_model_logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│   └── model_metadata.json
├── notebook/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb  
│   ├── 03_model_development.ipynb
│   ├── 04_model_analysis.ipynb
│   └── BONUS_MBERT-GoogleColab.ipynb
├── requirements.txt
└── README.md
```

Data Access & Security
Confidential Data Protection

Raw banking data is NOT included in this repository for security reasons
Files in data/raw/ are protected by .gitignore


##  Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd BANK-QUESTION-CLASSIFIER

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Notebooks

Execute notebooks in the following order:

```bash
# 1. Data Exploration
jupyter notebook notebook/01_data_exploration.ipynb

# 2. Data Preprocessing  
jupyter notebook notebook/02_data_preprocessing.ipynb

# 3. Model Development
jupyter notebook notebook/03_model_development.ipynb

# 4. Model Analysis
jupyter notebook notebook/04_model_analysis.ipynb
```

### 3. BONUS: Deep Learning Approach

For the MBERT experiment:
```bash
# Upload notebook/BONUS_MBERT-GoogleColab.ipynb to Google Colab
# Upload train_data.csv and test_data.csv when prompted
# Run all cells (requires GPU runtime)
```

##  Key Notebooks Overview

### `01_data_exploration.ipynb`
- **Purpose:** Exploratory data analysis of Bulgarian banking questions
- **Key Outputs:** Data quality assessment, class distribution analysis
- **What to expect:** 
  - Dataset overview (1,962 questions)
  - Class imbalance visualization (576:1 ratio)
  - Text length distribution analysis

### `02_data_preprocessing.ipynb`
- **Purpose:** Clean Bulgarian text and prepare modeling data
- **Key Outputs:** Clean train/test datasets, class consolidation
- **What to expect:**
  - Bulgarian text normalization
  - Class consolidation (15 → 13 departments)
  - Stratified train/test split

### `03_model_development.ipynb`
- **Purpose:** Train and compare multiple ML models
- **Key Outputs:** Best model selection, hyperparameter optimization
- **What to expect:**
  - 6 algorithm comparison
  - TF-IDF feature engineering
  - Model performance metrics
  - **Final Result:** 69.6% test accuracy

### `04_model_analysis.ipynb`
- **Purpose:** Advanced error analysis and visualization
- **Key Outputs:** Confusion matrices, confidence analysis, learning curves
- **What to expect:**
  - Detailed performance visualizations
  - Error pattern analysis
  - Production deployment insights

### `BONUS_MBERT-GoogleColab.ipynb` 
- **Purpose:** Deep learning approach with multilingual BERT
- **Key Outputs:** MBERT vs traditional comparison
- **What to expect:**
  - **MBERT Result:** 72.4% test accuracy (+4.0% improvement)
  - Resource usage comparison
  - Business decision framework

##  Using the Trained Model

### Load and Predict

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('models/best_model_logistic_regression.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Predict new questions
def predict_department(question_text):
    # Apply same preprocessing as training
    processed_text = clean_bulgarian_text(question_text)
    
    # Vectorize and predict
    text_vector = vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)[0]
    confidence = model.predict_proba(text_vector)[0].max()
    
    return prediction, confidence

# Example usage
question = "Колко е лихвата за потребителски кредит?"
department, confidence = predict_department(question)
print(f"Department: {department}, Confidence: {confidence:.3f}")
```

##  Expected Results

### Performance Metrics
- **Logistic Regression:** 69.6% accuracy, fast inference
- **MBERT:** 72.4% accuracy, higher computational cost
- **Production accuracy:** 88.6% on full dataset

### Model Files Generated
- **Traditional approach:** ~10MB total model size
- **MBERT approach:** ~620MB total model size

### Analysis Outputs
All notebooks generate visualizations in `models/analysis/`:
- `confusion_matrix.png` - Prediction accuracy by department  
- `performance_metrics.png` - F1 scores and precision/recall
- `confidence_analysis.png` - Confidence distribution analysis
- `learning_curves.png` - Overfitting and bias analysis

##  Requirements

### Software
- Python 3.8+
- Jupyter Notebook
- scikit-learn, pandas, numpy, matplotlib, seaborn

##  Conclusion

This system provides a production-ready solution for Bulgarian banking question classification with comprehensive traditional ML and deep learning evaluation. The traditional approach is recommended for immediate deployment due to optimal performance/resource balance.

Enes Ozyaramis.