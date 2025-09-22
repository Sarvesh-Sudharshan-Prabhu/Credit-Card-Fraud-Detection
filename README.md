# Credit Card Fraud Detection GUI

This project is a **Tkinter-based desktop application** for detecting fraudulent credit card transactions.  
It allows users to load datasets, preprocess data, train machine learning models, visualize results, and make custom predictions.

---

## 🚀 Features

- **Load CSV dataset**  
  - Upload your dataset and view shape, summary statistics, and available columns.  

- **Data Preprocessing**  
  - Splits data into training and testing sets (80/20 split).  
  - Requires a target column named **`Class`** (0 = Normal, 1 = Fraud).  

- **Model Training**  
  - Train one of the following models:  
    - Random Forest  
    - Logistic Regression  
    - Decision Tree  

- **Analysis Tools**  
  - Generate a **confusion matrix** heatmap.  
  - Create **scatterplots** with custom X and Y columns.  
  - Plot **distributions** of selected features.  

- **Custom Predictions**  
  - Enter comma-separated feature values and get a fraud/normal prediction.  

---

## 📦 Requirements

Make sure you have Python **3.8+** installed.  
Install dependencies with:

```bash
pip install -r requirements.txt
requirements.txt
nginx
Copy code
tk
pandas
numpy
matplotlib
seaborn
scikit-learn
🖥️ Usage
Clone this repository or download the script.

Run the application:

bash
Copy code
python fraud_detection_gui.py
Use the interface to:

Load Data → Select a CSV file.

Preprocess Data → Split into training/testing sets.

Train Model → Choose a model and train it.

Analyze Results → Generate confusion matrix, scatterplots, or distributions.

Predict → Enter custom input values to classify transactions.

📊 Example Workflow
Load a dataset (creditcard.csv recommended).

Preprocess the data → Splits into X_train, X_test, y_train, y_test.

Select Random Forest and train the model.

Generate confusion matrix → Visualize fraud detection accuracy.

Create scatterplots/distributions for feature exploration.

Enter custom values like:

Copy code
0.12, -1.35, 0.98, 2.14, ...
→ Output: Fraud or Normal.

⚠️ Notes
The dataset must include a Class column (binary: 0 = Normal, 1 = Fraud).

For large datasets, training may take a while.

Logistic Regression may require max_iter=1000 for convergence.

📌 Future Improvements
Add support for more ML models (SVM, XGBoost).

Save and load trained models.

Improve visualization options inside the Tkinter window.

Optimize UI responsiveness for large datasets.

👨‍💻 Author
Developed by Sarvesh Prabhu.
