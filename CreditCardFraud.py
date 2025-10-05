import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Global Variables
data = None
model = None
xTrain, xTest, yTrain, yTest = None, None, None, None

# Functions
def load_data():
    global data
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        data = pd.read_csv(file_path)
        messagebox.showinfo("Info", "Data loaded successfully!")
        data_summary()

def data_summary():
    if data is None:
        messagebox.showwarning("Warning", "No data loaded!")
        return
    summary_text.delete("1.0", tk.END)
    summary_text.insert(tk.END, f"Dataset Shape: {data.shape}\n\n")
    summary_text.insert(tk.END, f"Dataset Summary:\n{data.describe()}\n")
    summary_text.insert(tk.END, f"Columns: {', '.join(data.columns)}\n")

def preprocess_data():
    global xTrain, xTest, yTrain, yTest
    if data is None:
        messagebox.showwarning("Warning", "Load data before preprocessing!")
        return
    if 'Class' not in data.columns:
        messagebox.showerror("Error", "Dataset must have a 'Class' column for labels!")
        return
    X = data.drop(['Class'], axis=1)
    Y = data['Class']
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=42)
    messagebox.showinfo("Info", "Data preprocessed successfully!")

def train_model_threaded():
    if data is None or xTrain is None:
        messagebox.showwarning("Warning", "Preprocess the data before training!")
        return
    if model_var.get() == "Select Model":
        messagebox.showwarning("Warning", "Select a model to train!")
        return

    def train():
        global model
        # Show "Training in progress" message
        messagebox.showinfo("Info", "Training in progress... This may take a while.")
        
        model_choice = model_var.get()
        if model_choice == 'Random Forest':
            model = RandomForestClassifier()
        elif model_choice == 'Logistic Regression':
            model = LogisticRegression(max_iter=1000)
        elif model_choice == 'Decision Tree':
            model = DecisionTreeClassifier()

        # Train the model
        model.fit(xTrain, yTrain)

        # Notify the user that training is complete
        messagebox.showinfo("Info", "Model trained successfully!")

    # Run training in a separate thread
    threading.Thread(target=train).start()

def generate_confusion_matrix():
    if model is None:
        messagebox.showwarning("Warning", "Train the model before generating the confusion matrix!")
        return
    yPred = model.predict(xTest)
    cm = confusion_matrix(yTest, yPred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.show()

def generate_scatterplot():
    if data is None:
        messagebox.showwarning("Warning", "Load the data to generate scatterplot!")
        return
    x_col = scatter_x_var.get()
    y_col = scatter_y_var.get()
    if x_col not in data.columns or y_col not in data.columns:
        messagebox.showwarning("Warning", "Invalid columns for scatterplot!")
        return
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x=x_col, y=y_col, hue='Class', ax=ax)
    ax.set_title("Scatterplot")
    plt.show()

def generate_distribution_plot():
    if data is None:
        messagebox.showwarning("Warning", "Load the data to generate distribution plot!")
        return
    dist_col = dist_var.get()
    if dist_col not in data.columns:
        messagebox.showwarning("Warning", "Invalid column for distribution plot!")
        return
    fig, ax = plt.subplots()
    sns.histplot(data[dist_col], kde=True, ax=ax)
    ax.set_title("Distribution Plot")
    plt.show()

def custom_prediction():
    if model is None:
        messagebox.showwarning("Warning", "Train the model before making predictions!")
        return
    try:
        input_data = [float(val) for val in custom_input.get().split(",")]
        prediction = model.predict([input_data])[0]
        messagebox.showinfo("Prediction", f"The transaction is: {'Fraud' if prediction == 1 else 'Normal'}")
    except ValueError:
        messagebox.showerror("Error", "Invalid input! Provide numerical values separated by commas.")

# GUI Setup
root = tk.Tk()
root.title("Credit Card Fraud Detection")
root.geometry("900x600")

# Dataset Section
frame_data = ttk.LabelFrame(root, text="Dataset")
frame_data.pack(fill="both", padx=10, pady=5)

btn_load_data = ttk.Button(frame_data, text="Load Data", command=load_data)
btn_load_data.pack(side="left", padx=5, pady=5)

btn_preprocess_data = ttk.Button(frame_data, text="Preprocess Data", command=preprocess_data)
btn_preprocess_data.pack(side="left", padx=5, pady=5)

# Summary Section
summary_text = tk.Text(root, height=10, wrap="word")
summary_text.pack(fill="both", padx=10, pady=5)

# Model Training Section
frame_model = ttk.LabelFrame(root, text="Model Training")
frame_model.pack(fill="both", padx=10, pady=5)

model_var = tk.StringVar(value="Select Model")
model_menu = ttk.OptionMenu(frame_model, model_var, "Select Model", "Random Forest", "Logistic Regression", "Decision Tree")
model_menu.pack(side="left", padx=5, pady=5)

btn_train_model = ttk.Button(frame_model, text="Train Model", command=train_model_threaded)
btn_train_model.pack(side="left", padx=5, pady=5)

# Analysis Section
frame_analysis = ttk.LabelFrame(root, text="Analysis")
frame_analysis.pack(fill="both", padx=10, pady=5)

btn_confusion_matrix = ttk.Button(frame_analysis, text="Confusion Matrix", command=generate_confusion_matrix)
btn_confusion_matrix.pack(side="left", padx=5, pady=5)

# Scatterplot
scatter_x_var = tk.StringVar()
scatter_y_var = tk.StringVar()
frame_scatter = ttk.LabelFrame(root, text="Scatterplot")
frame_scatter.pack(fill="both", padx=10, pady=5)

ttk.Label(frame_scatter, text="X-Axis:").pack(side="left", padx=5)
scatter_x_entry = ttk.Entry(frame_scatter, textvariable=scatter_x_var)
scatter_x_entry.pack(side="left", padx=5)

ttk.Label(frame_scatter, text="Y-Axis:").pack(side="left", padx=5)
scatter_y_entry = ttk.Entry(frame_scatter, textvariable=scatter_y_var)
scatter_y_entry.pack(side="left", padx=5)

btn_scatterplot = ttk.Button(frame_scatter, text="Generate Scatterplot", command=generate_scatterplot)
btn_scatterplot.pack(side="left", padx=5, pady=5)

# Distribution Plot
dist_var = tk.StringVar()
frame_dist = ttk.LabelFrame(root, text="Distribution Plot")
frame_dist.pack(fill="both", padx=10, pady=5)

ttk.Label(frame_dist, text="Column:").pack(side="left", padx=5)
dist_entry = ttk.Entry(frame_dist, textvariable=dist_var)
dist_entry.pack(side="left", padx=5)

btn_distribution_plot = ttk.Button(frame_dist, text="Generate Distribution Plot", command=generate_distribution_plot)
btn_distribution_plot.pack(side="left", padx=5, pady=5)

# Custom Prediction
frame_prediction = ttk.LabelFrame(root, text="Custom Prediction")
frame_prediction.pack(fill="both", padx=10, pady=5)

custom_input = tk.StringVar()
ttk.Label(frame_prediction, text="Input (comma-separated):").pack(side="left", padx=5)
custom_input_entry = ttk.Entry(frame_prediction, textvariable=custom_input)
custom_input_entry.pack(side="left", padx=5)

btn_predict = ttk.Button(frame_prediction, text="Predict", command=custom_prediction)
btn_predict.pack(side="left", padx=5, pady=5)

root.mainloop()
