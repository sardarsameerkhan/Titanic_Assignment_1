import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report
import os

def evaluate():
    print("Evaluating Model Performance...")
    
    # Load data
    df = pd.read_csv('features/final_features.csv')
    
    # Load model
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize']
    X = df[features]
    y_true = df['Survived']
    
    # Generate Predictions
    predictions = model.predict(X)
    
    # Calculate Metrics
    acc = accuracy_score(y_true, predictions)
    # This generates precision, recall, and f1-score for both classes (0 and 1)
    report = classification_report(y_true, predictions)
    
    os.makedirs('results', exist_ok=True)
    
    # 1. Save Metrics to text file
    with open('results/metrics.txt', 'w') as f:
        f.write("--- Titanic Model Evaluation ---\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report)

    # 2. Save Predictions to CSV
    df['Predictions'] = predictions
    df.to_csv('results/predictions.csv', index=False)
    
    print(f"DONE! Accuracy: {acc:.2f}")
    print("Full report (Precision, Recall, F1) saved in results/metrics.txt")

if __name__ == "__main__":
    evaluate()