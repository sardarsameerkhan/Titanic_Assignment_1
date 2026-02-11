import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import os

def evaluate():
    print("Evaluating Model Accuracy...")
    
    df = pd.read_csv('features/final_features.csv')
    
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize']
    X = df[features]
    y_true = df['Survived']
    
    predictions = model.predict(X)
    
    acc = accuracy_score(y_true, predictions)
    
    os.makedirs('results', exist_ok=True)
    

    with open('results/metrics.txt', 'w') as f:
        f.write(f"Random Forest Accuracy Score: {acc:.4f}\n")

    df['Predictions'] = predictions
    df.to_csv('results/predictions.csv', index=False)
    
    print(f"DONE! Model Accuracy: {acc:.2f}")
    print("Results saved in results/metrics.txt")

if __name__ == "__main__":
    evaluate()