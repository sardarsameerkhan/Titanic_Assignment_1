import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def train():
    print("Training Random Forest Model...")
    
    input_path = 'features/final_features.csv'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found! Run feature_engineering.py first.")
        return
        
    df = pd.read_csv(input_path)

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize']
    X = df[features]
    y = df['Survived']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    

    os.makedirs('models', exist_ok=True)
    
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    print("DONE! Model trained and saved in models/model.pkl")

if __name__ == "__main__":
    train()