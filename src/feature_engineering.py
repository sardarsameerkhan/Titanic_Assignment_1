import pandas as pd
import os

def engineer_features():
    print("Starting Feature Engineering...")
    
    input_path = 'data/processed/cleaned_titanic.csv'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found! Run preprocess.py first.")
        return
        
    df = pd.read_csv(input_path)
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    os.makedirs('features', exist_ok=True)
    
    df.to_csv('features/final_features.csv', index=False)
    print("DONE! Features saved in features/final_features.csv")

if __name__ == "__main__":
    engineer_features()