import pandas as pd
import os

def preprocess():
    print("Starting Preprocessing...")

    if not os.path.exists('data/raw/titanic.csv'):
        print("Error: data/raw/titanic.csv not found!")
        return
        
    df = pd.read_csv('data/raw/titanic.csv')

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna('S')

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    cols_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    df_cleaned = df.drop(columns=cols_to_drop)

    os.makedirs('data/processed', exist_ok=True)
    df_cleaned.to_csv('data/processed/cleaned_titanic.csv', index=False)
    
    print("DONE! Data is now clean and saved in data/processed/cleaned_titanic.csv")

if __name__ == "__main__":
    preprocess()