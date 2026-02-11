import pandas as pd
import os
import requests

# This is the direct link to the data
URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

def download():
    if not os.path.exists('data/raw'):
        os.makedirs('data/raw')
        print("Created directory: data/raw")

    print("Attempting to download Titanic data...")
    
    try:
        df = pd.read_csv(URL)
        
        df.to_csv('data/raw/titanic.csv', index=False)
        print("DONE! File created at: data/raw/titanic.csv")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download()