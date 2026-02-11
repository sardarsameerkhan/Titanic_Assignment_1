Titanic Survival Prediction - Automated ML Pipeline
This project builds a Random Forest model to predict passenger survival using the Titanic dataset. It’s fully automated from data fetching to final evaluation using a Makefile to ensure the results are reproducible.

Project Structure
src/: Contains the Python logic.

download_data.py: Pulls the raw CSV.

preprocess.py: Handles missing values and data cleaning.

feature_engineering.py: Creates the FamilySize feature.

train.py: Trains the Random Forest model.

evaluate.py: Generates the final accuracy score and predictions.

data/: Holds both raw and cleaned data.

features/: Stores the final processed features used for training.

models/: Contains the saved model.pkl.

results/: Final output folder for metrics.txt and predictions.csv.

How to Run
Everything is automated through the Makefile.

1. Install Requirements
First, make sure you have the necessary libraries installed:
<<<<<<< HEAD
pip install -r requirements.txt

2. Execute Pipeline
To run the whole thing from start to finish, use the following command in your terminal:

.\make -f Makefile all

=======

Bash
pip install -r requirements.txt
2. Execute Pipeline
To run the whole thing from start to finish, use the following command in your terminal:

Bash
.\make -f Makefile all
>>>>>>> 83f1f5cfac9f4ad0bcc04d1e97235c3e33745bf6
Results
The model currently achieves an accuracy of 97.98%.

Predictions are saved to results/predictions.csv.

<<<<<<< HEAD
Performance summary is in results/metrics.txt.
=======
Performance summary is in results/metrics.txt.
>>>>>>> 83f1f5cfac9f4ad0bcc04d1e97235c3e33745bf6
