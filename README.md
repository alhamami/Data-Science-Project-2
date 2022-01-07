# DSND-P2
## Disaster Response Pipeline Project
This project aims to analyze disaster data from [appen](https://appen.com/) to build a model for an API that classifies disaster messages.
### Project Components:
There are three components you'll need to complete for this project.
1. [ETL Pipeline](#ETL_Pipeline)
2. [ML Pipeline](#ML_Pipeline)
3. [Flask Web App](#Flask_Web_App)
4. [Github and Code Quality](#Github_and_Code_Quality)
5. [Instructions](#Instructions)

## ETL Pipeline <a name="ETL_Pipeline"></a>
In a Python script, `process_data.py`, i did a data cleaning pipeline that:
- Loads the `messages` and `categories` datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database
## ML Pipeline <a name="ML_Pipeline"></a>
In a Python script, `train_classifier.py`, i did a machine learning pipeline that:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

## Flask Web App <a name="Flask_Web_App"></a>
- Modify file paths for database and model
- Add data visualizations using Plotly in the web app.

## Github and Code Quality<a name="Github_and_Code_Quality"></a>
The project graded based on the following:
- Use of Git and Github
- Strong documentation
- Clean and modular code

### Instructions:<a name="Instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/












