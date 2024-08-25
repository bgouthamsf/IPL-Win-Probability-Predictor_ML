import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pickle

class ModelTrainer:
    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, drop='first', categories=[
            ['Mumbai Indians', 'Gujarat Titans', 'Kolkata Knight Riders', 'Delhi Capitals', 'Sunrisers Hyderabad',
             'Royal Challengers Bangalore', 'Lucknow Super Giants', 'Rajasthan Royals', 'Chennai Super Kings',
             'Punjab Kings'],
            ['Mumbai Indians', 'Gujarat Titans', 'Kolkata Knight Riders', 'Delhi Capitals', 'Sunrisers Hyderabad',
             'Royal Challengers Bangalore', 'Lucknow Super Giants', 'Rajasthan Royals', 'Chennai Super Kings',
             'Punjab Kings'],
            ['Mumbai', 'Chennai', 'Bangalore', 'Ahmedabad', 'Hyderabad', 'Kolkata', 'Visakhapatnam', 'Indore', 'Durban',
             'Chandigarh', 'Delhi', 'Dharamsala', 'Ranchi', 'Nagpur', 'Mohali', 'Pune', 'Bengaluru', 'Jaipur',
             'Port Elizabeth', 'Centurion', 'Raipur', 'Sharjah', 'Cuttack', 'Johannesburg', 'Cape Town', 'East London',
             'Abu Dhabi', 'Kimberley', 'Bloemfontein', 'Guwahati', 'Navi Mumbai', 'Dubai', 'Lucknow']
        ])

    def preprocess_data(self, df):
        column_transformer = ColumnTransformer(
            transformers=[
                ('encoder', self.encoder, ['batting_team', 'bowling_team', 'city'])
            ],
            remainder='passthrough'
        )
        return column_transformer.fit_transform(df)

    def train_model(self, X, y):
        column_transformer = ColumnTransformer(
            transformers=[
                ('encoder', self.encoder, ['batting_team', 'bowling_team', 'city'])
            ],
            remainder='passthrough'
        )
        pipeline = Pipeline(
            steps=[
                ('preprocessor', column_transformer),
                ('classifier', RandomForestClassifier())
            ]
        )
        pipeline.fit(X, y)
        return pipeline

if __name__ == '__main__':
    data = pd.read_csv('FinalIPLDataSet.csv')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    trainer = ModelTrainer()
    model = trainer.train_model(X, y)

    with open('model_pipeline.pkl', 'wb') as file:
        pickle.dump(model, file)
