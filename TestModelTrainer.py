import unittest
import pandas as pd
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import ModelTrainer

class TestModelTrainer(unittest.TestCase):

    def setUp(self):
        self.trainer = ModelTrainer()
        self.sample_data = pd.DataFrame({
            'batting_team': ['Mumbai Indians', 'Chennai Super Kings'],
            'bowling_team': ['Chennai Super Kings', 'Mumbai Indians'],
            'city': ['Mumbai', 'Chennai'],
            'target': [200, 180],
            'score': [150, 120],
            'balls_left': [30, 60],
            'wickets': [3, 2]
        })
        self.sample_target = pd.Series([1, 0])

    def test_train_model(self):
        model = self.trainer.train_model(self.sample_data, self.sample_target)
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
