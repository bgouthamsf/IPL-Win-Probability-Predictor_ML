import unittest
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

class TestFlaskApp(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Enter Match Details', response.data)

    def test_predict(self):
        # Check if model_pipeline.pkl exists
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model_pipeline.pkl'))
        if not os.path.exists(model_path):
            self.skipTest("model_pipeline.pkl does not exist, skipping the test.")
        
        response = self.app.post('/predict', data=dict(
            batting_team='Mumbai Indians',
            bowling_team='Chennai Super Kings',
            selected_city='Mumbai',
            target=200,
            score=150,
            balls_left=30,
            wickets=3
        ))
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Match Result Prediction', response.data)
        self.assertIn(b'Mumbai Indians: ', response.data)
        self.assertIn(b'Chennai Super Kings: ', response.data)

if __name__ == '__main__':
    unittest.main()
