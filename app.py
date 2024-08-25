from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Define team colors
team_colors = {
    "Mumbai Indians": "#004ba0",
    "Gujarat Titans": "#ec9a29",
    "Kolkata Knight Riders": "#512d6d",
    "Delhi Capitals": "#174ea6",
    "Sunrisers Hyderabad": "#f18c1e",
    "Royal Challengers Bangalore": "#aa0000",
    "Lucknow Super Giants": "#24a19c",
    "Rajasthan Royals": "#ea1a80",
    "Chennai Super Kings": "#f7e30b",
    "Punjab Kings": "#ed1b24"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        batting_team = request.form['batting_team']
        bowling_team = request.form['bowling_team']
        selected_city = request.form['selected_city']
        target = int(request.form['target'])
        score = int(request.form['score'])
        balls_left = int(request.form['balls_left'])
        wickets = int(request.form['wickets'])

        runs_left = target - score
        wickets_remaining = 10 - wickets
        overs_completed = (120 - balls_left) / 6 if balls_left != 120 else 0  # Calculate overs_completed from balls_left

        # Prevent division by zero for current run rate (crr)
        crr = score / overs_completed if overs_completed > 0 else 0

        # Prevent division by zero for required run rate (rrr)
        rrr = runs_left / (balls_left / 6) if balls_left > 0 else float('inf')

        input_data = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets_remaining': [wickets_remaining],
            'total_run_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        model_path = os.path.join(os.path.dirname(__file__), 'model_pipeline.pkl')
        pipe = pickle.load(open(model_path, 'rb'))
        result = pipe.predict_proba(input_data)

        win_probability = round(result[0][1] * 100)
        loss_probability = round(result[0][0] * 100)

        return render_template(
            'result.html', 
            batting_team=batting_team, 
            bowling_team=bowling_team, 
            win_probability=win_probability, 
            loss_probability=loss_probability,
            team1_color=team_colors[batting_team],
            team2_color=team_colors[bowling_team]
        )

if __name__ == '__main__':
    app.run(debug=True)
