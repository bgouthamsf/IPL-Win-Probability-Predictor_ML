IPL Win Probability Calculator

## Overview

The IPL Win Probability Calculator is a machine learning-based tool designed to predict the likelihood of a team's victory in an Indian Premier League (IPL) cricket match. This project leverages historical IPL data and advanced machine learning techniques to provide real-time predictions, enhancing the viewing experience for cricket enthusiasts and supporting analysts with data-driven insights.

## Features

- **Real-Time Predictions**: Input match details to get instant win probability predictions.
- **User-Friendly Interface**: A web-based application built using Flask for easy interaction.
- **Ethical Use**: Includes warnings to discourage the use of predictions for betting purposes.

## Data Sources

This project utilizes two primary datasets:

1. **matches.csv**: Contains historical IPL match data.
2. **deliveries.csv**: Provides ball-by-ball data for each IPL match.

These datasets are merged and processed to create a comprehensive dataset used for model training and evaluation.

## Model

The Random Forest Classifier (RFC) was selected as the most suitable model after comparing various machine learning techniques. The model is trained to handle the complex and non-linear nature of cricket match data, ensuring accurate predictions.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/YourUsername/IPL-Win-Probability-Calculator.git
   ```
2. Navigate to the project directory:
   ```bash
   cd IPL-Win-Probability-Calculator
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Flask application:
   ```bash
   python app.py
   ```
2. Open your web browser and go to `http://127.0.0.1:5000/`.
3. Input match details and get the win probability predictions.

## Contact:
For any inquiries or issues, please reach out to the project maintainer:  
Goutham Bommu  
