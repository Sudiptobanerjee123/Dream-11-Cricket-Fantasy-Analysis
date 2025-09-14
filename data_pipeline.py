import pandas as pd

def load_data(path="data_players_sample.csv"):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    df = df.copy()
    # Fill missing values
    df['avg_runs'] = df['avg_runs'].fillna(0)
    df['avg_wickets'] = df['avg_wickets'].fillna(0)

    # Feature engineering
    df['bat_score'] = df['avg_runs'] * 1.0
    df['bowl_score'] = df['avg_wickets'] * 8.0
    df['experience_score'] = (df['playing_prob'] * 10)

    # Features used for modeling
    features = ['bat_score', 'bowl_score', 'experience_score', 'price']
    return df, features
