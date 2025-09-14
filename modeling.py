import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_model(df, features, target='avg_points'):
    X = df[features]

    if target not in df.columns:
        df['avg_points'] = (
            df['avg_runs'] * 1.2 + 
            df['avg_wickets'] * 12 + 
            df['playing_prob'] * 5
        )

    y = df['avg_points']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    return model, mse

def predict(model, df, features):
    df = df.copy()
    df['pred_points'] = model.predict(df[features])
    return df
