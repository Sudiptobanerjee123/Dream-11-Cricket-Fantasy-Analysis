from data_pipeline import load_data, preprocess
from modeling import train_model, predict
from optimizer import select_team

def main():
    df = load_data("data_players_sample.csv")
    df, features = preprocess(df)
    model, mse = train_model(df, features)
    print("Model MSE:", mse)
    df_pred = predict(model, df, features)
    team = select_team(df_pred, budget=100, team_size=11)
    print("Selected Team:\n", team[['player_name','team','role','price','pred_points']])

if __name__ == '__main__':
    main()
