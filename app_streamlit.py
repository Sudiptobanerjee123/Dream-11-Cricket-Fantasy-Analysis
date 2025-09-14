import streamlit as st
import pandas as pd
from data_pipeline import load_data, preprocess
from modeling import train_model, predict
from optimizer import select_team

st.set_page_config(page_title="Fantasy Cricket Team Builder", layout="wide")
st.title("Fantasy Cricket Team Builder — Demo")

st.sidebar.header("Controls")
budget = st.sidebar.slider("Budget (credits)", 60, 120, 100)
team_size = st.sidebar.slider("Team size", 7, 11, 11)
max_from_team = st.sidebar.slider("Max players from same real team", 3, 7, 7)

st.sidebar.markdown("### Role limits (min, max)")
default_roles = {'Batsman': (3,5), 'Bowler': (3,5), 'Allrounder': (1,4), 'Wicketkeeper': (1,1)}
role_limits = {}
for r, (a,b) in default_roles.items():
    mn = st.sidebar.number_input(f"{r} min", min_value=0, max_value=11, value=a, key=f"{r}_min")
    mx = st.sidebar.number_input(f"{r} max", min_value=0, max_value=11, value=b, key=f"{r}_max")
    role_limits[r] = (mn, mx)

df = load_data("data_players_sample.csv")
df, features = preprocess(df)

st.header("Player dataset (sample)")
st.dataframe(df)

st.header("Train model (demo)")
if st.button("Train & Predict"):
    model, mse = train_model(df, features, target='avg_points')
    st.write(f"Trained RandomForest model — MSE on demo split: {mse:.2f}")
    df_pred = predict(model, df, features)

    st.subheader("Predictions")
    st.dataframe(df_pred[['player_name', 'team', 'role', 'price', 'pred_points']].sort_values('pred_points', ascending=False))

    st.subheader("Optimized Team")
    team = select_team(df_pred, budget=budget, team_size=team_size, max_from_team=max_from_team, role_limits=role_limits)
    st.dataframe(team[['player_name', 'team', 'role', 'price', 'pred_points']])

    st.write("Total price:", team['price'].sum(), "Total predicted points:", team['pred_points'].sum())
else:
    st.info("Click 'Train & Predict' to run the demo pipeline.")
