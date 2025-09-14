import pulp
import pandas as pd

def select_team(df, budget=100, team_size=11, max_from_team=7, role_limits=None):
    if role_limits is None:
        role_limits = {
            'Batsman': (3,5),
            'Bowler': (3,5),
            'Allrounder': (1,4),
            'Wicketkeeper': (1,1)
        }

    players = df['player_id'].tolist()
    prob = pulp.LpProblem("dream11_select", pulp.LpMaximize)
    x = pulp.LpVariable.dicts('select', players, lowBound=0, upBound=1, cat=pulp.LpBinary)

    # Objective: maximize total predicted points
    pred = dict(zip(df['player_id'], df['pred_points']))
    prob += pulp.lpSum([pred[i]*x[i] for i in players])

    # Team size constraint
    prob += pulp.lpSum([x[i] for i in players]) == team_size

    # Budget constraint
    price = dict(zip(df['player_id'], df['price']))
    prob += pulp.lpSum([price[i]*x[i] for i in players]) <= budget

    # Max players from same team
    for t in df['team'].unique():
        members = df[df['team']==t]['player_id'].tolist()
        prob += pulp.lpSum([x[i] for i in members]) <= max_from_team

    # Role constraints
    for role, (rmin, rmax) in role_limits.items():
        members = df[df['role']==role]['player_id'].tolist()
        if members:
            prob += pulp.lpSum([x[i] for i in members]) >= rmin
            prob += pulp.lpSum([x[i] for i in members]) <= rmax

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    selected = [i for i in players if pulp.value(x[i])==1]
    return df[df['player_id'].isin(selected)].sort_values(by='pred_points', ascending=False)
