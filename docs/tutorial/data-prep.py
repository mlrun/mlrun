import pandas as pd
import mlrun

@mlrun.function(outputs=[
    "label_column",
    ("dataset", mlrun.ArtifactType.DATASET, {'format': 'csv', 'index': False}),
])
def worldcup_data_generator(context, data_path="./WorldCupMatches.csv"):
    """a function which generates and preprocess the world cup dataset"""
    data = pd.read_csv(data_path, encoding='UTF-8')
    data.dropna(inplace=True)
    data = preprocess(data)
    teams = set(list(data['Home Team Name'].unique()) + (list(data['Away Team Name'].unique())))
    stages = list(data['Stage'].unique())
    data['Home Team Name'] = pd.Categorical(data['Home Team Name'], categories=list(teams))
    data['Away Team Name'] = pd.Categorical(data['Away Team Name'], categories=list(teams))
    data['Stage'] = pd.Categorical(data['Stage'], categories=list(stages))
    home_team = pd.get_dummies(data['Away Team Name'], prefix='Away Team Name')
    away_team = pd.get_dummies(data['Home Team Name'], prefix='Home Team Name')
    stage = pd.get_dummies(data['Stage'], prefix='Stage')
    
    data.drop(columns=['Home Team Goals', 'Away Team Goals', 'Away Team Name', 'Home Team Name', 'Stage', 'Attendance'], inplace=True)
    data = pd.concat([data, stage, home_team, away_team], axis=1)
    data.dropna(inplace=True)
    context.logger.info("saving world cup matches dataframe")
    
    return 'Win', data
    

def preprocess(data):
    teams = set(list(data['Home Team Name'].unique()) + (list(data['Away Team Name'].unique())))
    stages = list(data['Stage'].unique())
    victories, finals, goals, victories_tur, victories_tur, goals_tur = create_zero_dict(teams, 6)
    year = 1930
    
    for i, row in data.iterrows():
        if row['Stage'] == 'Final' or row['Stage'] == 'finals':
            finals[row['Home Team Name']] += 1
            finals[row['Away Team Name']] += 1
        if year != row['Year']:
            victories_tur, goals_tur = create_zero_dict(teams, 2)
        data = insert_to_data(data, i, row, ['finals', 'victory current tournament', 'victory', 'goals', 'goals current tournament'], 
                              [finals, victories_tur, victories, goals, goals_tur])

        goals, goals_tur = update_goals(data, row, [goals, goals_tur])
        if row['Win'] == 1:
            victories[row['Home Team Name']] += 1
            victories_tur[row['Home Team Name']] += 1
        elif row['Win'] == 2:
            victories[row['Away Team Name']] += 1
            victories_tur[row['Away Team Name']] += 1
        
    data = data[data.Year > 1950]
    data = data[data.Win > 0]
    return data

def create_zero_dict(teams, num_of_dict):
    
    return [dict(zip(teams, [0]*len(teams))) for i in range(num_of_dict)]

def insert_to_data(data, index, row, fields, dictionaries):
    for field, dictionary in zip(fields, dictionaries):
        data.at[index, f'Home {field}'] = dictionary[row['Home Team Name']]
        data.at[index, f'Away {field}'] = dictionary[row['Away Team Name']]
    return data
    
def update_goals(data, row, goals_dictionaries):
    for dictionary in goals_dictionaries:
        dictionary[row['Home Team Name']] += row['Home Team Goals']
        dictionary[row['Away Team Name']] += row['Away Team Goals']
    return goals_dictionaries
