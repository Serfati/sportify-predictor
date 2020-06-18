# coding: utf-8

# About the Dataset:

# The ultimate Soccer database for data analysis and machine learning The dataset comes in the form of an SQL
# database and contains statistics of about 25,000 football matches. from the top football league of 11 European
# Countries. It covers seasons from 2008 to 2016 and contains match statistics (i.e: scores, corners, fouls etc...)
# as well as the team formations, with player names and a pair of coordinates to
# Players and Teams' attributes* sourced from EA Sports' FIFA video game series, including the weekly updates Team
# line up with squad formation (X, Y coordinates) Betting odds from up to 10 providers Detailed match events (goal
# types, possession, corner, cross, fouls, cards etc...) for +10,000 matches The dataset also has a set of about 35
# statistics for each player, derived from EA Sports' FIFA video games. It is not just the stats that come with a new

## STEPS: Data Wrangling, Business Understanding,
# ## Data Understanding, Data Cleaning, Data Preparation
# ## Create new variables, Exploratory Data Analysis, Conclusions
# ## Modelling, Evaluation, Deployment

# Importing libraries to the environment
import sqlite3

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from keras.layers import Dense
from keras.models import Sequential

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

##################################################################################
############################### Data Wrangling ###################################
##################################################################################

match = pd.DataFrame()
cnx = sqlite3.connect('database.sqlite')


def wrangling():
    print("Data Wrangling")

    global match, cnx
    # Plot styling and sizing
    plt.style.use('seaborn-darkgrid')
    pd.set_option('display.max_columns', None)
    sns.set(style='darkgrid', context='notebook', rc={'figure.figsize': (12, 6)})

    # create a connection object for sql db

    # database.sqlite IS SQLITE file downloaded from KAGGLE.COM used for importing the dataset

    # match = pd.read_sql_query("SELECT * FROM Match", cnx)

    match = pd.read_sql("""SELECT Match.id,
                                    Country.name AS country_name,
                                    League.name AS league_name,
                                    season,
                                    stage,
                                    date,
                                    HT.team_long_name AS home_team,
                                    HT.team_api_id AS home_id,
                                    AT.team_long_name AS away_team,
                                    AT.team_api_id AS away_id,
                                    B365H,  BWH,
                                    B365D,  BWD,
                                    B365A,  BWA,
                                    home_team_goal,
                                    away_team_goal
                                    FROM Match
                                    JOIN Country on Country.id = Match.country_id
                                    JOIN League on League.id = Match.league_id
                                    LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                    LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
                                    ORDER by date
                                    ;""", cnx)


# WHERE country_name in ('Spain', 'Germany', 'France', 'Italy', 'England')
##################################################################################
############################# Data Understanding #################################
##################################################################################
def understanding():
    print("Data Understanding")

    print(match.head())

    # ## Exploratory data analysis
    print(match.info())

    # to check data types of the all the columns
    print(match.dtypes)

    # for numeric variable to get stats
    print(match.describe())

    # nrows and ncols
    print(match.shape)

    # missing values in the data frame
    print(match.isnull().sum())

    percent_of_missing = (match.isnull().sum() / match.isnull().count()) * 100

    # is showing percent of the nulls present in every columns
    print(percent_of_missing)

    # is showing duplicates in every columns
    print(sum(match.duplicated()))

    # Print summary statistics of home_team_goal.
    match[['home_team_goal', 'away_team_goal']].describe()


##################################################################################
# *********************** Data Cleaning + Preparation ************************   #
#### Create new variables ########################################################
def cleaning():
    print("Data Cleaning")
    global match
    selected_col = ["home_team", "home_id", "away_team", "away_id", "season", "home_team_goal", "away_team_goal",
                    "league_name", 'date', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA']

    match = match[selected_col]

    match.dropna(subset=selected_col, inplace=True)

    match = pd.DataFrame(
        {"League": match.league_name, "season": match.season, "HomeTeam": match.home_team, "HomeID": match.home_id,
         "AwayTeam": match.away_team, "AwayID": match.away_id,
         "HTG": match.home_team_goal, "ATG": match.away_team_goal, "B365H": match.B365H, "B365D": match.B365D,
         "B365A": match.B365A, "BWH": match.BWH, "BWD": match.BWD, "BWA": match.BWA})

    # GOAL_DIFF for each match
    match['GOAL_DIFF'] = match['HTG'] - match['ATG']


def match_result(home_goal, away_goal):
    if home_goal > away_goal:
        return 1
    elif home_goal < away_goal:
        return 2
    else:
        return 0


def bets_result(bhome, bdarw, baway):
    if bhome <= bdarw <= baway or bhome <= baway <= bdarw:
        return 1
    elif bhome >= bdarw >= baway or bdarw >= bhome >= baway:
        return 2
    elif bdarw <= bhome <= baway or bdarw <= baway <= bhome:
        return 0


short2id = pd.read_sql("SELECT team_api_id AS ID, team_Short_name AS short FROM Team", cnx)


def preparation():
    print("Data Preparation")
    # Home team Goal Average
    match['HGA'] = match['HTG'].groupby(match['HomeTeam']).transform('mean')
    # Away team Goal Average
    match['AGA'] = match['ATG'].groupby(match['AwayTeam']).transform('mean')
    # B365 Result Bet
    match['B365'] = match.apply(lambda i: bets_result(i['B365H'], i['B365D'], i['B365A']), axis=1)
    # B-Win Result Bet
    match['BW'] = match.apply(lambda i: bets_result(i['BWH'], i['BWD'], i['BWA']), axis=1)
    # Target Variable -  Full Time Result

    match_temp = pd.read_sql("SELECT * FROM Match ORDER by date", cnx)
    match_temp['FTR'] = match_temp.apply(lambda i: match_result(i['home_team_goal'], i['away_team_goal']), axis=1)
    res = getLast5MatchesHome(match_temp)
    match["HomeWinLastFive"] = match.apply(lambda i: setWinningsHome(i["HomeID"], res), axis=1)
    res = getLast5MatchesAway(match_temp)
    match["AwayWinLastFive"] = match.apply(lambda i: setWinningsAway(i["AwayID"], res), axis=1)
    res = getLast5MatchesConf(match_temp)
    match["HomeWinLastFiveConfrontation"] = match.apply(lambda i: get_conf_winnings_home(i["HomeID"], i["AwayID"], res),
                                                        axis=1)
    match["AwayWinLastFiveConfrontation"] = match.apply(lambda i: get_conf_winnings_away(i["HomeID"], i["AwayID"], res),
                                                        axis=1)
    match['FTR'] = match.apply(lambda i: match_result(i['HTG'], i['ATG']), axis=1)
    print("done - transform to database")
    match.to_sql(name='Sportify', con=cnx)


def getLast5MatchesHome(matches):
    matches = matches.sort_values(by=['home_team_api_id', 'date'], ascending=False)
    matches = matches.groupby('home_team_api_id').head(5)
    sub_matches = matches[['home_team_api_id', 'FTR']]
    ids_and_res = matches[['home_team_api_id']].groupby('home_team_api_id').first().reset_index()
    winners = sub_matches[sub_matches['FTR'] == 1].groupby('home_team_api_id').count().reset_index()
    ids_and_res['winnings'] = 0
    ids_and_res['winnings'] = ids_and_res.apply(lambda i: setResHome(i["home_team_api_id"], winners), axis=1)
    return ids_and_res


def getLast5MatchesAway(matches):
    matches = matches.sort_values(by=['away_team_api_id', 'date'], ascending=False)
    matches = matches.groupby('away_team_api_id').head(5)
    sub_matches = matches[['away_team_api_id', 'FTR']]
    ids_and_res = matches[['away_team_api_id']].groupby('away_team_api_id').first().reset_index()
    winners = sub_matches[sub_matches['FTR'] == 1].groupby('away_team_api_id').count().reset_index()
    ids_and_res['winnings'] = 0
    ids_and_res['winnings'] = ids_and_res.apply(lambda i: setResAway(i["away_team_api_id"], winners), axis=1)
    return ids_and_res


def setResHome(home_team_id, winners_df):
    if home_team_id in winners_df["home_team_api_id"].values:
        return winners_df.loc[winners_df['home_team_api_id'] == home_team_id, 'FTR'].iloc[0]
    return 0


def setWinningsHome(home_team_id, winners_df):
    if home_team_id in winners_df["home_team_api_id"].values:
        return winners_df.loc[winners_df['home_team_api_id'] == home_team_id, 'winnings'].iloc[0]
    return 0


def setResAway(away_team_id_api, winners_df):
    if away_team_id_api in winners_df["away_team_api_id"].values:
        return winners_df.loc[winners_df['away_team_api_id'] == away_team_id_api, 'FTR'].iloc[0]
    return 0


def setWinningsAway(away_team_id, winners_df):
    if away_team_id in winners_df["away_team_api_id"].values:
        return winners_df.loc[winners_df['away_team_api_id'] == away_team_id, 'winnings'].iloc[0]
    return 0


def getLast5MatchesConf(matches):
    matches["homeAndAwayID"] = ""
    combineIDs(matches)
    matches.sort_values(by=['homeAndAwayID', 'date'], ascending=False, inplace=True)
    matches = matches.groupby('homeAndAwayID').head(5)
    sub_matches = matches[['home_team_api_id', 'away_team_api_id', 'homeAndAwayID', 'FTR']]
    return calcWinnings(sub_matches)


def calcWinnings(matches):
    matches['HomeWinLastFiveConfrontation'] = 0
    matches['AwayWinLastFiveConfrontation'] = 0
    for index, row in matches.iterrows():
        if row.FTR == 1:
            # return winners_df.loc[winners_df['home_team_api_id'] == home_team_id, 'FTR'].iloc[0]
            points = matches.loc[index, 'HomeWinLastFiveConfrontation']
            matches.loc[((matches["homeAndAwayID"] == row.homeAndAwayID) &
                         (matches["home_team_api_id"] == row.home_team_api_id)), 'HomeWinLastFiveConfrontation'] = \
                int(int(points) + 1)
            matches.loc[((matches["homeAndAwayID"] == row.homeAndAwayID) &
                         (matches["away_team_api_id"] == row.home_team_api_id)), 'AwayWinLastFiveConfrontation'] = \
                int(int(points) + 1)
        elif row.FTR == 2:
            points = matches.loc[index, 'AwayWinLastFiveConfrontation']
            matches.loc[((matches["homeAndAwayID"] == row.homeAndAwayID) &
                         (matches["away_team_api_id"] == row.away_team_api_id)), 'AwayWinLastFiveConfrontation'] = \
                int(int(points) + 1)
            matches.loc[((matches["homeAndAwayID"] == row.homeAndAwayID) &
                         (matches["home_team_api_id"] == row.away_team_api_id)), 'HomeWinLastFiveConfrontation'] = \
                int(int(points) + 1)

    matches = matches.groupby('homeAndAwayID').first().reset_index()
    return matches


def combineIDs(matches):
    for index, row in matches.iterrows():
        if (str(row.away_team_api_id) + str(row.home_team_api_id)) in matches["homeAndAwayID"].astype(str).values:
            matches.loc[index, 'homeAndAwayID'] = str(row.away_team_api_id) + str(row.home_team_api_id)
        else:
            matches.loc[index, 'homeAndAwayID'] = str(row.home_team_api_id) + str(row.away_team_api_id)


def combineID(id1, id2, matches, i):
    if (str(id2) + str(id1)) in matches["homeAndAwayID"].astype(str).values:
        matches.loc[i, 'homeAndAwayID'] = str(id2) + str(id1)
    matches.loc[i, 'homeAndAwayID'] = str(id1) + str(id2)


def get_conf_winnings_home(HomeID, AwayID, matches):
    if (str(HomeID) + str(AwayID)) in matches["homeAndAwayID"].astype(str).values:
        x = matches.loc[matches["homeAndAwayID"] == (str(HomeID) + str(AwayID))]
        return x["HomeWinLastFiveConfrontation"].iloc[0]
    x = matches.loc[matches["homeAndAwayID"] == (str(AwayID) + str(HomeID))]
    return x["HomeWinLastFiveConfrontation"].iloc[0]


def get_conf_winnings_away(HomeID, AwayID, matches):
    if (str(HomeID) + str(AwayID)) in matches["homeAndAwayID"].astype(str).values:
        x = matches.loc[matches["homeAndAwayID"] == (str(HomeID) + str(AwayID))]
        return x["AwayWinLastFiveConfrontation"].iloc[0]
    x = matches.loc[matches["homeAndAwayID"] == (str(AwayID) + str(HomeID))]
    return x["AwayWinLastFiveConfrontation"].iloc[0]


##################################################################################
# ******************************    Importing   ******************************   #
##################################################################################
def importing():
    print('Importing Prepared Data')
    global match
    match = pd.read_sql('SELECT * FROM Sportify', cnx)


##################################################################################
# ****************************** VISUALIZATION  ******************************   #
##################################################################################
def visualization():
    print("Data Visualization")

    global cnx
    # Plot the distribution of home_team_goal and its evolution over time.
    f, axes = plt.subplots(2, 1)
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
    sns.distplot(match['HTG'], kde=False, ax=axes[0]).set_title('Distribution of Home Team Goal')
    sns.distplot(match['ATG'], kde=False, ax=axes[1]).set_title('Distribution of Away Team Goal')

    plt.show()
    match.groupby(by='season')[['HTG', 'ATG']] \
        .mean().plot(ax=axes[1], title='Difference between goals scored at home and goals scored away')
    plt.show()

    results = match.groupby('FTR').count()
    results.HTG.plot(kind='bar')
    plt.title('Distribution of Full Time Results')
    plt.show()

    bet_fit = match.query('FTR==B365').count().HTG / match.shape[0]
    bet_fit_op = 1 - bet_fit
    plt.figure(figsize=(8, 8))
    plt.pie([bet_fit, bet_fit_op], shadow=True, autopct='%1.1f%%', labels=["Fit", "Miss"])
    plt.title('Distribution of BET365')
    plt.show()

    bet_fit = match.query('FTR==BW').count().HTG / match.shape[0]
    bet_fit_op = 1 - bet_fit
    plt.figure(figsize=(8, 8))
    plt.pie([bet_fit, bet_fit_op], shadow=True, autopct='%1.1f%%', labels=["Fit", "Miss"])
    plt.title('Distribution of BET-WIN')
    plt.show()

    total_away_goals = match.ATG.sum()
    total_home_goals = match.HTG.sum()
    plt.figure(figsize=(8, 8))
    plt.bar(x=[1, 2], tick_label=['Away Team', 'Home Team'], height=[total_away_goals, total_home_goals])
    plt.xlabel('Goals scored by')
    plt.ylabel('Number of goals scored')
    plt.title('Goals scored by team')
    plt.show()

    home_percent = results.query('FTR=="1"').HTG / results.shape[0]
    tie_percent = results.query('FTR=="0"').HTG / results.shape[0]
    away_percent = results.query('FTR=="2"').HTG / results.shape[0]

    plt.figure(figsize=(8, 8))
    pie_labels = ['Tied', 'Away Team Won', 'Home Team Won']
    plt.pie([0.253, away_percent, home_percent], labels=pie_labels, autopct='%1.1f%%', shadow=True)
    plt.title('Distribution of match results by winning team')
    plt.show()

    players_height = pd.read_sql("""SELECT CASE
                                            WHEN ROUND(height)<165 then 165
                                            WHEN ROUND(height)>195 then 195
                                            ELSE ROUND(height)
                                            END AS calc_height,
                                            COUNT(height) AS distribution,
                                            (avg(PA_Grouped.avg_overall_rating)) AS avg_overall_rating,
                                            (avg(PA_Grouped.avg_potential)) AS avg_potential,
                                            AVG(weight) AS avg_weight
                                FROM PLAYER
                                LEFT JOIN (SELECT Player_Attributes.player_api_id,
                                            avg(Player_Attributes.overall_rating) AS avg_overall_rating,
                                            avg(Player_Attributes.potential) AS avg_potential
                                            FROM Player_Attributes
                                            GROUP BY Player_Attributes.player_api_id)
                                            AS PA_Grouped ON PLAYER.player_api_id = PA_Grouped.player_api_id
                                GROUP BY calc_height
                                ORDER BY calc_height
                                    ;""", cnx)

    players_height.plot(x='calc_height', y='avg_overall_rating', figsize=(12, 5), title='Potential vs Height')
    plt.show()

    # Plot the distribution of GOAL_DIFF and its evolution over the considered timeframe.
    plt.figure(figsize=(12, 6))
    figure, axes = plt.subplots(2, 1)
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
    sns.distplot(match['GOAL_DIFF'], kde=False, ax=axes[0]).set_title('Distribution of goal scored')
    sns.lineplot(x='season', y='GOAL_DIFF', data=match, err_style=None, ax=axes[1]).set_title(
        'How does the variable GOAL_DIFF evolve over time?')
    plt.show()


def draw_confusion(cm):
    plt.matshow(cm)
    plt.title('Confusion matrix for validation data\n'
              + '                               ')
    plt.colorbar()
    plt.show()


##################################################################################
# ****************************** Data Splitting  ******************************  #
##################################################################################
def build_deep_neural(arr):
    model = Sequential()
    for i in range(len(arr)):
        if i != 0 and i != len(arr) - 1:
            if i == 1:
                model.add(Dense(arr[i], input_dim=arr[0], kernel_initializer='normal', activation='relu'))
            else:
                model.add(Dense(arr[i], activation='relu'))
    model.add(Dense(arr[-1], kernel_initializer='normal', activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
    return model


def percentage_split(model, data):
    # LogisticRegression - 0.53%
    # LinearRegression - 0.47%
    # KNN - 0.64%
    # NB - 0.459%
    # DecisionTreeClassifier - 00
    # SVM -  0.53%
    data = data.drop('GOAL_DIFF', axis=1)
    print('Modeling')
    train, test = train_test_split(data, test_size=0.35, random_state=42)
    train = np.array(train)
    test = np.array(test)

    y = train[:, -1]
    x = train[:, 9:22]
    y1 = test[:, -1]
    x1 = test[:, 9:22]

    model.fit(x, y.astype('int'))
    y_predict = model.predict(x1)
    predictions = [round(value) for value in y_predict]
    evaluation(y_true=y1.astype('int'), y_pred=predictions)


def cross_validation(model, data, predictors, outcome):
    # RandomForestClassifier - 49.4%
    # SVM - 53.608%%
    seed = 42
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    accuracy = []
    print('Cross-Validation')
    for train, test in kf.split(data):
        print('.')
        train_predictors = (data[predictors].iloc[train, :])
        train_target = data[outcome].iloc[train]
        model.fit(train_predictors, train_target)
        accuracy.append(model.score(data[predictors].iloc[test, :], data[outcome].iloc[test]))
    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(accuracy)))


def train_cross_model():
    predictor_var = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD',
                     'BWA', 'HGA', 'AGA', 'B365', 'BW', 'HomeWinLastFive', 'AwayWinLastFive',
                     'HomeWinLastFiveConfrontation', 'AwayWinLastFiveConfrontation']
    outcome_var = 'FTR'
    cross_validation(cross, match, predictor_var, outcome_var)


##################################################################################
# ********************************* Modeling  *********************************  #
##################################################################################
def train_split_model():
    percentage_split(split, match)


def train_dnn_model():
    # DNN - 46.07% // 1k epochs => 46.34%
    # divide dataset into x(input) and y(output)
    predictor_var = ['B365H', 'B365D', 'B365A', 'BWH', 'BWD',
                     'BWA', 'HGA', 'AGA', 'B365', 'BW', 'HomeWinLastFive', 'AwayWinLastFive',
                     'HomeWinLastFiveConfrontation', 'AwayWinLastFiveConfrontation']
    X = match[predictor_var]
    y = match["FTR"]

    # - Splitting
    # divide dataset into training set, cross validation set, and test set
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)
    trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2, random_state=42)
    dnn.fit(np.array(trainX), np.array(trainY), epochs=250)
    # # - Evaluation
    scores = dnn.evaluate(np.array(valX), np.array(valY))
    print('scores: %.3f%%' % scores[1])
    predY = dnn.predict(np.array(testX))
    predY = np.round(predY).astype(int).reshape(1, -1)[0]
    from sklearn.metrics import confusion_matrix
    cm = pd.crosstab(predY, testY)
    m = confusion_matrix(predY, testY)
    draw_confusion(m)
    print("Confusion matrix")
    print(cm)


##################################################################################
# ********************************* Evaluation  *********************************  #
##################################################################################
def evaluation(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print("Model accuracy: %.4f%% " % accuracy)
    # mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    # Feature Importance rank
    # fi = enumerate(rfc.feature_importances_)
    # cols = train.columns
    # fi = [(value, cols[i]) for (i, value) in fi if value > 0.005]
    # fi.sort(key=lambda tup: tup[0], reverse=True)
    # print(fi)


def init_models():
    global dnn, cross, split
    cross = SVC()
    split = GaussianNB()
    dnn = build_deep_neural([14, 42, 70, 28, 1])


def run_main_loop():
    wrangling()
    # understanding()
    # cleaning()
    # preparation()

    importing()
    # visualization()
    init_models()
    train_cross_model()
    # train_split_model()
    # train_dnn_model()


##################################################################################
# -------------------------------- Main Loop ----------------------------------- #
##################################################################################
if __name__ == '__main__':
    cross = SVC()
    split = GaussianNB()
    dnn = build_deep_neural([14, 42, 70, 28, 1])
    run_main_loop()
