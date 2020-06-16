# coding: utf-8

# Predicting Results:

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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

##################################################################################
############################### Data Wrangling ###################################
##################################################################################

# Plot styling and sizing
plt.style.use('seaborn-darkgrid')
pd.set_option('display.max_columns', None)
sns.set(style='darkgrid', context='notebook', rc={'figure.figsize': (12, 6)})

# create a connection object for sql db
cnx = sqlite3.connect('database.sqlite')
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
                                WHERE country_name in ('Spain', 'Germany', 'France', 'Italy', 'England')

                                ORDER by date
                                LIMIT 100000;""", cnx)

##################################################################################
############################# Data Understanding #################################
##################################################################################

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
missing = match.isnull().sum()

percent_of_missing = (match.isnull().sum() / match.isnull().count()) * 100

# is showing percent of the nulls present in every columns
print(percent_of_missing)

# is showing duplicates in every columns
print(sum(match.duplicated()))

# Print summary statistics of home_team_goal.
match[['home_team_goal', 'away_team_goal']].describe()

##################################################################################
####################### Data Cleaning + Preparation ##############################
##################################################################################
selected_col = ["home_team", "home_id", "away_team", "away_id", "season", "home_team_goal", "away_team_goal",
                "league_name", 'date', 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA']

match = match[selected_col]

match.dropna(subset=selected_col, inplace=True)

match = pd.DataFrame(
    {"HomeTeam": match.home_team, "HomeID": match.home_id, "AwayTeam": match.away_team, "AwayID": match.away_id,
     "HTG": match.home_team_goal, "ATG": match.away_team_goal, "B365H": match.B365H, "B365D": match.B365D,
     "B365A": match.B365A, "BWH": match.BWH, "BWD": match.BWD, "BWA": match.BWA})

print(match.head())
#
# selected_col = ['home_team_api_id', 'away_team_api_id', 'home_team_goal', 'away_team_goal',
#                 'date', 'country_id', 'league_id', 'season', 'stage', 'B365H', 'B365D', 'B365A',
#                 'BWH', 'BWD', 'BWA']
#
# match.dropna(subset=selected_col, inplace=True)
#
# match = match[selected_col]
#
#
# def match_result(home_goal, away_goal):
#     if home_goal > away_goal:
#         return 'Home'
#     elif home_goal < away_goal:
#         return 'Away'
#     else:
#         return 'Tie'
#
#
# def bets_result(bh, bd, ba):
#     if bh < bd <= ba or bh < ba <= bd:
#         return 'Home'
#     elif bh >= bd > ba or bd >= bh > ba:
#         return 'Away'
#     elif bd < bh <= ba or bd < ba <= bh:
#         return 'Tie'
#
# # Home team Goal Average
# match['HGA'] = match['home_team_goal'].groupby(match['home_team_api_id']).transform('mean')
# # Away team Goal Average
# match['AGA'] = match['away_team_goal'].groupby(match['away_team_api_id']).transform('mean')
# # Full Time Result
# match['FTR'] = match.apply(lambda i: match_result(i['home_team_goal'], i['away_team_goal']), axis=1)
# # B365 Result Bet
# match['b365_res'] = match.apply(lambda i: bets_result(i['B365H'], i['B365D'], i['B365A']), axis=1)
# # B-Win Result Bet
# match['bw_res'] = match.apply(lambda i: bets_result(i['BWH'], i['BWD'], i['BWA']), axis=1)
# # goal_sum for each match
# match['GOAL_SUM'] = match['home_team_goal'] + match['away_team_goal']
#
# print(match.isnull().sum())
#
# print(match.head())


##################################################################################
################################ VISUALIZATION ###################################
##################################################################################

# # Plot the distribution of home_team_goal and its evolution over time.
# f, axes = plt.subplots(2, 1)
# plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
# sns.distplot(match['home_team_goal'], kde=False, ax=axes[0]).set_title('Distribution of home_team_goal')
# sns.distplot(match['away_team_goal'], kde=False, ax=axes[1]).set_title('Distribution of away_team_goal')
#
# plt.show()
# match.groupby(by='season')[['home_team_goal', 'away_team_goal']] \
#     .mean().plot(ax=axes[1], title='Difference between goals scored at home and goals scored away')
# plt.show()
#
# results = match.groupby('FTR').count()
# print(results)
# results.home_team_goal.plot(kind='bar')
# plt.show()
#
# bet_fit = match.query('FTR==b365_result').count().home_team_goal / match.shape[0]
# bet_fit_op = 1 - bet_fit
# plt.figure(figsize=(8, 8))
# plt.pie([bet_fit, bet_fit_op], shadow=True, autopct='%1.1f%%', labels=["B365 FITS", "B365 MISS"])
# plt.title('BET POPULAR SITS CORRECTION ')
# plt.show()
#
# bet_fit = match.query('FTR==bw_result').count().home_team_goal / match.shape[0]
# bet_fit_op = 1 - bet_fit
# plt.figure(figsize=(8, 8))
# plt.pie([bet_fit, bet_fit_op], shadow=True, autopct='%1.1f%%', labels=["BW FITS", "BW MISS"])
# plt.title('BET-WIN POPULAR SITS CORRECTION ')
# plt.show()
#
# total_away_goals = match.away_team_goal.sum()
# total_home_goals = match.home_team_goal.sum()
# plt.figure(figsize=(8, 8))
# plt.bar(x=[1, 2], tick_label=['Away Team', 'Home Team'], height=[total_away_goals, total_home_goals])
# plt.xlabel('Goals scored by')
# plt.ylabel('Number of goals scored')
# plt.title('Goals scored by team')
# plt.show()
#
# away_percent = results.query('FTR=="Away"').home_team_goal / results.shape[0]
# home_percent = results.query('FTR=="Home"').home_team_goal / results.shape[0]
# tie_percent = results.query('FTR=="Tie"').home_team_goal / results.shape[0]
#
# plt.figure(figsize=(8, 8))
# pie_labels = ['Tied', 'Away Team Won', 'Home Team Won']
# plt.pie([tie_percent, away_percent, home_percent], labels=pie_labels, autopct='%1.1f%%', shadow=True)
# plt.title('Distribution of match results by winning team')
# plt.show()
#
# players_height = pd.read_sql("""SELECT CASE
#                                         WHEN ROUND(height)<165 then 165
#                                         WHEN ROUND(height)>195 then 195
#                                         ELSE ROUND(height)
#                                         END AS calc_height,
#                                         COUNT(height) AS distribution,
#                                         (avg(PA_Grouped.avg_overall_rating)) AS avg_overall_rating,
#                                         (avg(PA_Grouped.avg_potential)) AS avg_potential,
#                                         AVG(weight) AS avg_weight
#                             FROM PLAYER
#                             LEFT JOIN (SELECT Player_Attributes.player_api_id,
#                                         avg(Player_Attributes.overall_rating) AS avg_overall_rating,
#                                         avg(Player_Attributes.potential) AS avg_potential
#                                         FROM Player_Attributes
#                                         GROUP BY Player_Attributes.player_api_id)
#                                         AS PA_Grouped ON PLAYER.player_api_id = PA_Grouped.player_api_id
#                             GROUP BY calc_height
#                             ORDER BY calc_height
#                                 ;""", cnx)
#
# players_height.plot(x='calc_height', y='avg_overall_rating', figsize=(12, 5), title='Potential vs Height')
# plt.show()
#
# # Plot the distribution of goal_sum and its evolution over the considered timeframe.
# plt.figure(figsize=(12, 6))
# figure, axes = plt.subplots(2, 1)
# plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
# sns.distplot(match['goal_sum'], kde=False, ax=axes[0]).set_title('Distribution of goal scored')
# sns.lineplot(x='season', y='goal_sum', data=match, err_style=None, ax=axes[1]).set_title(
#     'How does the variable goal_sum evolve over time?')
# plt.show()
