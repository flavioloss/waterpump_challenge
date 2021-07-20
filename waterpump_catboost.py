import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import randint
from catboost import CatBoostClassifier
from hyperopt import fmin, hp, tpe
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

train_labels = pd.read_csv('train-labels.csv')
train_values = pd.read_csv('train-values.csv')
df_train = pd.merge(train_labels, train_values, on='id')
train_eda = df_train
df_test = pd.read_csv('test-values.csv')
print(len(df_train), len(df_test))

# datetime features
df_train['date'] = pd.to_datetime(df_train['date_recorded'])
df_train['day'] = df_train['date'].dt.day
df_train['month'] = df_train['date'].dt.month
df_train['year'] = df_train['date'].dt.year
df_train['weekday'] = df_train['date'].dt.weekday
df_train = df_train.drop('date_recorded', axis=1)

df_test['date'] = pd.to_datetime(df_test['date_recorded'])
df_test['day'] = df_test['date'].dt.day
df_test['month'] = df_test['date'].dt.month
df_test['year'] = df_test['date'].dt.year
df_test['weekday'] = df_test['date'].dt.weekday
df_test = df_test.drop('date_recorded', axis=1)

# drop useless
df_train = df_train.drop('scheme_name', axis=1)
df_test = df_test.drop('scheme_name', axis=1)

# geo feature engeneering
df_train = df_train.round({'latitude': 4, 'longitude': 4})
df_train['latitude_radians'] = np.radians(df_train['latitude'])
df_train['longitude_radians'] = np.radians(df_train['longitude'])
df_train['x'] = np.cos(df_train['latitude']) * np.cos(df_train['longitude'])
df_train['y'] = np.cos(df_train['latitude']) * np.sin(df_train['longitude'])
df_train['z'] = np.sin(df_train['latitude'])

df_test = df_test.round({'latitude': 4, 'longitude': 4})
df_test['latitude_radians'] = np.radians(df_test['latitude'])
df_test['longitude_radians'] = np.radians(df_test['longitude'])
df_test['x'] = np.cos(df_test['latitude']) * np.cos(df_test['longitude'])
df_test['y'] = np.cos(df_test['latitude']) * np.sin(df_test['longitude'])
df_test['z'] = np.sin(df_test['latitude'])

# outliers
df_train.loc[df_train['population'] <= 15000, 'population'] = 10000
df_test.loc[df_test['population'] <= 15000, 'population'] = 10000

# separating categorical and numeric features
y = df_train.status_group
df = pd.concat([df_train, df_test]).set_index('id')
df.drop('status_group', axis=1, inplace=True)
train_ids = df_train.id.values
test_ids = df_test.id.values

aux_df = pd.DataFrame(index=df.columns, data={'uniques': df.nunique(),
                                                    'type': df.dtypes,
                                                    'nulls': df.isnull().sum(),
                                                    'nulls%': df.isnull().sum()/len(df) * 100
                                                    })
cat_vars = aux_df[aux_df['type'] == 'object'].index
num_vars = aux_df[aux_df['type'] != 'object'].index

# fill categorical nans
df_train[cat_vars] = df_train[cat_vars].fillna('none')
# df_train[num_vars] = df_train[num_vars].fillna(0)

df_test[cat_vars] = df_test[cat_vars].fillna('none')
# df_test[num_vars] = df_test[num_vars].fillna(0)

# first catboost model (simple)
model = CatBoostClassifier(
                           iterations=1000,
                           depth=9,
                           learning_rate=0.07,
                           loss_function='MultiClass',
                           eval_metric='MultiClass',
                           nan_mode="Min",
                           cat_features=cat_vars,
                           # task_type="GPU", devices='0:1'
                           )

X = df_train.drop('status_group', axis=1)
y = df_train['status_group']

model.fit(X, y)

# feature importances
plt.figure(figsize=(8, 11))
importances = model.get_feature_importance(prettified=True)
sns.barplot(x='Importances', y='Feature Id', data=importances)
plt.show()

# first submission
preds = model.predict(df_test)
predict_df = pd.DataFrame({'id': test_ids,
                           'status_group': [i[0] for i in preds]}).set_index('id')
print(predict_df['status_group'].value_counts())
predict_df.to_csv('submission_catboost_v3.csv')


# hyoeropt optimization - Optimize between 10 and 1000 iterations and depth between 2 and 12
# search_space = {'iterations': hp.quniform('iterations', 10, 1000, 10),
#                 'depth': hp.quniform('depth', 2, 12, 1),
#                 'lr': hp.uniform('lr', 0.01, 1)
#                 }
#
# X = df_train.drop('status_group', axis=1)
# y = df_train['status_group']
#
#
# def optimal_function(search_space):
#     nfolds = 5
#     skf = StratifiedKFold(n_splits=nfolds, shuffle=True)
#     acc = []
#
#     for train_index, valid_index in skf.split(X, y):
#         X_train, X_valid = X.iloc[train_index].copy(), X.iloc[valid_index].copy()
#         y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
#
#         model = CatBoostClassifier(iterations=search_space['iterations'],
#                                    depth=search_space['depth'],
#                                    learning_rate=search_space['lr'],
#                                    loss_function='MultiClass',
#                                    od_type='Iter')
#
#         model.fit(X_train, y_train, cat_vars, logging_level='Silent')
#         predictions = model.predict(X_valid)
#         accuracy = accuracy_score(y_valid, predictions.squeeze())
#         acc.append(accuracy)
#
#     mean_acc = sum(acc) / nfolds
#     return -1 * mean_acc
#
#
# best = fmin(fn=optimal_function, space=search_space, algo=tpe.suggest, max_evals=100)


# Grid Search
# grid = {'iterations': np.arange(10, 1000, 10),
#         'max_depth': np.arange(2, 13, 1),
#         'learning_rate': np.arange(0.01, 1, 0.01)
#        }
#
# X = df_train.drop('status_group', axis=1)
# y = df_train['status_group']
#
# model = CatBoostClassifier(cat_features=cat_vars,
#                            nan_mode='Min',
#                            loss_function='MultiClass',
#                            eval_metric='MultiClass',
#                            logging_level='Silent')
#
# grid_search = GridSearchCV(estimator=model, param_grid=grid,
#                            scoring='accuracy', cv=5)
#
# grid_search.fit(X, y)
#
# print(grid_search.best_estimator_)
# print(grid_search.best_score_)
# print(grid_search.best_params_)


# Random Search
# grid = {
#         'max_depth': np.arange(2, 13, 1),
#         'learning_rate': np.linspace(0.01, 0.2, 5)
#        }
#
# X = df_train.drop('status_group', axis=1)
# y = df_train['status_group']
#
# model = CatBoostClassifier(cat_features=cat_vars,
#                            nan_mode='Min',
#                            loss_function='MultiClass',
#                            eval_metric='MultiClass',
#                            # logging_level='Silent',
#                            # task_type="GPU", devices='0:2'
#                            )
#
# grid_search = GridSearchCV(estimator=model, param_grid=grid,
#                            scoring='accuracy', cv=5, verbose=1)
#
# grid_search.fit(X, y)
#
# print(grid_search.best_score_)
# print(grid_search.best_params_)
