import xgboost
import shap
import numpy as np
import pandas as pd
import shap
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
from pymrmre import mrmr
from pca import pca
from sklearn.pipeline import Pipeline
from scipy import stats
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_boston


warnings.filterwarnings("ignore")
%run featimp

# Load data and make dataframe
boston = load_boston()
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data['PRICE'] = boston.target
data.head()

# Createing target variable 'y' and split data into test and training set
y = data['PRICE']
X = data.drop('PRICE', axis=1)
num_columns = X.columns
X_, X_test, y_, y_test = train_test_split(X, y, test_size=0.20)

# To deal with numerical variables - utilize Standard Scaler
num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler())])

# Put together in column transformer and take care of both data types at the same time
preprocessing = ColumnTransformer([('numerical',  num_pipe, num_columns)])

algorithms = algos = [RandomForestRegressor(),
                      ExtraTreesRegressor(),
                      XGBRegressor(objective='reg:squarederror')]

results = dict()

for algo in algorithms:
    pipe = Pipeline([('preprocessing', preprocessing),
                     ('clf', algo)])

    pipe.fit(X_, y_)
    y_pred = pipe.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    print(f"{algo.__class__.__name__:<17} - RMSE: {rmse:,.2f}")

    X_lessfeat = X.drop(columns=['LSTAT'], axis=1)
    
    
# Split training set into train and validation set
X_2, X_test2, y_2, y_test2 = train_test_split(X_lessfeat, y, test_size=0.20)

num_columns = X_lessfeat.columns

# To deal with numerical variables - utilize Standard Scaler
num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler())])

# Put together in column transformer and take care of both data types at the same time
preprocessing = ColumnTransformer([('numerical',  num_pipe, num_columns)])


algorithms = algos = [RandomForestRegressor(),
                      ExtraTreesRegressor(),
                      XGBRegressor(objective='reg:squarederror')]

results = dict()

for algo in algorithms:
    pipe = Pipeline([('preprocessing', preprocessing),
                     ('clf', algo)])

    pipe.fit(X_2, y_2)
    y_pred2 = pipe.predict(X_test2)
    rmse = mean_squared_error(y_test2, y_pred2)
    print(f"{algo.__class__.__name__:<17} - RMSE: {rmse:,.2f}")


regressor = RandomForestRegressor()
regressor.fit(X_, y_)
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Real Values': y_test, 'Predicted Values': y_pred})

# Function to create correlation heatmap
def correlation_heatmap(train):
    correlations = train.corr()

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f', square=True,
                linewidths=.5, annot=True, cmap="Blues", cbar_kws={"shrink": .70})
    plt.show()

correlation_heatmap(X_)



# Using built in spearman rank
stats.spearmanr(y_test, y_pred)

corr_df = pd.DataFrame()
corr_df['Features'] = X.columns
corr2 = []
for (columnName, columnData) in X.iteritems():
    corr2.append(stats.spearmanr(columnData, y)[0])

corr_df['Spearmans Rank Coef'] = np.abs(corr2)
corr_df = corr_df.sort_values(by=['Spearmans Rank Coef'], ascending=False)
corr_df = corr_df.reset_index(drop=True)
spearman_list = list(corr_df['Features'].values)



# Bar Plot corr df
plt.figure(figsize=(10, 10))
sns.barplot(x="Spearmans Rank Coef", y="Features", data=corr_df,
            palette="Reds_d").set_title("Spearmans' Rank Feature Importance")
plt.show()

solution = mrmr.mrmr_ensemble(features=X, targets=pd.DataFrame(
    y), solution_length=13, estimator='spearman')
mrmr_list = solution.iloc[0][0]
mrmr_list



# Using Shap
shap.initjs()
explainer = shap.TreeExplainer(regressor)
shap_values = explainer.shap_values(X_)


i = 2
shap.force_plot(explainer.expected_value,
                shap_values[i], features=X_.iloc[i], feature_names=X_.columns)
shap.summary_plot(shap_values, features=X_, feature_names=X_.columns)

shap.summary_plot(shap_values, features=X_,
                  feature_names=X_.columns, plot_type='bar')

shap_list = list(X.columns[np.argsort(np.abs(shap_values).mean(0))])
shap_list.reverse()
shap_list



# Using permutation strategy to resample
def permutation_importances(model, X_, y_,  X_test, y_test):
    model.fit(X_, y_)
    baseline = mean_squared_error(y_test, model.predict(X_test))
    imp = []
    for col in X_test.columns:
        save = X_test[col].copy()
        X_test[col] = np.random.permutation(X_test[col])
        m = mean_squared_error(y_test, model.predict(X_test))
        X_test[col] = save
        imp.append(baseline - m)

    perm_dict = {X_test.columns[i]: imp[i] for i in range(len(imp))}
    perm_df = pd.DataFrame()
    perm_df['Features'] = perm_dict.keys()
    perm_df['MSE'] = perm_dict.values()
    perm_df = perm_df.sort_values(by=['MSE'])
    perm_df = perm_df.reset_index(drop=True)
    perm_list = list(perm_df['Features'].values)

    return perm_list


perm_list = permutation_importances(regressor, X_, y_,  X_test, y_test)



# Using drop columns strategy to resample
def dropcol_importances(model, X_, y_, X_test, y_test):
    model.fit(X_, y_)
    baseline = mean_squared_error(y_test, model.predict(X_test))
    imp = []
    for col in X_.columns:
        X__ = X_.drop(col, axis=1)
        X_test_ = X_test.drop(col, axis=1)
        model_ = clone(model)
        model_.fit(X__, y_)
        m = mean_squared_error(y_test, model_.predict(X_test_))
        imp.append(baseline - m)

    drop_dict = {X_test.columns[i]: imp[i] for i in range(len(imp))}
    drop_dict
    drop_df = pd.DataFrame()
    drop_df['Features'] = drop_dict.keys()
    drop_df['MSE'] = drop_dict.values()
    drop_df = drop_df.sort_values(by=['MSE'])
    drop_df = drop_df.reset_index(drop=True)
    drop_list = list(drop_df['Features'].values)

    return drop_list


drop_list = dropcol_importances(regressor, X_, y_, X_test, y_test)


# Initialize
model = pca()

# Fit transform
out = model.fit_transform(X)

# Print the top features. The results show that f1 is best, followed by f2 etc
print(out['topfeat'])

# Model
model.plot()
plt.show()

# PCA List
PCA_list = list(out['topfeat'].feature)


def auto_selection(feature_list, X, y, X_test, y_test):
    regressor = RandomForestRegressor()
    regressor.fit(X, y)
    y_pred = regressor.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)

    k = len(feature_list)
    tracker = {}
    i = 1

    least_impo = feature_list[::-1]
    final = feature_list[::-1]

    for iteration, feat in enumerate(least_impo):
        if iteration == 12:
            break

        X = X.drop(feat, axis=1)
        X_test = X_test.drop(feat, axis=1)

        regressor = RandomForestRegressor()
        regressor.fit(X, y)

        new_list = permutation_importances(regressor, X, y,  X_test, y_test)
        feat = new_list[-1]

        y_pred = regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        tracker[k - iteration - 1] = mse

        mse_list = list(tracker.values())

        min_loc = mse_list.index(min(mse_list))

    k_impor = final[min_loc::]

    return k_impor[::-1], tracker


k_impor, tracker = auto_selection(perm_list, X, y, X_test, y_test)


# Lots of plotting
f, ax = plt.subplots(figsize=(8, 8))
sns.lineplot(x=tracker.keys(), y=tracker.values(), data=tracker,
             color='Green').set_title("K important features with respective RMSE ")
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# XG Boost
xg_model = XGBRegressor(objective='reg:squarederror', use_label_encoder=False)
imp_drop_xg = dropcol_importances(xg_model, X_, y_,  X_test, y_test)
imp_perm_xg = permutation_importances(xg_model, X_, y_,  X_test, y_test)

xg_df_drop = pd.DataFrame()
xg_df_drop['Features'] = X.columns
xg_df_drop['Difference'] = np.array(imp_drop_xg)
xg_df_drop = xg_df_drop.sort_values(by=['Difference'], ascending=True)
xg_df_drop = xg_df_drop.reset_index(drop=True)

xg_df_perm = pd.DataFrame()
xg_df_perm['Features'] = X.columns
xg_df_perm['Difference'] = np.array(imp_perm_xg)
xg_df_perm = xg_df_perm.sort_values(by=['Difference'], ascending=True)
xg_df_drop = xg_df_drop.reset_index(drop=True)


# Random Forest
rf_regressor = RandomForestRegressor()
imp_drop_rf = dropcol_importances(rf_regressor, X_, y_,  X_test, y_test)
imp_perm_rf = permutation_importances(rf_regressor, X_, y_,  X_test, y_test)

rf_df_drop = pd.DataFrame()
rf_df_drop['Features'] = X.columns
rf_df_drop['Difference'] = np.array(imp_drop_rf)
rf_df_drop = rf_df_drop.sort_values(by=['Difference'], ascending=True)
rf_df_drop = rf_df_drop.reset_index(drop=True)

rf_df_perm = pd.DataFrame()
rf_df_perm['Features'] = X.columns
rf_df_perm['Difference'] = np.array(imp_perm_rf)
rf_df_perm = rf_df_perm.sort_values(by=['Difference'], ascending=True)
rf_df_drop = rf_df_drop.reset_index(drop=True)


# Extra Tree
et_regressor = ExtraTreesRegressor()
imp_drop_et = dropcol_importances(et_regressor, X_, y_,  X_test, y_test)
imp_perm_et = permutation_importances(et_regressor, X_, y_,  X_test, y_test)

et_df_drop = pd.DataFrame()
et_df_drop['Features'] = X.columns
et_df_drop['Difference'] = np.array(imp_drop_et)
et_df_drop = et_df_drop.sort_values(by=['Difference'], ascending=True)
et_df_drop = et_df_drop.reset_index(drop=True)

et_df_perm = pd.DataFrame()
et_df_perm['Features'] = X.columns
et_df_perm['Difference'] = np.array(imp_perm_et)
et_df_perm = et_df_perm.sort_values(by=['Difference'], ascending=True)
et_df_drop = et_df_drop.reset_index(drop=True)

# Drop and Permutation importance models

fig, axes = plt.subplots(6, figsize=(6, 6*6))

dfs = [rf_df_drop, xg_df_drop, et_df_drop, rf_df_perm, xg_df_perm, et_df_perm]
colors = ["Reds_d", "Blues_d", "Greens_d", "Reds_d",
          "Blues_d", "Greens_d", "Reds_d", "Blues_d", "Greens_d"]
titles = ["Feature Importance Via Drop Columns - Random Forest", "Feature Importance Via Drop Columns - XGBOOST",
          "Feature Importance Via Drop Columns - Extra Tree", "Feature Importance Via Permuting Columns - Random Forest",
          "Feature Importance Via Permuting Columns - XGBOOST", "Feature Importance Via Permuting Columns - Extra Tree"]

for i, df in enumerate(dfs):
    sns.barplot(x="Difference",
                y="Features",
                data=dfs[i],
                palette=colors[i],
                ax=axes[i]).set_title(titles[i])

    corr_array = np.zeros((13, 100))

# Loop 100 times to generate a variance in feature importances
for i in range(100):
    corr2 = []
    # Bootstrap
    X_, X_test, y_, y_test = train_test_split(X, y, train_size=0.7)

    for (columnName, columnData) in X_.iteritems():
        corr2.append(stats.spearmanr(columnData, y_)[0])
    corr_array[:, i] = corr2

# Compute mean and standard deviation of Spearman Correlation for each feature
mean_corr = np.mean(corr_array, axis=1)
std_corr = np.std(corr_array, axis=1)

dev_df = pd.DataFrame()
dev_df['Features'] = X.columns
dev_df['Mean'] = abs(mean_corr)
dev_df['SD'] = std_corr
dev_df = dev_df.sort_values(by=['Mean'], ascending=False)
dev_df = dev_df.reset_index(drop=True)
dev_df

# Standard Deviation Plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.barh('Features', 'Mean', data=dev_df,
        xerr=dev_df['SD'], align='center', alpha=0.75, ecolor='black', capsize=5, color='#Fc9efe')
ax.set_xlabel('Spearman Correlation', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
ax.set_title('Standard Deviation of Spearman Correlation', fontsize=16)

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.invert_yaxis()
plt.show()
