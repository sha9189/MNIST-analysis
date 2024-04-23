# %%
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,balanced_accuracy_score,confusion_matrix, ConfusionMatrixDisplay,roc_auc_score,top_k_accuracy_score
from keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV







# %%

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X.shape,train_y.shape,test_X.shape,test_y.shape

# %%
# Plotting some sample data with their true labels
for i in range(3):
    plt.imshow(train_X[i])
    plt.show()
    print(f'True Label: {train_y[i]}')

# %%
#Checking For Null in Data
print(f"Number of NaN in Training Data: {np.isnan(train_X).sum()}")

print(f"Number of NaN in Test Data: {np.isnan(test_X).sum()}")

# %%
# Distribution of Classes in Training Data
unique, counts = np.unique(train_y, return_counts=True)
counts = np.round((counts / counts.sum())*100,2)
print(np.asarray((unique, counts)).T)

# %%
# Distribution of Classes in Test Data
unique, counts = np.unique(test_y, return_counts=True)
counts = np.round((counts / counts.sum())*100,2)
print(np.asarray((unique, counts)).T)

# %%
# Reshaping Data
X_train = train_X.reshape(train_X.shape[0],train_X.shape[1]*train_X.shape[2])
X_test = test_X.reshape(test_X.shape[0],test_X.shape[1]*test_X.shape[2])

# %%
# Removing all features with constant values
cols_with_const = pd.DataFrame(X_train).nunique(dropna=False) == 1
print(f"removing {cols_with_const.values.sum()} Columns with constant values")
X_train = X_train[:,~cols_with_const.values]
X_test = X_test[:,~cols_with_const.values]


# %%
#Scaling the Data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# %%
#Performing PCA on Data to reduce dimensionality
pca = PCA()
pca.fit(X_train)


# %%
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumulative_variance_ratio, marker='o')

plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio by Principal Components')
plt.show()

# %%
pca_df = pd.DataFrame(pca.explained_variance_ratio_,columns=['value'])
pca_df = pca_df.sort_values(by='value',ascending=False)
pca_df['cum_value'] = pca_df['value'].cumsum()
pca_df = pca_df.reset_index(drop=True)
pca_df






# %%
num_componets = pca_df[pca_df['cum_value'] > 0.95].index[0]
print(f'{(num_componets)} pca components required for greater than 95% explained variance')



# %%
#Refitting PCA with n_components = 330 and reducing dimensionality
pca = PCA(n_components=330)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(X_train_pca.shape,X_test_pca.shape)

# %% [markdown]
# # Building Model with PCA

# %%
# Fitting Gradient Boosted Trees
lgb_train = lgb.Dataset(X_train_pca, label=train_y,free_raw_data=False)
lgb_test = lgb.Dataset(X_test_pca, label=test_y,free_raw_data=False)

# specify your configurations as a dict
params = {
    "boosting_type": "gbdt",
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_leaves": 100,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
    "num_class" :10,
}

model = lgb.train(params,
                  lgb_train)
print('LGBM Performance')

# Multi Class Classification Metrics
test_pred = model.predict(X_test_pca)
test_pred_acc = test_pred.argmax(axis=1)
print(f"The balanced accuracy is : {balanced_accuracy_score(test_y,test_pred_acc):.3f}")
labels = [i for i in range(10)]
cm = confusion_matrix(test_y, test_pred_acc, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot()
plt.show()

print(f"ROC AUC SCORE (One vs One ) : {roc_auc_score(test_y,test_pred,multi_class='ovo'):.3f}")
print(f" Top 2 Accuracy score : {top_k_accuracy_score(test_y, test_pred, k=2):.3f}")



# %%
# Fitting Random Forest
model = RandomForestClassifier(n_estimators=200,n_jobs=-1)
model.fit(X_train_pca,train_y)

print('Random Forest Performance')
# Multi Class Classification Metrics
test_pred = model.predict_proba(X_test_pca)
test_pred_acc = test_pred.argmax(axis=1)
print(f"The balanced accuracy is : {balanced_accuracy_score(test_y,test_pred_acc):.3f}")
labels = [i for i in range(10)]
cm = confusion_matrix(test_y, test_pred_acc, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot()
plt.show()

print(f"ROC AUC SCORE (One vs One ) : {roc_auc_score(test_y,test_pred,multi_class='ovo'):.3f}")
print(f" Top 2 Accuracy score : {top_k_accuracy_score(test_y, test_pred, k=2):.3f}")

# %% [markdown]
# # Building Model Without PCA

# %%

# Fitting Gradient Boosted Trees
lgb_train = lgb.Dataset(X_train, label=train_y,free_raw_data=False)
lgb_test = lgb.Dataset(X_test, label=test_y,free_raw_data=False)

# specify your configurations as a dict
params = {
    "boosting_type": "gbdt",
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_leaves": 100,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
    "num_class" :10,
}

model = lgb.train(params,
                  lgb_train)
print('LGBM Performance')
# Multi Class Classification Metrics
test_pred = model.predict(X_test)
test_pred_acc = test_pred.argmax(axis=1)
print(f"The balanced accuracy is : {balanced_accuracy_score(test_y,test_pred_acc):.3f}")
labels = [i for i in range(10)]
cm = confusion_matrix(test_y, test_pred_acc, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot()
plt.show()

print(f"ROC AUC SCORE (One vs One ) : {roc_auc_score(test_y,test_pred,multi_class='ovo'):.3f}")
print(f" Top 2 Accuracy score : {top_k_accuracy_score(test_y, test_pred, k=2):.3f}")

# %%
# Fitting Random Forest
model = RandomForestClassifier(n_estimators=200,n_jobs=-1)
model.fit(X_train,train_y)

print('Random Forest Performance')
# Multi Class Classification Metrics
test_pred = model.predict_proba(X_test)
test_pred_acc = test_pred.argmax(axis=1)
print(f"The balanced accuracy is : {balanced_accuracy_score(test_y,test_pred_acc):.3f}")
labels = [i for i in range(10)]
cm = confusion_matrix(test_y, test_pred_acc, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot()
plt.show()

print(f"ROC AUC SCORE (One vs One ) : {roc_auc_score(test_y,test_pred,multi_class='ovo'):.3f}")
print(f" Top 2 Accuracy score : {top_k_accuracy_score(test_y, test_pred, k=2):.3f}")

# %%
# Choosing Best Hyperparamter for Random Forest Using RandomizedSearchCV

param_dist = {
    "n_estimators": list(range(1,201)),
    "max_depth" : list(range(1,15)),
    "min_samples_leaf": list(range(1,20))
}

model = RandomForestClassifier()

n_iter_search = 50
random_search = RandomizedSearchCV(
    model, param_distributions=param_dist, n_iter=n_iter_search,n_jobs=-1,cv=3,verbose=1,scoring='balanced_accuracy'
)
random_search.fit(X_train,train_y)

# %%
cv_result_df = pd.DataFrame(random_search.cv_results_)
cv_result_df

# %%
#Best Parameters
random_search.best_params_

# %%
#Fitting Random Forest with Best Parameters
model = RandomForestClassifier(**random_search.best_params_,n_jobs=-1)
model.fit(X_train,train_y)

print('Random Forest Performance')
# Multi Class Classification Metrics
test_pred = model.predict_proba(X_test)
test_pred_acc = test_pred.argmax(axis=1)
print(f"The balanced accuracy is : {balanced_accuracy_score(test_y,test_pred_acc):.3f}")
labels = [i for i in range(10)]
cm = confusion_matrix(test_y, test_pred_acc, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot()
plt.show()

print(f"ROC AUC SCORE (One vs One ) : {roc_auc_score(test_y,test_pred,multi_class='ovo'):.3f}")
print(f" Top 2 Accuracy score : {top_k_accuracy_score(test_y, test_pred, k=2):.3f}")


