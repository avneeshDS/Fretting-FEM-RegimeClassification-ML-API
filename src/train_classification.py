import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (GridSearchCV,RandomizedSearchCV,RepeatedStratifiedKFold)

#Loading data from the csv file
df= pd.read_csv(r"V:\VZU\Python\Fretting_models\fretting_fem_regimeclassification_ml_api\data\ML_fretting_screening.csv", encoding="cp1252")

#Dropping rows with null values
df_ml = df.dropna().reset_index(drop=True)

#Derive features for better relationship between data
df_ml['µP'] = df_ml['µ'] * df_ml['P']
df_ml['µPD'] = df_ml['µ'] * df_ml['P'] * df_ml['D']
df_ml['D_over_a'] = df_ml['D'] / df_ml['a']

#Seperating features and target
X = df_ml[['P', 'D', 'µ', 'a', 'µP', 'µPD', 'D_over_a']]
Y = df_ml[['Regime']]

#Split for training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=41)
print(X_test)

# Convert target variables from column vectors to 1D arrays.
Y_train = np.asarray(Y_train).ravel()
Y_test = np.asarray(Y_test).ravel()

# Class encoding: Gross slip regime = 0, Mixed slip regime = 1, and partial slip regime = 2
class_label = [0, 1, 2]
class_name = ["Gross", "Mixed", "Partial"]
print("\nTraining-class distribution:")
print(pd.Series(Y_train).value_counts().sort_index())

print("\nTest-class distribution:")
print(pd.Series(Y_test).value_counts().sort_index())

# Stratified repeated-fold cross validation and scoring metrics
#Stratification ensures that every training and test fold contains a representative proportion of gross slip, mixed slip, and partial slip samples
cv_strategy = RepeatedStratifiedKFold(n_splits=5,n_repeats=10,random_state=42)

scoring_metrics = {"Accuracy": "accuracy","Precision_macro": "precision_macro", "Recall_macro":
                   "recall_macro","F1_macro": "f1_macro"}

# Pipeline for selected algorithms
logistic_pipeline = Pipeline([("scalar", StandardScaler()), ("model", LogisticRegression(solver="lbfgs", max_iter=5000, random_state=42))]) 
svc_pipeline = Pipeline([("scalar", StandardScaler()), ("model", SVC(probability=True, random_state=42))])
rf_pipeline = Pipeline([("model", RandomForestClassifier(random_state=42, n_jobs=1))])
xgb_pipeline = Pipeline([("model", xgb.XGBClassifier(objective = "multi:softprob", num_class = 3, eval_metric = "mlogloss", random_state = 42, n_jobs =1))])

#Hyperparameter search space
logistic_param_grid = {"model__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000], "model__class_weight": [None, "balanced"], "model__fit_intercept": [True, False]}
svc_param_grid = {"model__kernel": ["rbf"], "model__C": [0.1, 1, 10, 100, 1000], "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1.0], "model__class_weight": [None, "balanced"], "model__tol": [1e-4, 1e-3, 1e-2]}
rf_param_grid = {"model__n_estimators": [100, 200, 300, 500], "model__max_depth": [None, 3,5,8,12], "model__min_samples_split": [2,4,6,8], "model__min_samples_leaf": [1, 2, 3, 4], "model__max_features": ["sqrt", "log2", 0.5, 0.75, 1.0], "model__class_weight": [None, "balanced", "balanced_subsample"]}
xgb_param_dist = {"model__n_estimators": [50, 100, 200, 300 , 500], "model__max_depth": [2,3,4,5,6], "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2], "model__subsample": [0.6,0.8,1.0], "model__colsample_bytree": [0.6,0.8,1.0], "model__min_child_weight": [1,3,5], "model__reg_alpha": [0,0.001,0.01,0.1,1.0], "model__reg_lambda": [0.1,1.0,5.0, 10.0]}

# Configuring hyperparameter searches
logistic_search = GridSearchCV(estimator=logistic_pipeline, param_grid=logistic_param_grid, scoring=scoring_metrics, refit="F1_macro", cv=cv_strategy, n_jobs=-1,
                               return_train_score=True)
#Macro-F1 was selected for equal weight to all three regimes.
svc_search = GridSearchCV(estimator=svc_pipeline, param_grid=svc_param_grid, scoring=scoring_metrics,
                           refit="F1_macro", cv=cv_strategy, return_train_score=True)
rf_search = RandomizedSearchCV(estimator=rf_pipeline, param_distributions=rf_param_grid, n_iter=60, scoring=scoring_metrics, refit="F1_macro", cv=cv_strategy, random_state=42, n_jobs=-1, 
                               return_train_score=True)
# RandomizedSearchCV was used to reduce the search space as the combination of search is too high.
xgb_search = RandomizedSearchCV(estimator=xgb_pipeline, param_distributions=xgb_param_dist, scoring=scoring_metrics, n_iter=80, refit="F1_macro", cv=cv_strategy, random_state=42, n_jobs=-1, return_train_score=True)
searches = {"Logistic Regression": logistic_search, "Support Vector Classifier": svc_search, "Random Forest Classifier": rf_search, "XGBoost": xgb_search}

# Performing hyperparameter optimization
for model_name, search in searches.items():
    print("\n" + "--" *20)
    print(f"optimizing {model_name}")
    search.fit(X_train, Y_train)
    print("\nSelected hyperparameters:")
    print(search.best_params_)
    
    # Index of the selected hyperparameter combination
    best_idx = search.best_index_

    cv_f1_mean = search.cv_results_["mean_test_F1_macro"][best_idx]
    cv_f1_sd = search.cv_results_["std_test_F1_macro"][best_idx]
    cv_accuracy_mean = search.cv_results_["mean_test_Accuracy"][best_idx]
    cv_accuracy_sd = search.cv_results_["std_test_Accuracy"][best_idx]
    print("\nBest five-fold CV Macro-F1:", f"{cv_f1_mean:.3f} ± {cv_f1_sd:.3f}")   
    print("Corresponding five-fold CV Accuracy:",f"{cv_accuracy_mean:.3f} ± "f"{cv_accuracy_sd:.3f}")

# Evaluation of the optimized models
classification_results = []
predictions = {}
fitted_models = {}
fold_score_rows =[]
for model_name, search in searches.items():
    best_model = search.best_estimator_
    best_idx = search.best_index_
    cv_results = search.cv_results_

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    predictions[model_name] = y_test_pred
    fitted_models[model_name] = best_model

    # CV metrics for selected model

    cv_accuracy_mean = cv_results["mean_test_Accuracy"][best_idx]
    cv_accuracy_sd = cv_results["std_test_Accuracy"][best_idx]
    cv_precision_mean = cv_results["mean_test_Precision_macro"][best_idx]
    cv_precision_sd = cv_results["std_test_Precision_macro"][best_idx]
    cv_recall_mean = cv_results["mean_test_Recall_macro"][best_idx]
    cv_recall_sd = cv_results["std_test_Recall_macro"][best_idx]
    cv_f1_mean = cv_results["mean_test_F1_macro"][best_idx]
    cv_f1_sd = cv_results["std_test_F1_macro"][best_idx]

    # Training accuracy
    train_accuracy = accuracy_score(Y_train, y_train_pred)
    train_f1 = f1_score(Y_train, y_train_pred, labels=class_label, average="macro", zero_division=0)

    # Independent test metrices
    test_accuracy = accuracy_score(Y_test, y_test_pred)
    test_precision = precision_score(Y_test, y_test_pred, labels=class_label, average="macro", zero_division=0)
    test_recall = recall_score(Y_test, y_test_pred, labels=class_label, average="macro", zero_division=0)
    test_f1 = f1_score(Y_test, y_test_pred, labels=class_label, average="macro", zero_division=0)
    classification_results.append({"Model": model_name, "Train_Accuracy": train_accuracy, "Train_Macro_F1": train_f1, "CV_Accuracy_Mean": cv_accuracy_mean, "CV_Accuracy_SD":
                                   cv_accuracy_sd, "CV_Precision_Mean": cv_precision_mean,
                                   "CV_Precision_SD": cv_precision_sd,
                                   "CV_Recall_Mean": cv_recall_mean, "CV_Recall_SD": cv_recall_sd, "CV_F1_Mean": cv_f1_mean, "CV_F1_SD": cv_f1_sd, "Test_Accuracy": test_accuracy, "Test_Precision": test_precision, "Test_Recall": test_recall, "Test_F1": test_f1})


    # Individual fold results 
    for fold_number in range(5):
        fold_score_rows.append({"Model": model_name, "Fold": fold_number + 1, "Accuracy": cv_results[f"split{fold_number}_test_Accuracy"][best_idx], "Macro_Precision": cv_results[
                        f"split{fold_number}_test_Precision_macro"][best_idx], "Macro_Recall": cv_results[f"split{fold_number}_test_Recall_macro"][best_idx],"Macro_F1": cv_results[f"split{fold_number}_test_F1_macro"][best_idx]})

    #print optimized performance

    print(f"Optimized performance: {model_name}")
    print(f"Five-fold CV Accuracy:"f"{cv_accuracy_mean:.3f} ± "f"{cv_accuracy_sd:.3f}")
    print(f"Five-fold CV Macro-Precision:"f"{cv_precision_mean:.3f} ± "f"{cv_precision_sd:.3f}")
    print(f"Five-fold CV Macro-Recall:"f"{cv_recall_mean:.3f} ± "f"{cv_recall_sd:.3f}")
    print(f"Five-fold CV Macro-F1:"f"{cv_f1_mean:.3f} ± "f"{cv_f1_sd:.3f}")
    print(f"Independent test Accuracy:"f"{test_accuracy:.3f}")
    print(f"Independent test Macro-Precision:"f"{test_precision:.3f}")
    print(f"Independent test Macro-Recall:"f"{test_recall:.3f}")
    print(f"Independent test Macro-F1:"f"{test_f1:.3f}")