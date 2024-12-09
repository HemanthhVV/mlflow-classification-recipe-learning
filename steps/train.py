"""
This module defines the following routines used by the 'train' step:

- ``estimator_fn``: Defines the customizable estimator type and parameters that are used
  during training to produce a model recipe.
"""
from typing import Dict, Any

import mlflow
from sklearn.model_selection import ParameterGrid
from sklearn. metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier,RandomForestClassifier
from xgboost import XGBClassifier


def estimator_fn(estimator_params: Dict[str, Any] = None) -> Any:
#     """
#     Returns an *unfitted* estimator that defines ``fit()`` and ``predict()`` methods.
#     The estimator's input and output signatures should be compatible with scikit-learn
#     estimators.
#     """
#     #
#     # FIXME::OPTIONAL: return a scikit-learn-compatible classification estimator with fine-tuned
#     #                  hyperparameters.
    base_classifiers = [
    ('logistic', LogisticRegression(max_iter=200)),
    ('svc', SVC(probability=True)),  # Set probability=True for SVC to work with stacking
    ('decision_tree', DecisionTreeClassifier()),
    ('random_forest',RandomForestClassifier())
    ]
    print("=========> Base Classifier set as Logistic Regression, SVC, Decision Tree, Random Forest")
    final_estimator = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    print("=========> Final Estimator set as XGB Classifier")

    stackClassifier = StackingClassifier(
       estimators=base_classifiers,final_estimator=final_estimator,cv=5
      )
    try:
      return stackClassifier
    except:
      raise NotImplementedError





# def train_with_multiple_algorithms(estimator_params:Dict = None):
#     # Load configuration values
#     hyperparameters = estimator_params["hyperparameters"]
#     run_only_model = estimator_params.get("run_only_model", None)  # Get the specific model to run
#     selection_metric = estimator_params["selection_criteria"]["metric"]
#     store_best_algorithm = estimator_params["store_best_model"]["algorithm"]

#     best_model = None
#     best_score = -float("inf")
#     best_params = None
#     best_algorithm_name = None

#     for algorithm_name, params in hyperparameters.items():
#         # Skip algorithms not specified in run_only_model
#         if run_only_model and algorithm_name != run_only_model:
#             print(f"Skipping {algorithm_name} as it is not specified in run_only_model.")
#             continue

#         print(f"Training {algorithm_name}...")
#         param_grid = list(ParameterGrid(params))  # Generate all hyperparameter combinations

#         for param_set in param_grid:
#             if algorithm_name == "RandomForest":
#                 model = RandomForestClassifier(**param_set, random_state=42)
#             elif algorithm_name == "LogisticRegression":
#                 model = LogisticRegression(**param_set, random_state=42)
#             elif algorithm_name == "SVC":
#                 model = SVC(**param_set, random_state=42, probability=True)
#             else:
#                 print(f"Unknown algorithm: {algorithm_name}")
#                 continue

#             # Train and evaluate the model
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#             score = f1_score(y_test, y_pred)  # Replace with dynamic metric if needed

#             # Log the results to MLflow
#             with mlflow.start_run():
#                 mlflow.log_param("algorithm", algorithm_name)
#                 mlflow.log_params(param_set)
#                 mlflow.log_metric(selection_metric, score)

#             # Update the best model if necessary
#             if score > best_score and algorithm_name == store_best_algorithm:
#                 best_model = model
#                 best_score = score
#                 best_params = param_set
#                 best_algorithm_name = algorithm_name

#     # Log the best model to MLflow Model Registry
#     if best_model is not None:
#         with mlflow.start_run():
#             mlflow.sklearn.log_model(best_model, "model")
#             mlflow.log_param("best_algorithm", best_algorithm_name)
#             mlflow.log_params(best_params)
#             mlflow.log_metric(selection_metric, best_score)
#             print(f"Best model logged: {best_algorithm_name} with score {best_score}")
#     else:
#         print("No valid model found.")

#     return best_model


# def estimator_fn(estimator_params: Dict[str, Any] = None) -> Any:
#     """
#     Returns an *unfitted* estimator that defines ``fit()`` and ``predict()`` methods.
#     The estimator's input and output signatures should be compatible with scikit-learn
#     estimators.
#     """
#     print("+==============================================> ",estimator_params['model_name'])

#     ## If no model name or no parameter given ,
#     ## Model will run the XGBoost default
#     # if estimator_params is None:
#       # estimator_params = {
#       #     'n_estimators': 400,           # Number of boosting rounds
#       #     'max_depth': 6,                # Maximum depth of a tree
#       #     'learning_rate': 0.1,          # Boosting learning rate (eta)
#       #     'subsample': 0.8,              # Subsample ratio of the training data
#       #     'colsample_bytree': 0.8,       # Subsample ratio of columns when constructing each tree
#       #     'min_child_weight': 1,         # Minimum sum of instance weight (hessian) needed in a child
#       #     'gamma': 0,                    # Minimum loss reduction required to make a further partition on a leaf node
#       #     'reg_alpha': 0,                # L1 regularization term on weights
#       #     'reg_lambda': 1,               # L2 regularization term on weights
#       #     'objective': 'binary:logistic',# Specify learning task and corresponding objective ('binary:logistic' for classification)
#       #     'verbosity': 1,                # Verbosity of printing messages during training
#       #     'booster': 'gbtree'            # Specify which booster to use ('gbtree', 'gblinear', 'dart')
#       # }
#     model_name = None
#     ## Default model creation
#     if estimator_params['model_name']:
#       model_name = estimator_params['model_name']
#       del(estimator_params['model_name'])
#     else:
#       model = xgboost.XGBClassifier(**estimator_params)

#     if model_name == 'RandomForest':
#        model = RandomForestClassifier(**estimator_params)

#     if model_name == 'XGBoost':
#        model = xgboost.XGBClassifier(**estimator_params)

#     if model_name == 'LightGBM':
#        model = lgbm.LGBMClassifier(**estimator_params)

#     if model_name == 'DecisionTree':
#        model = DecisionTreeClassifier(**estimator_params)

#     try:
#       return model
#     except:
#       raise NotImplementedError
