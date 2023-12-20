import xgboost as xgb
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

def model_fit_Xg(X, X_train, X_test, y_train, y_test, train):
    # Create an XGBoost classifier
    model = xgb.XGBClassifier(objective='binary:logistic', random_state=2)

    # Train the model on the training data
    model.fit(X_train, y_train)

    start_time = time.time()
    # Make predictions on the testing data
    y_pred = model.predict(X_test)
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Time taken: {elapsed_time} seconds")
    d = {'y_pred': y_pred, 'y_test': y_test}
    print(pd.DataFrame(d))
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,zero_division=1)

    # Print the results
    print(f"xgboost Accuracy: {accuracy:.2f}")
    print("xgboost Classification Report:\n", report)

   # Create a plot of y_pred and y_test
    plt.figure(figsize=(8, 6))
    #plt.scatter(np.arange(len(y_test)), y_test, color='blue', label='Actual (y_test)')
    plt.scatter(np.arange(len(y_test)), y_pred-y_test, color='red', label='Predicted (y_pred-y_test)', marker='x')
    plt.title('y_pred-y_test (xgboost)')
    plt.xlabel('xgboost Test Sample Index')
    plt.ylabel('y_pred-y_test')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot to a file (e.g., PNG format)
    if train:
        plt.savefig('./Figures/xgboost_predictions_train.png')
    else:
        plt.savefig('./Figures/xgboost_predictions_inf.png')
    if train:
        feature_names = np.array(["wgprec", "dxprec", "dwprec","wgmaxqerr","actmaxqerr","dxmaxqerr","dwmaxqerr","maxds","meands","vards","signds" ,"convlayers", "actlayers", "actmethod", "wgmethod", "dxmethod", "dwmethod"])  # Assuming X_train is a DataFrame
        #import pdb;pdb.set_trace()
    else:
        feature_names = np.array(["wgprec", "wgmaxqerr","actmaxqerr", "maxds","meands","vards","signds" ,"convlayers", "actlayers", "actmethod", "wgmethod"])  # Assuming X_train is a DataFrame

    # Get feature importance scores
    feature_importance = model.feature_importances_
    #print(feature_importance)
    # Combine feature names and their importance scores
    features = list(zip(feature_names, feature_importance))
    #print(features)

    # Sort features by importance in descending order
    sorted_feature_importance = sorted(features, key=lambda x: x[1], reverse=True)
    print(sorted_feature_importance)

    # Extract the top 5 features and their importance scores
    top5_features = sorted_feature_importance[:5]

    # Extract feature names and their importance scores
    top5_feature_names = [feature[0] for feature in top5_features]
    importance_scores = [feature[1] for feature in top5_features]

    # Print the top 5 features and their importance scores
    #print("Top 5 Features:")
    #for feature_name, importance_score in zip(top5_feature_names, importance_scores):
        #print(f"{feature_name}: {importance_score}")

    # Plot the top 5 features
    plt.figure(figsize=(10, 6))
    plt.bar(top5_feature_names,  importance_scores, color='skyblue')
    plt.xlabel('Feature Importance')
    plt.title('Top 5 Features from XGBoost Model')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot as a PNG file
    if train:
        plt.savefig('./Figures/top5_features_bar_plot_train.png', bbox_inches='tight')
    else:
        plt.savefig('./Figures/top5_features_bar_plot_inf.png', bbox_inches='tight')
    # Format the time in scientific notation with 2 decimal places
    time_formatted = "{:.2e}".format(elapsed_time)
    # Save results to a file
    if train:
        output_filename = "./Outputs/output_xgboost_train.txt"
    else:
        output_filename = "./Outputs/output_xgboost_inf.txt"
    with open(output_filename, "w") as output_file:
        output_file.write(f"accuracy: {accuracy}\n")
        output_file.write(f"Classification report:\n {report}\n")
        output_file.write(f"Time Consumed: {time_formatted} seconds\n")
        output_file.write("Top 5 Features:\n")
        output_file.write(f"{top5_features}: {importance_scores}\n")

    print(f"Results saved to {output_filename}")