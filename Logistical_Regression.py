import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

def model_fit_log(X, X_train, X_test, y_train, y_test, train):
    # Initialize the logistic regression model
    model = LogisticRegression( solver='lbfgs', max_iter=1000)

    # Train the model
    model.fit(X_train, y_train)

    threshold = 0.9
    # Make predictions on the test set
    start_time = time.time()
    raw_predictions = model.predict_proba(X_test)[:, 1]  # Probability of class 1
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    print(f"Time taken: {elapsed_time} seconds")
    predictions = (raw_predictions > threshold).astype(int)  # Apply the classification threshold

    # Evaluate the model
    d = {'y_pred': predictions, 'y_test': y_test}
    print(pd.DataFrame(d))
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    feature_rankings = model.coef_[0][:18]
    intercept = model.intercept_

    # Print the results
    print(f"Log regression Accuracy: {accuracy:.2f}")
    print("Log regression Classification Report:\n", report)
    print("Log regression Model Coefficients:", feature_rankings)
    print("Log regression Model Intercept:", intercept)
    #import pdb;pdb.set_trace()
    if train:
        feature_names =np.array(["wgprec", "dxprec", "dwprec","wgmaxqerr","actmaxqerr","dxmaxqerr","dwmaxqerr","maxds","meands","vards","signds" ,"convlayers", "actlayers", "actmethod", "wgmethod", "dxmethod", "dwmethod", "ISOaccuracy"])      
        #import pdb;pdb.set_trace()
    else:
        feature_names = np.array(["wgprec", "wgmaxqerr","actmaxqerr", "maxds","meands","vards","signds" ,"convlayers", "actlayers", "actmethod", "wgmethod"])  # Assuming X_train is a DataFrame
    #feature_names = X.columns
    #import pdb;pdb.set_trace()
    # Create a dictionary mapping features to their coefficients
    coefficients_dict = dict(zip(feature_names, feature_rankings))

    # Rank features based on the absolute value of coefficients
    sorted_features = sorted(coefficients_dict.items(), key=lambda x: abs(x[1]), reverse=True)

    # Print the ranked features
    #import pdb;pdb.set_trace()
    print("Log regression Ranked Features:")
    for feature, coefficient in sorted_features:
        print(f"{feature}: {coefficient:.4f}")

    top_features_indices = np.argsort(np.abs(feature_rankings))[::-1][:5]
    #print(top_feature_names)
    top_feature_names = [feature_names[i] for i in top_features_indices]
    top_feature_rankings = feature_rankings[top_features_indices]

    # Create a histogram for the top features
    plt.figure(figsize=(10, 6))
    plt.bar(top_feature_names, top_feature_rankings, color='skyblue', edgecolor='black')
    plt.title('Top Features Based on Feature Ranking (Logistic Regression)')
    plt.xlabel('Log regression Feature Name')
    plt.ylabel('Log regression Coefficient Value')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the plot to a file (e.g., PNG format)
    if train:
        plt.savefig('./Figures/top_features_logistic_regression_histogram_train.png')
    else: 
        plt.savefig('./Figures/top_features_logistic_regression_histogram_inf.png')


   # Create a plot of y_pred and y_test
    plt.figure(figsize=(8, 6))
    #plt.scatter(np.arange(len(y_test)), y_test, color='blue', label='Actual (y_test)')
    plt.scatter(np.arange(len(y_test)), predictions-y_test, color='red', label='Predicted (y_pred)', marker='x')
    plt.title('y_pred-y_test (Logistic Regression)')
    plt.xlabel('Log regression Test Sample Index')
    plt.ylabel('y_pred-y_test')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    if train:
        # Save the plot to a file (e.g., PNG format)
        plt.savefig('./Figures/logistic_regression_predictions_train.png')
    else:
        # Save the plot to a file (e.g., PNG format)
        plt.savefig('./Figures/logistic_regression_predictions_inf.png')

    # Format the time in scientific notation with 2 decimal places
    time_formatted = "{:.2e}".format(elapsed_time)
    # Save results to a file
    if train:
        output_filename = "./Outputs/output_log_train.txt"
    else:
        output_filename = "./Outputs/output_log_inf.txt"
    with open(output_filename, "w") as output_file:
        output_file.write(f"accuracy: {accuracy}\n")
        output_file.write(f"Classification report:\n {report}\n")
        output_file.write(f"Time Consumed: {time_formatted} seconds\n")
        output_file.write("Top 5 Features:\n")
        output_file.write(f"{top_feature_names}: {top_feature_rankings}\n")

    print(f"Results saved to {output_filename}")