from Preprocess_AxC import preprocess_data, transform_X_Y
from Utils import import_data
from Multiple_Linear_Regression import model_fit_linear
from Logistical_Regression import model_fit_log
from XGBoost import model_fit_Xg
preprocessed=1
def main():
    if not preprocessed:
        Dataset_path_inf = './Data_AxC/EE8351_DATASET_for_Project_inf.csv'
        dataset_dict_inf=import_data(Dataset_path_inf)
        #print(dataset_dict)

        preprocess_data(dataset_dict_inf,0)
        Dataset_path_train = './Data_AxC/EE8351_DATASET_for_Project_train.csv'
        dataset_dict_train=import_data(Dataset_path_train)
        #print(dataset_dict)

        preprocess_data(dataset_dict_train,1)
    train= 0
    X, X_train, X_test, y_train, y_test = transform_X_Y(train)
    #model_fit_linear(X_train, X_test, y_train, y_test )
    model_fit_log(X, X_train, X_test, y_train, y_test, train)
    model_fit_Xg(X, X_train, X_test, y_train, y_test, train)

    train = 1
    X, X_train, X_test, y_train, y_test = transform_X_Y(train)
    #model_fit_linear(X_train, X_test, y_train, y_test )
    model_fit_log(X, X_train, X_test, y_train, y_test,train)
    model_fit_Xg(X, X_train, X_test, y_train, y_test,train)
    

if __name__ == '__main__':
	main()