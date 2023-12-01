from Preprocess_AxC import import_data, preprocess_data
from Multiple_Linear_Regression import model_fit_linear
def main():
    Dataset_path = './Data/EE8351_DATASET_for_Project.csv'
    dataset_dict=import_data(Dataset_path)
    #print(dataset_dict)

    preprocess_data(dataset_dict)
    model_fit_linear()
    

if __name__ == '__main__':
	main()