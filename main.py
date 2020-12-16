from KNN.knn import KNN
from data_cleaning import load_kropt
from distance_metric import manhattan_metric
from policies_knn import majority_class
from weighting_knn import reliefF, equal_weight

train_data, test_data = load_kropt.load_train_test_fold('datasets/kropt', 1)
# knearest = KNN(manhattan_metric, 3, majority_class, reliefF)
knearest = KNN(manhattan_metric, 3, majority_class, equal_weight)

knearest.fit(train_data)
knearest.predict(test_data)




