### author: 20153055 Park Taehyeong

from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

###
# Load iris data
###

iris = datasets.load_iris()

###
# Default variables
###

split_rs = 100
tree_rs = 100
tree_maxdepth = 5
knn_nei = 5

###
# Features in real
###

# print(iris.keys()) # data, target, target_names, DESCR, feature_names
# print(iris.target_names) # setosa, versicolor, viginica
# print(iris.feature_names) # 4 features-sepal length and width, petal length and width

###
# Train and test data split
###

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=split_rs)
newDataSet= x_test

###
# Training data adapting
###

dt = dt(random_state=tree_rs, max_depth=tree_maxdepth)
dt = dt.fit(x_train, y_train)
knn = KNeighborsClassifier(n_neighbors=knn_nei)
knn.fit(x_train,y_train)

###
# Printing predicted class 
###

predicted_class_dt = dt.predict(newDataSet)
predicted_class_knn = knn.predict(newDataSet)
print('D_tree prediction\n {}'.format(predicted_class_dt))
print('D_tree prediction\n {}'.format(predicted_class_knn))

print("\nCompare decision tree and KNN model as selected values")
print('D_tree train:\t{:.3f}'.format(dt.score(x_train, y_train)))
print('D_tree test:\t{:.3f}'.format(dt.score(x_test, y_test)))
print('KNN train:\t{:.3f}'.format(knn.score(x_train, y_train)))
print('KNN test:\t{:.3f}'.format(knn.score(x_test, y_test)))