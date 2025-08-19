# iris_classifier.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 


#load data 
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)   
df['target'] = iris.target
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})


#visualize data
plt.figure(figsize=(10, 6))
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'], label=species)
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.legend()
    plt.savefig('iris_plot.png')
    plt.close()


#Prepare data for training
X = df.drop(['target', 'species'], axis=1)
Y = df['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


#Train the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, Y_train)


#Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(f"Model Accuracy: {accuracy :.2f}")


#Make a prediction
new_flower = [[5.1, 3.5, 1.4, 0.2]]  
prediction = model.predict(new_flower)
print(f"Predicted species: {iris.target_names[prediction][0]}")
