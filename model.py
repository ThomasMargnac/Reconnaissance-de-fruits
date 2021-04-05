import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

if __name__ == '__main__' :

	#On lit le CSV
	df = pd.read_csv('dataframe.csv')

	#On sépare les données
	X = df.iloc[:,:6]
	y = df.iloc[:,-1]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

	#On déclare et on entraine notre modèle
	neigh = KNeighborsClassifier(n_neighbors=3)
	neigh.fit(X_train, y_train)

	#On évalue notre modèle
	train_acc = neigh.score(X_train, y_train)
	print("\nTaux de reconnaissance en apprentissage : ", train_acc)
	test_acc = neigh.score(X_test, y_test)
	y_pred = neigh.predict(X_test)
	print("\nTaux de reconnaissance en test : ", test_acc)
	print("\nVoici la matrice de confusion pour les données de test :\n\n",confusion_matrix(y_test, y_pred),"\n")
