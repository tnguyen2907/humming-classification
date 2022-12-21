from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np

df = pd.read_csv('fingerprint100_hum_minmax.csv')
used_songs = ['Frozen', 'Hakuna', 'Mamma', 'Panther', 'Potter',]

for song in ['Frozen', 'Hakuna', 'Mamma', 'Panther', 'Potter', 'Rain', 'Showman', 'StarWars']:
    if song not in used_songs:
        df = df[df['label'] != song]
print(df.head())
y = df['label']
X = df.drop('label', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12345)

clf = MLPClassifier(hidden_layer_sizes=(700, 700, 350), random_state=12345, max_iter=500).fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
print('Train accuracy: {}'.format(accuracy_score(y_train, y_train_pred)))

y_test_pred = clf.predict(X_test)
print('Test accuracy: {}'.format(accuracy_score(y_test, y_test_pred)))