from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np

df = pd.read_csv('frequency_attempt/fingerprint100_hum_minmax.csv')

two_songs = ['Frozen', 'Hakuna']
three_songs = ['Frozen', 'Hakuna', 'Potter']
five_songs = ['Frozen', 'Hakuna', 'Potter', 'Panther', 'StarWars']
all_songs = ['Frozen', 'Hakuna', 'Mamma', 'Panther', 'Potter', 'Rain', 'Showman', 'StarWars']
songs_lst = [two_songs, three_songs, five_songs, all_songs]

details_file = open('frequency_attempt/details.txt', 'w')
results_file = open('frequency_attempt/results.txt', 'w')

for songs in songs_lst:
    song_str = ''
    pos_hidden_layer_sizes = []
    cur_df =  df
    for song in ['Frozen', 'Hakuna', 'Mamma', 'Panther', 'Potter', 'Rain', 'Showman', 'StarWars']:
        if song not in songs:
            cur_df = cur_df[cur_df['label'] != song]

    if len(songs) == 2:
        song_str = 'Two songs: {}'.format(songs)
        pos_hidden_layer_sizes = [(700, 700), (700, 70), (700, 7), (700, 70, 7)]
    elif len(songs) == 3:
        song_str = 'Three songs: {}'.format(songs)
        pos_hidden_layer_sizes = [(700, 700), (700, 70), (700, 7), (700, 70, 7), (700, 700, 70), (700, 700, 7), (700, 350, 7), (700, 700, 350, 7), (700, 350, 70, 7)]
    elif len(songs) == 5:
        song_str = 'Five songs: {}'.format(songs)
        pos_hidden_layer_sizes = [(700, 70, 7), (700, 700, 70), (700, 700, 7), (700, 350, 7), (700, 700, 350, 7), (700, 350, 70, 7), (700, 700, 700, 350), (700, 700, 700), (700, 350, 175, 70), (700, 350, 350, 7), (700, 350, 175, 70, 7)]
    else:
        song_str = 'Eight songs: {}'.format(songs)
        pos_hidden_layer_sizes = [(700, 700, 70), (700, 700, 7), (700, 700, 350, 7), (700, 350, 70, 7), (700, 700, 700, 350), (700, 700, 700, 7), (700, 700, 700, 700), (700, 700, 700), (700, 350, 175, 70), (700, 350, 175, 7), (700, 350, 175, 175), (700, 350, 350, 7), (700, 350, 175, 70, 7), (700, 700, 175, 70, 7), (700, 350, 350, 70, 7)]

    print(song_str)
    details_file.write(song_str + '\n')
    results_file.write(song_str + '\n')

    max_train_accuracy = 0
    max_test_accuracy = 0
    max_hidden_layer_size = (0, 0)

    y = cur_df['label']
    X = cur_df.drop('label', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12345)

    for hidden_layer_sizes in pos_hidden_layer_sizes:
        print(hidden_layer_sizes)
        clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=12345, max_iter=500).fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        if test_accuracy > max_test_accuracy:
            max_test_accuracy = test_accuracy
            max_train_accuracy = train_accuracy
            max_hidden_layer_size = hidden_layer_sizes

        details_file.write('\tMLPClassifier(hidden_layer_sizes={}, random_state=12345, max_iter=500)\n'.format(pos_hidden_layer_sizes))
        details_file.write('\t\tTrain accuracy: {}\n'.format(train_accuracy))
        details_file.write('\t\tTest accuracy: {}\n\n'.format(test_accuracy))

    details_file.write('\n\n')

    results_file.write('\tMLPClassifier(hidden_layer_sizes={}, random_state=12345, max_iter=500)\n'.format(max_hidden_layer_size))
    results_file.write('\t\tTrain accuracy: {}\n'.format(max_train_accuracy))
    results_file.write('\t\tTest accuracy: {}\n\n'.format(max_test_accuracy))
    results_file.write('\n\n')

results_file.close()
details_file.close()