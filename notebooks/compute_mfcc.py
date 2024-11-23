import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import librosa
import multiprocessing
import time


def read_file(f):
    try:
        data, sampling_rate = librosa.load(f, sr=44100, duration=7)
        # print(sampling_rate)
    except:
        print("This wav file {} has error".format(f))
        return None
    samples = []
    if len(data.shape) == 2:
        samples = data[:, 0]
    else:
        samples = data

    mfcc = librosa.feature.mfcc(y=samples, sr=sampling_rate, n_mfcc=128)
    return mfcc



if __name__ == '__main__':
    songs = ['Frozen', 'Hakuna', 'Potter', 'Mamma', 'Panther', 'Rain', 'Showman', 'StarWars']
    start = time.time()
    song_files = []
    labels = []
    for song in songs:
        directory = 'audio/Hum/' + song + '_hum'
        for filename in os.listdir(directory):
            song_files.append(os.path.join(directory, filename))
            labels.append(song)

    with multiprocessing.Pool(16) as p:
        mfcc_data = p.map(read_file, song_files)

    end = time.time()
    print("time: ", end - start)

    # plt.figure()
    # librosa.display.specshow(mfcc_data[2], x_axis='time')
    # plt.colorbar()
    # plt.show()

    # Save X and y
    np.save('rnn_attempt\mfcc_data.npy', mfcc_data)
    np.save('rnn_attempt\labels.npy', labels)
