from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import os

songs = ['Frozen', 'Hakuna', 'Potter', 'Mamma', 'Panther', 'Rain', 'Showman', 'StarWars']
freq_range = [16, 60, 250, 500, 2000, 4000, 6000, 20000]
freq_range_name = ['sub_bass', 'bass', 'lower_midrange', 'mid_range', 'higher_midrange', 'presence', 'brilliance']

num_of_bins = 100

output_file = open('fingerprint{}_hum.csv'.format(num_of_bins), 'w')
output_file.write('label')
for i in range(len(freq_range_name)):
    for j in range(num_of_bins):
        output_file.write(',{}_{}'.format(freq_range_name[i], j))
output_file.write('\n')

for song in songs:
    directory = song + '_hum'
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        print(f)
        sampling_rate, data = (0, 0)
        try:
            sampling_rate, data = wavfile.read(f)
        except:
            print("This wav file {} has error".format(f))
            continue

        samples = []
        if len(data.shape) == 2:
            samples = data[:, 0]
        else:
            samples = data

        ft = np.fft.fft(samples)
        amplitude = np.abs(ft)
        
        from sklearn.preprocessing import MinMaxScaler
        normalized_amplitude = MinMaxScaler().fit_transform(amplitude.reshape(-1, 1)).flatten()

        freq = np.fft.fftfreq(ft.size) * sampling_rate

        freq_amp = np.transpose(np.concatenate((freq.reshape((1, -1)), normalized_amplitude.reshape((1, -1))), axis=0))

        output_file.write(song)

        for i in range(len(freq_range_name)):
            step = (freq_range[i + 1] - freq_range[i]) / num_of_bins
            for j in range(num_of_bins):
                low_bound = freq_range[i] + j * step
                hi_bound = freq_range[i] + (j + 1) * step
                lst = freq_amp[(freq_amp[:, 0] >= low_bound) & (freq_amp[:, 0] < hi_bound)]
                max_amp = np.max(lst[:, 1])

                output_file.write(',{}'.format(max_amp))
        output_file.write('\n')


output_file.close()


