from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

songs = ['Frozen', 'Hakuna', 'Potter']
freq_range = [16, 60, 250, 500, 2000, 4000, 6000, 20000]
freq_range_name = ['sub_bass', 'bass', 'lower_midrange', 'mid_range', 'higher_midrange', 'presence', 'brilliance']
for song in songs:
    for i in range(3):
        data = []
        sampling_rate = 0
        if song == 'Frozen':
            sampling_rate, data = wavfile.read('Frozen/{}.wav'.format(5798 + i))
        elif song == 'Hakuna':
            sampling_rate, data = wavfile.read('Hakuna/{}.wav'.format(3331 + i))
        else:
            sampling_rate, data = wavfile.read('Potter/0{}.wav'.format(100 + i))

        samples = []
        if len(data.shape) == 2:
            samples = data[:, 0]
        else:
            samples = data

        ft = np.fft.fft(samples)

        freq_domain = np.abs(ft)
        mx = np.max(freq_domain)
        normalized_freq_domain = (freq_domain * 10000) / mx

        freq = np.fft.fftfreq(ft.size) * 44100

        print(sampling_rate)
        print(len(samples))
        print(freq)

    
        plt.figure()
        
        #plt.plot(freq_domain)
        #plt.plot(freq, freq_domain)
        plt.plot(freq, normalized_freq_domain)
        plt.ylabel('Amplitude')
        plt.xlabel('Frequency (not Hz)')

        plt.savefig('frequency domain/{}/{}_{}'.format(song, song, i))

