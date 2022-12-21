import numpy as np
import pandas as pd

df = pd.read_csv('fingerprint100_hum_minmax.csv')

freq_range_name = ['sub_bass', 'bass', 'lower_midrange', 'mid_range', 'higher_midrange', 'presence', 'brilliance']

def find_max_freq(row):
    new_row = pd.Series({'label': 0, 'sub_bass': 0, 'bass': 0, 'lower_midrange': 0, 'mid_range': 0, 'higher_midrange': 0, 'presence': 0, 'brilliance': 0})
    new_row['label'] = row['label']
    i = 1
    for name in freq_range_name:
        new_row[name] = pd.to_numeric(row.iloc[i: i + 100].reset_index(drop=True)).argmax()
        i = i + 100
    return new_row

df = df.apply(find_max_freq, axis=1)
df.to_csv('fingerprint_hum_freq.csv', index=False)