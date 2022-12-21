import os
import csv
with open('metadata.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        if row[1] in ['Mamma', 'Panther', 'Rain', 'Showman', 'StarWars']:
            os.rename('{}/{}'.format(row[1], row[0]), '{}_hum/{}'.format(row[1], row[0]))