Identify Songs from Humming
Author: Trung Nguyen, Nam Anh Nguyen, Brian Do


This project to identify songs from humming audio using two approaches
    + Transform time domain into frequency domain, then train using standard Neural Network (implementation from scikit-learn)
    + Transform audio into images, then train using Convolutional Neural Network

Dataset: https://www.kaggle.com/datasets/jesusrequena/mlend-hums-and-whistles
Songs: ['Frozen', 'Hakuna', 'Mamma', 'Panther', 'Potter', 'Rain', 'Showman', 'StarWars']

Frequency domain approach (folder frequency_attempt)
File:
    make_fingerprint_minmax.py: Create dataset for training, transforming time domain into frequency domain
    mlp.py: Model
Result:
    Two songs: Test accuracy: 0.8821548821548821
    Three songs: Test accuracy: 0.7817371937639198
    Five songs: Test accuracy: 0.5792276964047937
    Eight songs: Test accuracy: 0.5465890183028286

Image + CNN approach (folder cnn_attempt)
File:
    train.ipynb: Create dataset for training, transforming audio into images, follow by the model afterward
Result:
    Two songs: Test accuracy: 0.9380
    Three songs: Test accuracy: 0.9086
    Five songs: Test accuracy: 0.8776
    Eight songs: Test accuracy: 0.8415
