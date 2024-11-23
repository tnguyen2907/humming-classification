# Humming Classification

This project aims to classify songs from humming audio.  

**Dataset:** [Hums and Whistles Dataset](https://www.kaggle.com/datasets/jesusrequena/mlend-hums-and-whistles)  
**Songs:** ['Frozen', 'Hakuna', 'Mamma', 'Panther', 'Potter', 'Rain', 'Showman', 'StarWars']

**Note:** This project is a fork and adaptation of the final project for the course *CS 374: Machine Learning and Data Mining*, which I worked on alongside two classmates. The original project is available [here](https://github.com/CSCI374F22/project-tnguyen2907).

## Approach

### Using MLP on Features as the Most Prevalent Frequencies in the Audio
*Files: [mlp.py](https://github.com/tnguyen2907/humming-classification/tree/master/mlp/mlp.py) and [make_fingerprint_minmax.py](https://github.com/tnguyen2907/humming-classification/mlp/make_fingerprint_minmax.py)*  
The audio is converted from the time domain to the frequency domain using Fast Fourier Transform (FFT) to identify the most prevalent frequencies in the audio. These frequencies are grouped into 7 bins based on frequency ranges:  

- Sub-bass: 16 to 60 Hz  
- Bass: 60 to 250 Hz  
- Lower Midrange: 250 to 500 Hz  
- Midrange: 500 to 2000 Hz  
- Higher Midrange: 2000 to 4000 Hz  
- Presence: 4000 to 6000 Hz  
- Brilliance: 6000 to 20000 Hz  

Each range is further divided into 100 equal-length bins, resulting in a total of 700 bins. These extracted features are normalized and then input into a fully connected neural network with four hidden layers: (700, 700, 350, 70).  

### Using RNN on Features as MFCC
*Files: [rnn.ipynb](https://github.com/tnguyen2907/humming-classification/tree/master/notebooks/rnn.ipynb) and [compute_mfcc.py](https://github.com/tnguyen2907/humming-classification/tree/master/notebooks/compute_mfcc.py)*  
The audio is transformed into Mel-frequency cepstral coefficients (MFCC), which are an effective representation for audio classification tasks. These features are processed by a simple RNN model with two LSTM layers (each with 512 units) followed by a fully connected layer with 256 units and dropout for regularization.

### Using CNN (AlexNet) on Features as MFCC
*Files: [cnn.ipynb](https://github.com/tnguyen2907/humming-classification/tree/master/notebooks/cnn.ipynb) and [compute_mfcc.py](https://github.com/tnguyen2907/humming-classification/tree/master/notebooks/compute_mfcc.py)*  
The audio is converted to MFCCs, similar to the RNN approach. The model used is a modified AlexNet architecture with added batch normalization layers and dropout layers to enhance training stability. These MFCC features are fed into the CNN for classification.  

## Accuracy
- CNN (AlexNet): 0.939

- RNN: 0.576

- MLP: 0.547

The results demonstrate that the CNN-based approach using AlexNet significantly outperforms the other methods in classifying songs from humming audio, proving the effectiveness of AlexNet in audio classification tasks.