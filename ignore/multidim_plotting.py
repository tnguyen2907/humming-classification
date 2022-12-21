import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

df = pd.read_csv('fingerprint_freq.csv')

y = df['label']
X = df.drop('label', axis=1)

pca = LDA(n_components=2) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X, y))

plt.scatter(transformed[y=='Frozen'][0], transformed[y=='Frozen'][1], label='Class Frozen', c='red')
plt.scatter(transformed[y=='Hakuna'][0], transformed[y=='Hakuna'][1], label='Class Hakuna', c='blue')
plt.scatter(transformed[y=='Potter'][0], transformed[y=='Potter'][1], label='Class Potter', c='lightgreen')

plt.legend()
plt.show()