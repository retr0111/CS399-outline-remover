#import numpy as np
from wv import Model
#from scipy.stats import zscore

model = Model("glove.6B/glove.6B.100d.txt")

while True:
    words = input("Please enter a comma separated list of words: ").split(",")
    word_vectors = []
    for word in words:
        vector = model.find_word(word)
        if vector is not None:
            word_vectors.append(vector.vector)

    if len(word_vectors) == 0:
        print("No word vectors found for input words.")
        continue

    # Calculate z-scores
    z_scores = zscore(np.array(word_vectors))

    # Remove outliers based on z-scores
    threshold = 3
    word_vectors_filtered = [word_vectors[i] for i, z in enumerate(z_scores) if abs(z) < threshold]
    
    print("Original words:", words)
    print("Words without outliers:", [words[i] for i, z in enumerate(z_scores) if abs(z) < threshold])
