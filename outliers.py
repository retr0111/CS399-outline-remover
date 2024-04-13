"""
    This is the original Back-End script for removing the outliers. 
"""
from wv import Model
from scipy.stats import zscore

def eliminateword_outliers() -> list:
    """
    model_path: path to the word vector model file
    words: list of str words

    :returns: list of str words without outliers
    """
    # Initialize the model

    model = Model("glove_short.txt")
    while True:
        words = input("Please enter a list seperated by commas: ").split(", ")

        
        word_models = []  # Stores the model vectors of the words in the input list

        for word in words:
            # fetch vector form of word from model
            word_vec = model.find_word(word)

            # End program if a vector representation of a word could not be found in the model.
            if word_vec is None:
                print(f'Vector representation could not be found in model for word: {word}')
                continue

            # Normalize and append word vector
            word_vec.normalize()    
            word_models.append(word_vec)

        # get similarity to all other words from their models
        
        similarity_values = []
        for word in word_models:
            sim = sum([word.similarity(w) for w in word_models]) 
            similarity_values.append(sim)
        # Calculate Z scores to determine true similarity
        sim_zscores = zscore(similarity_values)
        # Return words with Z-score less than or equal to 1
        filtered_out_words = [words[x] for x in range(len(words)) if sim_zscores[x] <= .12]
        filtered_in_words = [words[x] for x in range(len(words)) if sim_zscores[x] >= .12]

        print(f"The filtered out words are: {filtered_out_words}\nThe Words that stayed are: {filtered_in_words}")
# Example usage
print(eliminateword_outliers())

