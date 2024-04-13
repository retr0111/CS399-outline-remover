from wv import Model
from scipy.stats import zscore
import streamlit as st

def eliminateword_outliers(words: list) -> tuple:
    """
    model_path: path to the word vector model file
    words: list of str words

    :returns: tuple containing lists of words without outliers and words that stayed
    """
    # Initialize the model
    model = Model(".gitattributes") # This is the model that is used as an example(glove_short.txt) but using lfs, if you want to change just change the path. 
    
    
    word_models = []  # Stores the model vectors of the words in the input list

    for word in words:
        # fetch vector form of word from model
        word_vec = model.find_word(word)

        # End program if a vector representation of a word could not be found in the model.
        if word_vec is None:
            st.warning(f'Vector representation could not be found in model for word: {word}')
            break

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

    return filtered_out_words, filtered_in_words

# Streamlit app
def main():
    st.title("Word Outlier Elimination App")
    st.write("Enter a list of words separated by commas to eliminate outliers.")

    input_words = st.text_input("Enter words:")
    if input_words:
        words = [word.strip() for word in input_words.split(",")]
        filtered_out_words, filtered_in_words = eliminateword_outliers(words)
        
        st.write("Words that are filtered out:")
        st.write(filtered_out_words)

        st.write("Words that stayed:")
        st.write(filtered_in_words)

if __name__ == "__main__":
    main()
