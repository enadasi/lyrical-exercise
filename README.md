## 1. Project Overview
_____________________________
- __Inspiration__: I have loved music my entire life, and am always on the look for new music. While using Spotify one day, I became curious as to why song recommenders (not just theirs but others such as Apple's and Pandora's too) only focus on the melody, genre, or general sound of the song you are listening to in order to suggest the next song. In my opinion, the music that people connect to at the deepest level has meaningful lyrics that resonate with the listener, and a successful recommender should include lyrical topics, so I created a model that would do just that.
- __Goals__:
    - To analyze topics in popular rap music through the unsupervised learning of topic modeling
    - To create a "lyrically similar song" model, which is able to show the user other songs with similar meaningful lyrics and themes    
- __Methods__:
    - Conducted Natural Language Processing (NLP) pipeline to prepare lyrics data
    - Created TF-IDF model for lyrics data
    - Used Non-negative Matrix Factorization (NMF) topic modeling to discover latent topics within the music lyrics corpus
    - Defined a cosine similarity metric to identify lyrically similar songs based on TF-IDF model
- __Results__:
    - Topics were mostly expected but some interesting and unexpected ones. Topics included money, women, relationships/love, life/death, ad-libs, and more.
    - Similarity model successfully shows user other songs of similar themes and lyrics (can't quantify accuracy of unsupervised learning, but check it out for yourself!)

## 2. Method Descriptions
_____________________________
- Used several different packages and technologies, including: Jupyter Notebook, NumPy, Pandas, SciPy, NLTK (Natural Language Toolkit), Gensim, and Scikit Learn
- __Algorithms__:
    - Gensim was used to calculate Term Frequency - Inverse Document Frequency (TF-IDF) vectors for all songs with lyrics data
        - TF-IDF assigns lesser importance to words that are found in more documents, making rare words more important to the classification of the song's lyrics
    - NMF for topic modeling from Scikit Learn was carried out on these TF-IDF vectors
        - Creates matrices showing individual words' associations with topics, and individual songs' associations with topics
    - Cosine similarity metric from Scikit Learn used to identify lyrically similar songs

## 3. Instructions for Running Program
__________________________________________
- All necessary files but two are in the repository
- How to get remaining two files:
    1. __mxm_779k_matches.txt__:
        - Click on this link: http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/mxm_779k_matches.txt.zip
        - Name the file ```mxm_779k_matches.txt```, and insert the file into the "data" folder in the repository
        - Remove the header with the # symbols at the start of lines, just leaving data in text file
    2. __mxm_dataset_FULL.txt__:
        - Click on this link for the train set: http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/mxm_dataset_train.txt.zip
        - Click on this link for the test set: http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/mxm_dataset_test.txt.zip
        - Remove headers with the # symbols at the start of lines, just leaving data in the text file, and copy paste the test set into the train set text file, appending it directly to the bottom of the file
        - Rename the combined file to ```mxm_dataset_FULL.txt```, and insert the file into the "data" folder in the repository
- How to run the program:
    - Enter the lyrical-exercise repository from the terminal, and run the command ```python lyrical-exercise.py```
