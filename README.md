# Lyrical Exercise
## A Data Science Project by Eric Nadasi

### Inspiration:
I have loved music my entire life, and am always on the look for new songs. While using Spotify one day, I became curious as to why song recommenders (not just Spotify's, also others such as Apple's and Pandora's) only focus on the melody, genre, or general sound of the song you are listening to in order to suggest the next song. In my opinion, the music that people connect to at the deepest level has meaningful lyrics that resonate with the listener, and a successful recommender should include songs with similar lyrical topics, so I created a model with Natural Language Processing that would do just that.

## 1. Project Overview
_____________________________

__Goals__:
- To analyze topics in popular music through the unsupervised learning of topic modeling
- To create a "lyrically similar song" model, which is able to show the user other songs with similar meaningful lyrics and themes  

__Methods__:
- Conducted Natural Language Processing (NLP) pipeline to prepare lyrics data
- Created TF-IDF model for lyrics data
- Tested both Latent Dirichlet Allocation and Non-negative Matrix Factorization (NMF) topic modeling
- Used the more successful NMF topic modeling to discover latent topics within the music lyrics corpus
- Defined a cosine similarity metric to identify lyrically similar songs based on TF-IDF model

__Results__:
- Topics were mostly expected but some interesting and unexpected ones
    - i.e.: For the Hip-Hop/Rap genre, topics included money, women, relationships/love, life/death, ad-libs, and more.
- Similar song program successfully shows user other songs of similar lyrics and themes (we can't quantify the accuracy of unsupervised learning, but check it out for yourself!)

## 2. Method Descriptions
_____________________________
- Coded in Python using several different packages and technologies, including: Jupyter Notebook, NumPy, Pandas, SciPy, NLTK (Natural Language Toolkit), Gensim, and Scikit Learn
- __Algorithms__:
    - Gensim was used to calculate Term Frequency - Inverse Document Frequency (TF-IDF) model vectors for all songs with lyrics data
        - TF-IDF assigns lesser importance to words that are found in more documents, making rare words more important to the classification of the song's lyrics
    - NMF for topic modeling from Scikit Learn was carried out on this TF-IDF model
        - Creates matrices showing individual words' associations with topics, and individual songs' associations with topics
    - Cosine similarity metric from Scikit Learn used to identify lyrically similar songs, also based on TF-IDF

## 3. Instructions for Running Program
__________________________________________
Fork the repository, and then clone it to your local machine. You should now have a repository named "lyrical-exercise" on your computer. <br>

All necessary data files but two are in the repository, in the "data" folder. Follow the steps below to get the remaining two files (You only need to do this once!) <br>

### How to get the remaining two files: 

1. __mxm_779k_matches.txt__:
    - Click on this link: http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/mxm_779k_matches.txt.zip
    - Name the file ```mxm_779k_matches.txt```, and insert the file into the "data" folder in the repository
    - Remove the header with the # symbols at the start of lines, just leaving data in text file
2. __mxm_dataset_FULL.txt__:
    - Click on this link for the train set: http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/mxm_dataset_train.txt.zip
    - Click on this link for the test set: http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/mxm_dataset_test.txt.zip
    - Remove headers with the # symbols at the start of lines, just leaving data in the text file, and copy paste the test set into the train set text file, appending it directly to the bottom of the file
    - Rename the combined file to ```mxm_dataset_FULL.txt```, and insert the file into the "data" folder in the repository
    
### Choose Preferred Genre:
Go to Line 22 of "Process.py," and change the global variable FILTER_GENRE to equal any of the following genres (default: Rap): 
- Blues
- Country
- Electronic
- Folk
- Jazz
- Latin
- Metal
- New Age
- Pop
- Punk
- Rap
- Reggae
- RnB
- Rock
- World

### Run the program:
- Enter the lyrical-exercise folder from the terminal (using 'cd' command), and enter the command ```python lyrical-exercise.py```
- Follow along with the prompts to choose preferred genre, target song, and more

### Check the Results:
- __Topic Modeling__: After running the program, check the "results" folder for a file named "topic_words.txt", which has all of the lyrics organized by latent topic
- __Similar Song Recommender__: This runs in the terminal until you exit
