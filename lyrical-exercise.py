# File name: lyrical-exercise.py
# Author: Eric Nadasi
# Date Created: 12/14/18
# Date Modified: 1/10/19'
# Description: This file contains the modeling work, done for
# the similarity metric and topic modeling of the project.

from process import *
from gensim import models
import scipy.sparse as sps
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF

# global variable for number of topics that we should have in topic modeling
NUM_TOPICS = 12

# global variable for number of similar songs to show user with cosine similarity
NUM_SIMILAR_SONGS = 10


def create_tfidf_col(bow_na_df):
    '''
    Create the tfidf column in the dataframe using gensim from the BOWs representations
    Input: bow_na_df - dataframe - dataframe with all info and BOWs in dict format
    Output: bow_tfidf_df - dataframe - dataframe with everything from before + tfidf vectors column
    '''

    bow_df = bow_na_df.dropna(subset=['bow-dict-filtered'])
    bow_corpus = bow_df['bow-dict-filtered'].tolist()

    for i in range(len(bow_corpus)):
        bow_corpus[i] = list(bow_corpus[i].items())

    # create TFIDF vectors using gensim tfidf model
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    # insert TFIDF into dataframe
    bow_tfidf_df = bow_df.copy()
    bow_tfidf_df['tfidf'] = list(corpus_tfidf)

    return bow_tfidf_df


def create_lyric_matrix(tfidf_list):
    '''
    Create the matrix of shape (n_samples, n_features) with tfidf vectors, for the full corpus of songs
    that have lyrics data
    Input: tfidf_list - list of lists - list containing n_samples (number of songs) number of tfidf vectors
    (which are in list form)
    Output: lyric_matrix - Compressed sparse row matrix - matrix containing lyric data for all songs,
    in a form that the model functions can handle
    '''

    tfidf_mat = sps.dok_matrix((len(tfidf_list), 5000), dtype=np.float64)

    for i, row in enumerate(tfidf_list):
        for k, v in row:
            tfidf_mat[i, k - 1] = v

    tfidf_mat = tfidf_mat.tocsr()

    return tfidf_mat


def topic_model_NMF(lyric_matrix, corpus, songs):
    '''
    Input: lyric_matrix - matrix - matrix in form that is valid for sklearn NMF model to fit
           corpus - list - list of words in the corpus
           songs - dataframe column - column full of song title labels for tm_W matrix
    Output: tm_H - dataframe - dataframe of shape (n_topics, n_features) with rows for each latent topic,
    and column labels representing all the words in the corpus. Values are word's relative importance in
    classifying a document to that row's topic
            tm_W - dataframe - dataframe of shape (n_samples, n_topics) with rows for each song, and
    column labels representing each topic in the corpus. Values are song's associations with that column's
    topic
    '''

    nmf_model = NMF(n_components=NUM_TOPICS, random_state=5, init='random')
    tm_W = nmf_model.fit_transform(lyric_matrix)
    tm_H = nmf_model.components_

    tm_W = pd.DataFrame(tm_W, index=songs)
    tm_H = pd.DataFrame(tm_H, columns=corpus)

    return tm_H, tm_W


def cos_sim_vector(tfidf_list):
    '''
    Create a vector that will be able to be fed into the cosine_similarity function of sklearn
    Input: tfidf_list - list - tfidf vector in list form
    Output: tf_vec - numpy array - tfidf vector in array form with column representing word index
    in corpus
    '''

    tf_vec = np.zeros((1, 5000))
    for k, v in tfidf_list:
        tf_vec[0, k - 1] = v
    return tf_vec


def create_song_list(tfidf_df):
        '''
        Create a text file with all available songs to choose for the similar_songs function
        Input: tfidf_df - dataframe - dataframe with columns containing artist, song title, and
               tfidf vectors at the least
        Output: None (text file song_list.txt is created though)

        '''

        song_list = list()
        for artist, title in zip(tfidf_df['artist_name'].tolist(), tfidf_df['title'].tolist()):
            song_list.append(artist + ' - ' + title)

        with open('song_list.txt', 'w+') as song_file:
            for song in sorted(song_list):
                song_file.write(str(song) + '\n')


def similar_songs(title, df, num_songs):
    '''
    Find songs that are lyrically similar to the song entered as a parameter based on cosine_similarity
    Input: title - string - title of song in string form
           df - dataframe - dataframe containing song artist, title, and tfidf vector
           num_songs - int - number of similar songs we want the function to return to user
    '''

    song_tfidf = df['tfidf'].loc[df['title'] == title].tolist()[0]
    df2 = df.loc[df['title'] != title]
    sims = pd.DataFrame(df2[['artist_name', 'title']])
    sims['cos_similarity'] = df2['tfidf'].map(lambda x: cosine_similarity(cos_sim_vector(song_tfidf), cos_sim_vector(x))[0][0])
    sims_sorted = sims.sort_values(by='cos_similarity', axis=0, ascending=False)
    return sims_sorted.iloc[:num_songs]


if __name__ == '__main__':

    # do all the processing of the data needed to set up our dataframes and files
    # for both of our NLP analysis methods within the main method
    genre_df = load_genre_df()
    lyrics_df = load_lyrics_df()
    artist_title_df = load_artist_title_df()

    stem_corpus = load_corpus()
    fullword_corpus = reverse_map_corpus(stem_corpus)

    stopwords_ = load_stopwords()
    stop_indxs = calc_stop_indices(fullword_corpus, stopwords_)

    # merge all dataframes on the track ID key
    all_data_df = merge_dataframes(genre_df, lyrics_df, artist_title_df)

    bow_dict_df = bow_cleanup(all_data_df, stop_indxs)

    # run gensim TFIDF model on the BOWs representation
    tfidf_df = create_tfidf_col(bow_dict_df)

    tfidf_list = tfidf_df['tfidf'].tolist()
    tfidf_matrix = create_lyric_matrix(tfidf_list)

    # get rid of unnecessary columns from DataFrame
    tfidf_df = pd.DataFrame(tfidf_df[['artist_name', 'title', 'tfidf']])

    # topic modeling section below
    topic_model_bool = input('Run topic modeling? Yes/No: ')
    if topic_model_bool.lower() == 'yes':
        H, W = topic_model_NMF(tfidf_matrix, fullword_corpus, tfidf_df['title'])

        # display the top 40 words associated with each topic in a text file
        with open('topic_words.txt', 'w+') as topic_file:
            for i in range(NUM_TOPICS):
                topic_file.write('Topic ' + str(i) + ':\n\n')
                topic_file.write(H.iloc[i].sort_values(ascending=False)[:40].to_csv(sep=' '))
                topic_file.write('\n')
        print('Topic Modeling Done.')

    # cosine similarity section below
    create_song_list(tfidf_df)
    user_choice = ' '

    # print lyrically similar songs to terminal
    user_choice = input('Enter song title to find lyrically similar songs.\
    Full song list in song_list.txt within project directory. Enter \'exit\' to quit: ').strip()

    while user_choice.lower() != 'exit':
        print('Song choice: ' + user_choice)
        artist = tfidf_df['artist_name'].loc[tfidf_df['title'] == user_choice].tolist()[0]
        print('Song Artist: ' + artist)
        print(similar_songs(user_choice, tfidf_df, NUM_SIMILAR_SONGS))
        user_choice = input('Enter song title to find lyrically similar songs. \
        Full song list in song_list.txt within project directory. Enter \'exit\' to quit: ').strip()
