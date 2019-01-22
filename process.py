# File name: process.py
# Author: Eric Nadasi
# Date Created: 12/14/18
# Date Modified: 1/10/19
# Description: This file contains the NLP and other processing work done on the dataset
# that is needed to clean the data and prepare it to be analyzed in the modeling file.


import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from ast import literal_eval
import sys


# # Only need to download these once and they stay on comp
# # Can comment out if they are not already downloaded
# nltk.download('stopwords')

# Future Implementation: global variable for which genre we want to look at
FILTER_GENRE = 'Rock'


def load_genre_df():
    """
    This function loads the text files containing Track IDs and genre labels into a pandas dataframe.

    Input: None
    Output: df_genre_final - pandas dataframe - has columns tagtraum_trackId,
    majority_genre, and minority_genre
    """

    cols = ['tagtraum_trackId', 'majority_genre', 'minority_genre']
    tt_genre = pd.read_csv('data/msd_tagtraum_cd2.txt', names=cols, sep='\t')

    magd_genre = pd.read_csv('data/topMAGD-genreAssignment.txt', names=cols, sep='\t')

    df_merged = tt_genre.append(magd_genre)

    # filter on genre. Currently filtering on rap but this can be tweaked if we want
    # using the global variable. Might see some potential new stopwords (based on topic modeling)
    # when filtering for genres other than rap
    df_majority = df_merged.loc[df_merged['majority_genre'] == FILTER_GENRE]
    df_minority = df_merged.loc[df_merged['minority_genre'] == FILTER_GENRE]

    # combine the dataframes for majority and minority rap genres
    df_genre = df_majority.append(df_minority)

    # get rid of track ID duplicates resulting from merging of different genre datasets
    df_genre_final = df_genre.drop_duplicates(subset='tagtraum_trackId', keep='first')

    return df_genre_final


def load_lyrics_df():
    """
    Load the MusiXMatch dataset into a pandas dataframe with 3 columns (track_id, mxm_track_id, bag-of-words)
    Input: None
    Output: mxm_lyrics - pandas dataframe - dataframe with columns track_id, mxm_track_id, bag-of-words
    """

    # First, we need to rewrite the text file to use a different delimiter for the columns
    # to avoid the confusion with the commas in the bag of words

    # open the original file, open a new file with the ending "FINAL". Then for each line in original file,
    # split it into 3 items (2 splits), and then write a new line to the FINAL file where we join the items of
    # the split with semicolons (';') instead of the commas
    with open('data/mxm_dataset_FULL.txt') as mxm_file:
        with open('data/mxm_dataset_FINAL.txt', 'w+') as mxm_file_final:
            for line in mxm_file:
                line_lst = line.split(',', 2)
                mxm_file_final.write(';'.join(line_lst))

    # read in the new semicolon delimited file into a pandas dataframe
    cols = ['track_id', 'mxm_track_id', 'bag-of-words']
    mxm_lyrics = pd.read_csv('data/mxm_dataset_FINAL.txt', names=cols, sep=';')

    return mxm_lyrics


def load_artist_title_df():
    """
    Load artist and title data about tracks into a dataframe.
    Input: None
    Output: artist_title_df2 - pandas dataframe - dataframe with columns 'track_id',
    'artist_name', 'title', and 'mxm_title'
    """

    # bring in the artist and track name data into a new DF
    cols = ['track_id', 'artist_name', 'title', 'mxm_track_id', 'mxm_artist_name', 'mxm_title']
    artist_title_df = pd.read_csv('data/mxm_779k_matches.txt', names=cols, sep='<SEP>', engine='python')

    # drop the extra columns that we don't need. Keeping mxm_title because it can be different from
    # 'title' sometimes
    artist_title_df2 = artist_title_df.drop(labels=['mxm_track_id', 'mxm_artist_name'], axis=1)

    return artist_title_df2


def load_corpus():
    """
    Load the 5,000 word stem corpus given in a text file by the MSD website into a list
    Input: None
    Output: corpus - list - list of stemmed representations of top 5,000 words found in lyrics corpus
    """

    # load the 5,000 most common words corpus into a list
    corpus_file = open("data/mxm_corpus.txt", "r")
    corpus = corpus_file.read().split(',')

    # first item of the list is weird for some reason, change it to 'i' which we know it is.
    # last item has a '}' attached to it, shave that off with -1 index. rest looks good
    corpus[0] = 'i'
    corpus[4999] = corpus[4999][:-1]
    return corpus


def reverse_map_corpus(stem_corpus):
    """
    Using the file containing reverse mapping from stems->full words, create a new corpus with the full words
    that are represented by the bag-of-words indices.
    Input: stem_corpus - list - list of length 5,000 with the top 5,000 most common words used in all songs
    Output: corpus_fullwords
    """

    cols = ['stem', 'word']

    # Load in reverse mapping file into a pandas dataframe
    stem_map = pd.read_csv('data/mxm_reverse_mapping.txt', sep='<SEP>', engine='python', header=None, names=cols)

    # iterate thru corpus changing stems to real words
    corpus_fullwords = stem_corpus.copy()
    for i, stem_1 in enumerate(corpus_fullwords):
        if stem_1 in stem_map['stem'].values:
            corpus_fullwords[i] = stem_map['word'].loc[stem_map['stem'] == stem_1].values[0]

    # check that our fullword corpus was actually changed
    # Comparing 'colleg' and 'college' (given it was changed)
    if (corpus_fullwords[4998] != stem_corpus[4998]):
        return corpus_fullwords
    else:
        print('Corpus was not reverse mapped correctly. Check process.py file, reverse_map_corpus function.')
        sys.exit()


def load_stopwords():
    """
    Load english stopwords from natural language toolkit (NLTK) module, and add own stopwords based
    on EDA done on corpus and initial results from topic modeling
    Input: None
    Output: stopwords_ - set - set of words that are stopwords for our lyrics. stopwords will be removed from
    lyrical representation for all songs
    """

    stopwords_ = set(stopwords.words('english'))

    # first group of stopwords added to set based on high frequency words from
    # full_word_list.txt that were not already in stopwords
    new_stopwords = ['oh', 'la', 'got', 'let', 'ca', 'en', 'un', 'el', 'tu', 'yeah']

    stopwords_.update(new_stopwords)

    # adding new words to stopword list based on the results of the first topic modeling we ran
    # on BOWs format with NMF 1/5/19
    new_stopwords2 = ['nigga', 'shit', 'fuck', 'yo', '\'em', 'niggaz', '\'cause', 'y\'all', 'fuckin\'', 'ai', 'motherfucker', '\'bout', '\'gon', 'gotta', 'wit', 'tha', 'ya', 'da', 'uh']

    # adding more to stopwords based on NMF using TFIDF vectors 1/9/19
    new_stopwords3 = ['u', 'n', '2', 'im', 'yea', 'dat', 'ur']

    stopwords_.update(new_stopwords2)
    stopwords_.update(new_stopwords3)

    return stopwords_


def calc_stop_indices(corpus_fullwords, stopwords_):
    """
    Calculate the indices of the bag-of-words representations that represent stopwords, so that
    we can filter those indices out when performing our topic modeling and similarity NLP analysis.
    Input: corpus_fullwords - list - list of 5,000 words that define the corpus of all song lyrics
           stopwords_ - set - set of english stopwords to weed out of our lyrics representations
    Output: stop_indxs - set - set of indices that represent stopwords in the corpus list
    """

    # get the indices of the stopwords within the full word corpus to make a stop indices list
    stop_indxs = set()

    for i in range(len(corpus_fullwords)):
        if corpus_fullwords[i] in stopwords_:
            stop_indxs.add(i + 1)

    return stop_indxs


def merge_dataframes(genre_df, lyrics_df, artist_title_df):
    """
    Merge the three dataframes with the
    Input: genre_df - pandas dataframe - has columns tagtraum_trackId, majority_genre, and minority_genre
           lyrics_df - pandas dataframe - has columns track_id, mxm_track_id, bag-of-words
           artist_title_df - pandas dataframe - has columns 'track_id', 'artist_name', 'title', and 'mxm_title'
    Output: all_data_df - pandas dataframe - dataframe with all of the relevant data needed in it. Has genre
    labels, lyrics in bag-of-words form, and artist and song title for all tracks
    """

    genre_lyrics_df = pd.merge(genre_df, lyrics_df, how='left', left_on='tagtraum_trackId', right_on='track_id')
    all_data_df = pd.merge(genre_lyrics_df, artist_title_df, how='left', left_on='tagtraum_trackId', right_on='track_id')

    return all_data_df


def remove_stopwords(bow_dict, stop_indxs):
    """
    Remove stopwords from the bag-of-words dictionary entered as a parameter, based on the stop indices
    entered as the second parameter
    Input: bow_dict - dictionary - bag-of-words representation of lyrics in a track
           stop_indxs - set - set of stopword indices
    Output: new_dict - bag-of-words lyrics representation with stopwords removed
    """

    new_dict = dict()
    for key in bow_dict.keys():
        if key not in stop_indxs:
            new_dict[key] = bow_dict[key]
    return new_dict


def bow_cleanup(all_data_df, stop_indxs):
    """
    Convert column with bag-of-words strings to dictionary types, and remove stopwords!
    Input: all_data_df - dataframe - dataframe with genre, lyric, artist/title info
    Output: new_df - dataframe -
    """
    new_df = all_data_df.copy()
    new_df['bag-of-words-dict'] = new_df['bag-of-words'].apply(lambda x: '{' + str(x) + '}')

    new_df['bow-dict-type'] = new_df['bag-of-words-dict'].apply(lambda x: literal_eval(x) if str(x) != '{nan}' else None)

    new_df['bow-dict-filtered'] = new_df['bow-dict-type'].apply(lambda x: remove_stopwords(x, stop_indxs) if str(type(x)) == '<class \'dict\'>' else x)

    new_df = new_df.drop(labels=['bag-of-words-dict', 'bow-dict-type'], axis=1)

    return new_df
