import json
from google.cloud import storage
import pandas as pd
import numpy as np
import os

# NLP Packages
import spacy
nlp  = spacy.load('en_core_web_sm')

from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.utils import simple_preprocess

# to break articles up into sentences
from nltk import tokenize

# for counting frequency of words
from collections import defaultdict

import spacy
nlp  = spacy.load('en_core_web_sm')



def return_suggested_articles(request):
    """
    returns suggested articles based on topic of one currently being viewed

    Parameters
    ----------
    request : request (flask.Request): The request object..

    Returns
    -------
    articles to read.

    """    
    
    # get the requested json for the webpage
    request_json = request.get_json(silent=True)
    
    # get the headline and article
    headline = request_json['headline']
    article = request_json['article']
    
    # make into one text file
    combined_article = headline+article
    
    # download stopwords list
    download_from_bucket('debiaser_data', 'sw1k.csv', '/tmp/stop_words_1k.csv')
    
    # load stop words into pandas and then into list
    stop_words = pd.read_csv('/tmp/stop_words_1k.csv')
    stop_words = stop_words['term']
    stop_words = [word for word in stop_words]
    
    # adding some custom words
    stop_words.append('said')
    stop_words.append('youre')
    # stop_words.append('mph')
    
    
    # download all_sides_media list
    download_from_bucket('debiaser_data','allsides_final_plus_others_with_domains.csv.csv', '/tmp/all_sides.csv')
    
    # load domain names into dataframe and then get only names and
    all_sides = pd.read_csv('/tmp/all_sides.csv')
    all_sides_names = all_sides['name']
    all_sides_domains = all_sides['domain']
    all_sides_names_domains = pd.concat([all_sides_names,all_sides_domains],axis=1)
    
    # clean up workspace for memory
    os.remove('/tmp/all_sides.csv')
    os.remove('/tmp/stop_words_1k.csv')
    
    # replace weird apostrophes
    combined_article = combined_article.replace("`","'")
    combined_article = combined_article.replace("’","'")
    combined_article = combined_article.replace("'","'")
    
    # replace long dashes with short dashes
    combined_article = combined_article.replace("—","-")
    
    # replace short dashes with spaces
    combined_article = combined_article.replace("-"," ")
    
    # break up into sentences
    combined_article = tokenize.sent_tokenize(combined_article)
    
    # process article
    article_processed = process_all_articles(combined_article)
    
    # remove stopwords
    article_processed = remove_stopwords(article_processed,stop_words)
    
    # floor for the frequency of words to remove
    word_frequency_threshold = 1
    
    # get corpus, dictionary, bag of words
    processed_corpus, processed_dictionary, bow_corpus = get_simple_corpus_dictionary_bow(article_processed,
                                                                                          word_frequency_threshold)
    
    # set the number of topics to generate (5 seems to work pretty well)
    num_lda_topics = 5
    
    # set the number of passes
    n_passes = 20
    
    # generate the LDA model
    lda = LdaModel(corpus = bow_corpus,
                   num_topics = num_lda_topics,
                   id2word = processed_dictionary,
                   passes = n_passes)
    
    # get the topics from the lda model
    lda_topics = lda.show_topics(formatted = False)
    
    # get the unique topic words
    # get dictionary of each topic word and their associated probabilities per topic
    # get dictionary of each topic word and their mean probability
    # get dictionary of each topic word and their std dev probability
    # get dictionary of each topic word and the frequency with which the words show up
    topics, topics_probs_dict, topics_mean_probs_dict, topics_std_probs_dict, topics_frequency_dict = get_topic_words_mean_std_prob_frequency(lda_topics)
    
    # get the topic mean probs and frequencies, sorted
    topics_means, means_sorted, std_sorted, topics_freq, freq_sorted = sort_topics_mean_frequency(topics,
                                             topics_mean_probs_dict,
                                             topics_std_probs_dict,
                                             topics_frequency_dict)
    
    return json.dumps(combined_article)

def process_text(article_text):
    """
    processes individual document text by removing stop words,
    making all lower case,
    and removing punctuation.
    

    Parameters
    ----------
    article_text : document text that you want to preprocess.

    -------
    processed text for document.

    """
    
    # run gensim preprocess
    article_text = simple_preprocess(article_text, deacc=True)
    
    return article_text


def process_all_articles(documents):
    """
    runs process_texts function on each document in documents.

    Parameters
    ----------
    documents : list[strs]
        a list of documents, where each document is a string

    Returns
    -------
    documents_processed : list of processed documents.

    """
    
    # list for processed documents
    documents_processed = []
    
    # loop through documents
    for document in documents:
        
        # process documents
        document_processed = process_text(document)
        
        # append to list of processed documents
        documents_processed.append(document_processed)
        
    return documents_processed
    
def download_from_bucket(bucket_name,source_data_name,destination_file_name):
    """
    

    Parameters
    ----------
    bucket_name : name of the bucket on google cloud.
    source_data_name : name of the data set.
    destination_file_name : where the data will be stored.

    Returns
    -------
    None.

    """
    
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    data_set = bucket.blob(source_data_name)
    
    print(f'Blob data {source_data_name} downloaded to {destination_file_name}')
    
    
def remove_stopwords(documents, stop_words):
    """
    removes stopwords from corpus
    

    Parameters
    ----------
    documents : list of documents, where each doc is string.

    Returns
    -------
    None.

    """
    
    documents_processed = []
    
    for document in documents:
        
        document_processed = [word for word in document if word not in stop_words]
        
        documents_processed.append(document_processed)
        
    return documents_processed
    
    
def get_simple_corpus_dictionary_bow(texts,word_frequency_threshold):
    """fxn returns corpus, processed dict, bag of words"""
    
    # Count word frequencies
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    # Only keep words that appear more than set frequency, to produce the corpus
    processed_corpus = [[token for token in text if frequency[token] > word_frequency_threshold] for text in texts]
    
    # generate a dictionary via gensim
    processed_dictionary = Dictionary(processed_corpus)
    
    # generate bag of words of the corpus
    bow_corpus = [processed_dictionary.doc2bow(text) for text in processed_corpus]
    
    return processed_corpus, processed_dictionary, bow_corpus

def get_topic_words_mean_std_prob_frequency(lda_topics):
    """
    

    Parameters
    ----------
    lda_topics : lda topics from lda, formatted = False, as tuples.

    Returns
    -------
    topics : unique topic words
    topics_mean_probs_dict : the mean probability of each topic word
    topics_std_probs_dict : the std dev prb of each topic word
    topics_frequency_dict : the frequency each topic word shows up in a topic.

    """
    
    # dictionary for topics and the probabilities associated with them
    topics_probs_dict = {}
    
    # list of unique topic names
    topics = []
    
    # loop through each list of generated topics
    for topic in lda_topics:
        
        # get the list of tuples of topic words and the associated probability
        topic_words = topic[1]
        
        # loop through topic words and probabilities
        for topic_word, prob in topic_words:
            
            # if the word isn't already in the list of topics, add it to list of topics
            if topic_word not in topics: 
                topics.append(topic_word)
                
            # if the word is not a key in the dictionary of topics to probabilities, add it to dictionary
            if topic_word not in topics_probs_dict.keys():
                
                topics_probs_dict[topic_word] = np.array([prob])
            
            # if the word is a key in the dictionary of topics to probabilities, append probability
            else:
                topics_probs_dict[topic_word] = np.append(topics_probs_dict[topic_word],[prob])
        
    # dictionary for topic probability means
    topics_mean_probs_dict = {}
    
    # dictionary for topic probability std devs
    topics_std_probs_dict = {}
    
    # dictionary for topic frequency
    topics_frequency_dict = {}

    # loop through topics and probabilities
    for topic, prob in topics_probs_dict.items():
        
        # update dictionary for mean probability
        topics_mean_probs_dict[topic] = np.mean(prob)
        
        # update dictionary for std dev probability
        topics_std_probs_dict[topic] = np.std(prob)
        
        # update dictionary for topic frequency
        topics_frequency_dict[topic] =  prob.size    

    return topics, topics_probs_dict, topics_mean_probs_dict, topics_std_probs_dict, topics_frequency_dict


def sort_topics_mean_frequency(topics,topics_mean_probs_dict,topics_std_probs_dict,topics_frequency_dict):
    """fxn returns topics sorted by their mean probability and frequency. also returns std dev of prob"""
    
    # empty dict for topics
    x_topics = []
    
    # to store mean probs
    y_means = np.zeros((len(topics_mean_probs_dict)))
    
    # to store std dev probs
    y_std = np.zeros((len(topics_std_probs_dict)))
    
    # to store frequency of topic
    y_frequencies = np.zeros((len(topics_frequency_dict)))

    # measure mean probs
    counter = -1
    for topics, mean_prob in topics_mean_probs_dict.items():
        counter += 1
        x_topics.append(topics)
        y_means[counter] = mean_prob

    # measure std dev probs
    counter2 = -1
    for topics, frequency in topics_frequency_dict.items():
        counter2 += 1
        y_frequencies[counter2] = frequency

    # measure frequency topic shows up
    counter3 = -1
    for topics, std in topics_std_probs_dict.items():
        counter3 += 1
        y_std[counter3] = std

    # sort by mean and frequency

    zipped_mean = zip(y_means, x_topics)
    sorted_zipped_mean = sorted(zipped_mean)
    y_means_sorted = [element1 for element1,element2 in sorted_zipped_mean]
    y_means_sorted = y_means_sorted[::-1]
    x_topics_means = [element2 for element1,element2 in sorted_zipped_mean]
    x_topics_means = x_topics_means[::-1]

    zipped_mean_std = zip(y_means,y_std)
    sorted_zipped_mean_std = sorted(zipped_mean_std)
    y_std_sorted = [element2 for element1,element2 in sorted_zipped_mean_std]
    y_std_sorted = y_std_sorted[::-1]

    zipped_freq = zip(y_frequencies,x_topics)
    sorted_zipped_freq = sorted(zipped_freq)
    y_freq_sorted = [element1 for (element1,element2) in sorted_zipped_freq]
    y_freq_sorted = y_freq_sorted[::-1]
    x_topics_freq = [element2 for element1,element2 in sorted_zipped_freq]
    x_topics_freq = x_topics_freq[::-1]
    
    return x_topics_means, y_means_sorted, y_std_sorted, x_topics_freq, y_freq_sorted


# def read_article(file_json):
#     """
#     receives text via read_article() function

#     initial template following 'building a serverless chrome extention',
#     TDS article by Bilal Tahir, May 29, 2019
#     Parameters
#     ----------
#     file_json : json file for given webpage.

#     Returns
#     -------
#     None.

#     """
    
#     article = ''
#     filedata = json.dumps(file_json)
    
#     if len(filedata) < 1000000:
#         article = filedata
    
#     return article
