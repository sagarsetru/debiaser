#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:29:29 2020

@author: sagarsetru
"""
import json

# from google.cloud import storage

import pandas as pd
import numpy as np
import os
import time

# NLP Packages

from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.utils import simple_preprocess

# to break articles up into sentences
from nltk import tokenize
from nltk.corpus import stopwords

# for counting frequency of words
from collections import defaultdict

# for bi tri quad grams via word 2 vec
import gensim.models.keyedvectors as word2vec

# import spacy
# nlp  = spacy.load('en_core_web_sm')
nlp = []

from googlesearch import search

import requests

from bs4 import BeautifulSoup

import time

import pickle

# import urllib3
# import urllib

# import streamlit as st

def return_suggested_articles2(url,do_unique_search_words=1,use_pre_trained_model=0,do_ngrams=0,do_sentences=1,n_search_words=5,num_lda_topics=1,n_passes=10,print_article=0):
    """
    returns suggested articles based on topic of one currently being viewed

    Parameters
    ----------
    request : request (flask.Request): The request object..

    Returns
    -------
    articles to read.

    """    
    
    # use_bucket = 0
    
    # if trying to insure unique words
    # do_unique_search_words = 1
    
    # # if using LDA model pre trained on article corpus
    # use_pre_trained_model = 0
    
    # do_sentences = 0
    
    # # only for use when using one topic; this is number of words from that topic
    # # that will be used in search
    # n_search_words = 5
    
    # num_lda_topics = 1
    
    # n_passes = 10
    
    # print_article = 0
    
    # get html content of url
    page = requests.get(url)
    coverpage = page.content
    
    # create soup object
    soup = BeautifulSoup(coverpage, 'html.parser')
    
    # get title
    headline = soup.find('h1').get_text()
    # print(headline)
    # print(' ')
    
    # get text from all <p> tags
    p_tags = soup.find_all('p')
    
    # get text from each p tag and strip whitespace
    p_tags_text = [tag.get_text().strip() for tag in p_tags]
    
    # filter out sentences without periods
    p_tags_text = [sentence for sentence in p_tags_text if '.' in sentence]
    
    # convert all p_tags_text to single article text string
    p_tags_text_1string = ''

    for p_tag_text in p_tags_text:
        p_tags_text_1string += p_tag_text

    # if print_ptags:
    #     print(p_tags)
    #     print(' ')
    #     print(p_tags_text_1string)
    
    combined_article = headline+'. '+p_tags_text_1string
    if print_article:
        print(combined_article)
    
    # # get the requested json for the webpage
    # request_json = request.get_json(silent=True)
    
    # # get the headline and article
    # headline = request_json['headline']
    # article = request_json['article']
    
    # make into one text file
    # combined_article = headline+article
    
    # download stopwords list
    # if use_bucket:
    #     download_from_bucket('debiaser_data', 'sw1k.csv', '/tmp/stop_words_1k.csv')
    
    #     # load stop words into pandas and then into list
    #     stop_words = pd.read_csv('/tmp/stop_words_1k.csv')
    
    # else:
        
    stop_words = pd.read_csv('/Users/sagarsetru/Documents/post PhD positions search/insightDataScience/project/debiaser/stop_words_db/news-stopwords-master/sw1k.csv')
    
    if do_ngrams:
        bigram_mod_file = '/Users/sagarsetru/Documents/post PhD positions search/insightDataScience/project/debiaser/all_the_news/bigram_mod.pkl'
        with open(bigram_mod_file, 'rb') as pickle_file:
            bigram_mod = pickle.load(pickle_file)
            
        trigram_mod_file = '/Users/sagarsetru/Documents/post PhD positions search/insightDataScience/project/debiaser/all_the_news/trigram_mod.pkl'
        with open(trigram_mod_file, 'rb') as pickle_file:
            trigram_mod = pickle.load(pickle_file)
        
        quadgram_mod_file = '/Users/sagarsetru/Documents/post PhD positions search/insightDataScience/project/debiaser/all_the_news/quadgram_mod.pkl'
        with open(quadgram_mod_file, 'rb') as pickle_file:
            quadgram_mod = pickle.load(pickle_file)
    
        # use nltk stopwords
        # stop_words = list(stopwords.words('english'))
        
    stop_words = stop_words['term']
    stop_words = [word for word in stop_words]
    
    # add custom stop words
    stop_words.append('said')
    stop_words.append('youre')
    stop_words.append('mph')
    stop_words.append('inc')
    stop_words.append('cov')
    stop_words.append('jr')
    stop_words.append('dr')
    stop_words.append('ads')
    stop_words.append('cookies')
    stop_words.append('factset')
    
    
    # # adding some custom words
    # stop_words.append('said')
    # stop_words.append('youre')
    # stop_words.append('mph')
    
    
    # download all_sides_media list
    # if use_bucket:
    #     download_from_bucket('debiaser_data','allsides_final_plus_others_with_domains.csv.csv', '/tmp/all_sides.csv')
    
    #     # load domain names into dataframe and then get only names and
    #     all_sides = pd.read_csv('/tmp/all_sides.csv')
    
    # else:
    all_sides = pd.read_csv('/Users/sagarsetru/Documents/post PhD positions search/insightDataScience/project/debiaser/all_sides_media_data/allsides_final_plus_others_with_domains.csv')
    
    all_sides_names = all_sides['name']
    all_sides_domains = all_sides['domain']
    # all_sides_names_domains = pd.concat([all_sides_names,all_sides_domains],axis=1)
    
    # clean up workspace for memory
    # if use_bucket:
    #     os.remove('/tmp/all_sides.csv')
    #     os.remove('/tmp/stop_words_1k.csv')
    
    # get dictionary of entities in article
    # entity_dict = entity_recognizer(combined_article,nlp)
    
    
    # break up into sentences
    if do_sentences:
        combined_article = tokenize.sent_tokenize(combined_article)
    else:
        combined_article = [combined_article]
    
    if print_article:
        print('TOKENIZED TO SENTENCES')
        print(combined_article)
        
    # start = time.process_time()
    # model = word2vec.KeyedVectors.load_word2vec_format('/Users/sagarsetru/Documents/post PhD positions search/insightDataScience/project/debiaser/GoogleNews-vectors-negative300.bin.gz',binary=True)
    # print('TIME FOR LOADING WORD2VEC MODEL')
    # print(time.process_time() - start)
    # print(combined_article)
    # process article
    article_processed = process_all_articles(combined_article,nlp)
    
    # remove stopwords
    article_processed = remove_stopwords(article_processed,stop_words)
    if print_article:
        print('AFTER STOPWORDS')
        print(article_processed)
    
    
    if do_ngrams:
        
        start = time.process_time()
        article_processed = make_quadgrams(article_processed,bigram_mod,trigram_mod,quadgram_mod)
        print('TIME FOR NGRAMS')
        print(time.process_time() - start)
        print('AFTER NGRAMS')
    
    # print(article_processed)
    
    # floor for the frequency of words to remove
    # word_frequency_threshold = 1
    
    # get corpus, dictionary, bag of words
    # processed_corpus, processed_dictionary, bow_corpus = get_simple_corpus_dictionary_bow(article_processed,
    #                                                                                       word_frequency_threshold)
    
    start = time.process_time()
    
    if use_pre_trained_model:
        
        # load dictionary used to train model on EC2
        id2word_file = '/Users/sagarsetru/Documents/post PhD positions search/insightDataScience/project/debiaser/all_the_news/id2word.pkl'
        with open(id2word_file, 'rb') as pickle_file:
            processed_dictionary = pickle.load(pickle_file)

        
        # get terms in article
        processed_corpus = [[token for token in text] for text in article_processed]
        
        # generate bag of words of the corpus
        bow_corpus = [processed_dictionary.doc2bow(text) for text in processed_corpus]
            
    else:
        
        processed_corpus, processed_dictionary, bow_corpus = get_simple_corpus_dictionary_bow(article_processed)
    
    print('TIME FOR BOW VECTOR')
    print(time.process_time() - start)
    
    
    # for bow in bow_corpus:
    #     print(bow)
    #     print(' ')
        
    if use_pre_trained_model:
        
        print('USING PRE TRAINED LDA MODEL')
        # load the LDA model file
        lda_mod_file = '/Users/sagarsetru/Documents/post PhD positions search/insightDataScience/project/debiaser/all_the_news/lda_model_n_topics_50_n_passes_100_n_docs_chunksize_60000.pkl'
        with open(lda_mod_file, 'rb') as pickle_file:
            lda = pickle.load(pickle_file)
            
        lda_topics = lda[bow_corpus]
        
    else:
        # generate the LDA model
        start = time.process_time()
        lda = LdaModel(corpus = bow_corpus,
                        num_topics = num_lda_topics,
                        id2word = processed_dictionary,
                        passes = n_passes)
        print('TIME FOR LDA MODEL GENERATION')
        print(time.process_time() - start)
        
        # get the topics from the lda model
        lda_topics = lda.show_topics(formatted = False)
    
    
    if use_pre_trained_model:
        return lda_topics, processed_dictionary
    # ALL INTERESTING BUT DEPRECATED FOR NOW
    # WILL FOLLOW SIMPLER APPROACH:
        # Just take top word in each generated topic
        
    # get top words per topic
    lda_top_topic_words_string, lda_top_topic_words_list = get_lda_top_topic_words(lda_topics,num_lda_topics,do_unique_search_words,n_search_words)

    '''
    #  # string is for final search string
    lda_top_topic_words = ''
    
    # list is for checking previous words
    lda_top_topic_words_list = []
    
    
    if num_lda_topics > 1:
    
        if not unique_topic_words:
        
            for topic in lda_topics:
                
                # get the list of topics
                topic_words = topic[1]
                
                lda_top_topic_words += ' '+topic_words[0][0]
            
        else:
            
            # loop through each list of generated topics
            for topic in lda_topics:
                
                # set word added to 0
                word_added = 0
                
                # get the list of topics
                topic_words = topic[1]
                    
                # loop through words in topic
                # add as search term only if they aren't already search terms
                for i in range(len(topic_words)):
                
                    # if the current word in topic is not in list of search terms
                    if topic_words[i][0] not in lda_top_topic_words_list:
                        
                        # add this word to list of topic/search terms
                        lda_top_topic_words_list.append(topic_words[i][0])
                        
                        # also update the string for the search terms
                        lda_top_topic_words += ' '+topic_words[i][0]
                        
                        # update word added
                        word_added = 1
                        break
                
                # if no word was added because all supporting words in topic are already
                # search terms, then just add the highest prob/first word in topic
                if word_added == 0:
                    # if every word in this topic is already a search term,
                    # just add the first most probable word and leave the while loop
                    lda_top_topic_words_list.append(topic_words[0][0])
                    lda_top_topic_words += ' '+topic_words[0][0]
                    
    else:
        
        for topic in lda_topics:
                
                # get the list of topic words
                topic_words = topic[1]
                
                # loop through these words and get the top n number
                counter = -1
                for topic_word in topic_words:
                    
                    counter += 1
                    
                    if counter < n_topic_words:
                    
                        lda_top_topic_words += ' '+topic_word[0]
    
        
    '''
    # get list of google queries
    # queries = []
    
    # queries_dict = {}
    
    # # quick manual entry
    # # all_sides_domains = ['nytimes.com','wsj.com']
    # # all_sides_names = ['nyt','wsj']
    
    # for domain in all_sides_domains:
    #     query = 'www.google.com/search?q=site:'+domain+lda_top_topic_words_string
    #     queries.append(query)
        
    #     queries_dict[domain] = query
    
    # queries_dict = {}
    # queries_dict = {'abcnews.go.com': 'www.google.com/search?q=site:abcnews.go.com biden debate joe',
    #             'aljazeera.com': 'www.google.com/search?q=site:aljazeera.com biden debate joe',
    #             'apnews.com': 'www.google.com/search?q=site:apnews.com biden debate joe',
    #             'bbc.com': 'www.google.com/search?q=site:bbc.com biden debate joe',
    #             'bloomberg.com': 'www.google.com/search?q=site:bloomberg.com biden debate joe',
    #             'breitbart.com': 'www.google.com/search?q=site:breitbart.com biden debate joe',
    #             'buzzfeednews.com': 'www.google.com/search?q=site:buzzfeednews.com biden debate joe',
    #             'cbn.com': 'www.google.com/search?q=site:cbn.com biden debate joe',
    #             'cbsnews.com': 'www.google.com/search?q=site:cbsnews.com biden debate joe',
    #             'csmonitor.com': 'www.google.com/search?q=site:csmonitor.com biden debate joe',
    #             'cnn.com': 'www.google.com/search?q=site:cnn.com biden debate joe',
    #             'thedailybeast.com': 'www.google.com/search?q=site:thedailybeast.com biden debate joe',
    #             'democracynow.org': 'www.google.com/search?q=site:democracynow.org biden debate joe',
    #             'factcheck.org': 'www.google.com/search?q=site:factcheck.org biden debate joe',
    #             'forbes.com': 'www.google.com/search?q=site:forbes.com biden debate joe',
    #             'foxnews.com': 'www.google.com/search?q=site:foxnews.com biden debate joe',
    #             'huffpost.com': 'www.google.com/search?q=site:huffpost.com biden debate joe',
    #             'motherjones.com': 'www.google.com/search?q=site:motherjones.com biden debate joe',
    #             'msnbc.com': 'www.google.com/search?q=site:msnbc.com biden debate joe',
    #             'nationalreview.com': 'www.google.com/search?q=site:nationalreview.com biden debate joe',
    #             'nbcnews.com': 'www.google.com/search?q=site:nbcnews.com biden debate joe',
    #             'nypost.com': 'www.google.com/search?q=site:nypost.com biden debate joe',
    #             'nytimes.com': 'www.google.com/search?q=site:nytimes.com biden debate joe',
    #             'newsmax.com': 'www.google.com/search?q=site:newsmax.com biden debate joe',
    #             'npr.org': 'www.google.com/search?q=site:npr.org biden debate joe',
    #             'politico.com': 'www.google.com/search?q=site:politico.com biden debate joe',
    #             'reason.com': 'www.google.com/search?q=site:reason.com biden debate joe',
    #             'reuters.com': 'www.google.com/search?q=site:reuters.com biden debate joe',
    #             'salon.com': 'www.google.com/search?q=site:salon.com biden debate joe',
    #             'spectator.org': 'www.google.com/search?q=site:spectator.org biden debate joe',
    #             'theatlantic.com': 'www.google.com/search?q=site:theatlantic.com biden debate joe',
    #             'theguardian.com': 'www.google.com/search?q=site:theguardian.com biden debate joe',
    #             'thehill.com': 'www.google.com/search?q=site:thehill.com biden debate joe',
    #             'wsj.com': 'www.google.com/search?q=site:wsj.com biden debate joe'}
        
    # print(len(queries))
    # return json.dumps(queries_dict)
    # return lda_topics, processed_dictionary, lda, bow_corpus
    return lda_top_topic_words_list, lda_top_topic_words_string, lda_topics

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
    
    # load instance of the storage client
    storage_client = storage.Client()
    
    # get the bucket from the bucket name
    bucket = storage_client.get_bucket(bucket_name)
    
    # get the data set/bloc
    data_set = bucket.blob(source_data_name)
    
    # save the dataset at the destination file name
    data_set.download_to_filename(destination_file_name)
    
    print(f'Blob data {source_data_name} downloaded to {destination_file_name}')
    
def entity_recognizer(raw_text,nlp):
    """Function that recognizes specific entitites, returns dictionary of them"""

    doc = nlp(raw_text)
    
    entity_dict = {}
    
    word_types = ['DATE', 'PERSON', 'ORG','MONEY','GPE']
    
    for word_type in word_types:
        entity_dict[word_type] = [entity.text for entity in doc.ents if entity.label_ in {word_type}]

    return entity_dict

def process_text(article_text,nlp):
    """
    processes individual document text by removing stop words,
    making all lower case,
    and removing punctuation.
    

    Parameters
    ----------
    article_text : document text that you want to preprocess.
    nlp : nlp lib (eg. nlp  = spacy.load('en_core_web_sm'))

    -------
    processed text for document.

    """
    
    # lemmatize NOTE: NOT IMPLEMENTED YET...
    # article_text = lemmatize(article_text,nlp)
    
    # run gensim preprocess
    article_text = simple_preprocess(article_text, deacc=True)
    
    return article_text


def process_all_articles(documents,nlp):
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
        document_processed = process_text(document,nlp)
        
        # append to list of processed documents
        documents_processed.append(document_processed)
        
    return documents_processed
    
def lemmatize(raw_texts,nlp):

    """Function that lemmatizes text"""

    out_text = []
    
    for text in raw_texts:
        doc = nlp(text)


        #Remove stopwords and lemmatize
        tokens = [token.lemma_ for token in doc]
        
        out_text.append(tokens)
    
    return out_text

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
    
def make_bigrams(texts,bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts,bigram_mod,trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def make_quadgrams(texts,bigram_mod,trigram_mod,quadgram_mod):
    return [quadgram_mod[trigram_mod[bigram_mod[doc]]] for doc in texts]

def get_simple_corpus_dictionary_bow(texts):
    """fxn returns corpus, processed dict, bag of words"""
    
    # Count word frequencies
    # frequency = defaultdict(int)
    # for text in texts:
    #     for token in text:
    #         frequency[token] += 1

    # Only keep words that appear more than set frequency, to produce the corpus
    # processed_corpus = [[token for token in text if frequency[token] > word_frequency_threshold] for text in texts]
    
    processed_corpus = [[token for token in text] for text in texts]

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


def get_lda_top_topic_words(lda_topics,num_topics,do_unique_search_words,n_search_words):
    """
    fxn for algorithm to return the top topic words
    algo varies based on:
    1) whether only unique words are wanted, and
    2) whether there is 1 or more topics
    
                
    if one topic, just take top word in each generated topic
    else, if do_unique_search_words, get top word in each topic that is unique,
          else, just get top word in each topic even if it isn't unique
    
    parameters
    ----------
    lda_topics - topic output from lda model
    num_topic - how many lda topics were generated
    do_unique_search_words - whether to all repeating words as search terms
    n_search_words - how many topic words to use as search terms
    
    outputs
    -------
    list and string of search/topic words
    """
    
    # string is for final search string
    lda_top_topic_words_string = ''

    # list is for checking previous words
    lda_top_topic_words_list = []
    
    # if lda model has only one topic
    if num_topics == 1:

        for topic in lda_topics:

            # get the list of topic words
            topic_words = topic[1]

            # loop through these words and get the top n number
            counter = -1
            for topic_word in topic_words:

                counter += 1

                if counter < n_search_words:
                    
                    word = topic_word[0]
                    
                    # handle cases where already added bigram is in unigram and then update list and string
                    lda_top_topic_words_string, lda_top_topic_words_list, counter = remove_unigram_if_in_ngram(word, lda_top_topic_words_list, lda_top_topic_words_string,do_unique_search_words,counter)

    # if lda model has more than one topic
    elif num_topics > 1:
            
        # this ind is to always get list of tuples of (word, prob)
        fixed_ind1 = 1

        # this ind is to always access the word in the tuple (word, prob)
        fixed_ind2 = 0

        # if you're okay with topic words repeating (often happens..)
        if not do_unique_search_words:

            # loop counter
            counter = 0
            
            # index of word within topic
            ind_use = 0
            
            # index of topic
            topic_use = -1
            
            for i in range(n_search_words):
                counter += 1

                if counter > num_topics:
                    ind_use += 1
                    counter = 1

                if topic_use < num_topics-1:
                    topic_use += 1
                else:
                    topic_use = 0

                # access the appropriate topic word
                word = lda_topics[topic_use][fixed_ind1][ind_use][fixed_ind2]

                # handle cases where already added bigram is in unigram and then update list and string
                lda_top_topic_words_string, lda_top_topic_words_list, counter = remove_unigram_if_in_ngram(word, lda_top_topic_words_list, lda_top_topic_words_string,do_unique_search_words,counter)
                    
        # don't reuse a word if it has already been used
        else:

            counter = 0
            ind_use = 0
            topic_use = -1
            
            # do loop over total words across all topics
            total_topic_words = len(lda_topics)*len(lda_topics[0][fixed_ind1])
            for i in range(total_topic_words):
                counter += 1

                if counter > num_topics:
                    ind_use += 1
                    counter = 1

                if topic_use < num_topics-1:
                    topic_use += 1
                else:
                    topic_use = 0

                # access the appropriate topic word
                word = lda_topics[topic_use][fixed_ind1][ind_use][fixed_ind2]

                # handle cases where already added bigram is in unigram and then update list and string
                lda_top_topic_words_string, lda_top_topic_words_list, counter = remove_unigram_if_in_ngram(word, lda_top_topic_words_list, lda_top_topic_words_string,do_unique_search_words,counter)
                
                # if the length of the topic words list is at the number of descired topics
                if len(lda_top_topic_words_list) == n_search_words:
                    break

    return lda_top_topic_words_string, lda_top_topic_words_list


def remove_unigram_if_in_ngram(word, lda_top_topic_words_list, lda_top_topic_words_string,do_unique_search_words,counter):
    """
    fxn handles cases when an already added unigram search term is in a bigram

    Parameters
    ----------
    word : string
        input word from topic.
    lda_top_topic_words_list : list
        list of topic words.
    lda_top_topic_words_string : string
        string of topic words.
    do_unique_search_words : 0 or 1
        whether to have unique search words.
    counter : int
        the number of words added

    Returns
    -------
    lda_top_topic_words_string : string
        edited search words string.
    lda_top_topic_words_list : list
        edited search words list.

    """
    
    # remove duplicate unigrams that are in an ngram
    if "_" in word:
        
        # check to make sure this isn't a duplicate bigram
        if word.replace("_"," ") not in lda_top_topic_words_string:
            
            for already_word in lda_top_topic_words_list:
                
                if already_word in word:
    
                    lda_top_topic_words_list.remove(already_word)
                    
                    counter -= 1
        
        lda_top_topic_words_string = ''
        
        # remake the string now that word has been removed
        for already_word in lda_top_topic_words_list:
            lda_top_topic_words_string += ' '+already_word

    
    if do_unique_search_words:
        # check to confirm this new word isn't a unigram in an ngram
        
        print(word)
        if word not in lda_top_topic_words_string:
            
            lda_top_topic_words_string += ' '+word.replace("_"," ")
            lda_top_topic_words_list.append(word.replace("_"," "))
    else:
        
        lda_top_topic_words_string += ' '+word.replace("_"," ")
        lda_top_topic_words_list.append(word.replace("_"," "))
        
    return lda_top_topic_words_string, lda_top_topic_words_list, counter

# url = 'https://www.theguardian.com/sport/2020/sep/25/la-lakers-denver-nuggets-game-4-recap'
# url = 'https://www.nytimes.com/2020/09/25/us/politics/rbg-retirement-obama.html'
url = 'https://www.nytimes.com/2020/09/27/us/politics/trump-biden-debate-expectations.html?action=click&module=Top%20Stories&pgtype=Homepage'
url = 'https://www.nytimes.com/2020/09/30/health/covid-cruise-ships.html'

# n_topics_test = [1,2,3,4,5]
n_topics_test = [1]

n_topics_top_words = {}
n_topics_lda_topics = {}

for n_topics in n_topics_test:
    
    lda_top_topic_words_list, lda_top_topic_words_string, lda_topics = return_suggested_articles2(url,num_lda_topics=n_topics,use_pre_trained_model=0,n_passes=10,do_ngrams=1,print_article=1)
    
    n_topics_top_words[n_topics] = lda_top_topic_words_list
    n_topics_lda_topics[n_topics] = lda_topics

print(lda_top_topic_words_string)


# # lda_topics, queries = return_suggested_articles2(url)
# lda_topics,processed_dictionary,lda,bow_corpus = return_suggested_articles2(url)





# # print(queries)
# print(lda_topics)
#%%
lda_topics, processed_dictionary = return_suggested_articles2(url,num_lda_topics=n_topics,use_pre_trained_model=1,do_sentences=0)

# lda_topics_with_words = []
# for topic in lda_topics:
#     print(topic)
    
#     for word in topic:
#         print(word[0])
#         print(processed_dictionary[word[0]])
#         lda_topics_with_words.append((processed_dictionary[word[0]],word[1]))

# lda_topics_with_words2 = []

# for topics in lda_topics[len(lda_topics)-1]:
#     support = topics[1]
    
#     for word in support:
#         lda_topics_with_words2.append((processed_dictionary[word[0]],word[1]))
        
        
# lda_topics_with_words3 = []

# for word in lda_topics[0]:
#     print(word)
#     lda_topics_with_words3.append((processed_dictionary[word[0]],word[1]))
    

# for domain in queries.keys():
#     domain_name = domain.split('.')[0]
    
#     print(f"let {domain_name} = articles_json['{domain}']")

# st.write(queries)

# json_url = return_suggested_articles(url)

# st.write(json_url)
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
