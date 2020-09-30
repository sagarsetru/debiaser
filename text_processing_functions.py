#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:53:43 2020

@author: sagarsetru
"""

# gensim
import json

from google.cloud import storage
from flask import escape

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

import spacy

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

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

        