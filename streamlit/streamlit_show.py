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

# NLP Packages

from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.utils import simple_preprocess

# to break articles up into sentences
from nltk import tokenize
from nltk.corpus import stopwords

# for counting frequency of words
from collections import defaultdict

# import spacy
# nlp  = spacy.load('en_core_web_sm')
nlp = []

from googlesearch import search

import requests

from bs4 import BeautifulSoup

# import urllib3
# import urllib

# import streamlit as st

def return_suggested_articles2(url):
    """
    returns suggested articles based on topic of one currently being viewed

    Parameters
    ----------
    request : request (flask.Request): The request object..

    Returns
    -------
    articles to read.

    """    
    
    use_bucket = 0
    
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
    
    combined_article = headline+p_tags_text_1string
    
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
        
        # use nltk stopwords
        # stop_words = list(stopwords.words('english'))
        
    stop_words = stop_words['term']
    stop_words = [word for word in stop_words]
    
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
    
    # all_sides_names = all_sides['name']
    # all_sides_domains = all_sides['domain']
    # all_sides_names_domains = pd.concat([all_sides_names,all_sides_domains],axis=1)
    
    # clean up workspace for memory
    # if use_bucket:
    #     os.remove('/tmp/all_sides.csv')
    #     os.remove('/tmp/stop_words_1k.csv')
    
    # get dictionary of entities in article
    # entity_dict = entity_recognizer(combined_article,nlp)
    
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
    article_processed = process_all_articles(combined_article,nlp)
    
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
    n_passes = 100
    
    # generate the LDA model
    lda = LdaModel(corpus = bow_corpus,
                   num_topics = num_lda_topics,
                   id2word = processed_dictionary,
                   passes = n_passes)
    
    # get the topics from the lda model
    lda_topics = lda.show_topics(formatted = False)
    
    
    # ALL INTERESTING BUT DEPRECATED FOR NOW
    # WILL FOLLOW SIMPLER APPROACH:
        # Just take top word in each generated topic
        
    # get top words per topic
    
    # string is for final search string
    lda_top_topic_words = ''
    
    # list is for checking previous words
    lda_top_topic_words_list = []
    
    for topic in lda_topics:
        
        # get the list of topics
        topic_words = topic[1]
        
        lda_top_topic_words += ' '+topic_words[0][0]
    
    # # loop through each list of generated topics
    # for topic in lda_topics:
        
    #     # set word added to 0
    #     word_added = 0
        
    #     # get the list of topics
    #     topic_words = topic[1]
            
    #     # loop through words in topic
    #     # add as search term only if they aren't already search terms
    #     for i in range(len(topic_words)):
        
    #         # if the current word in topic is not in list of search terms
    #         if topic_words[i][0] not in lda_top_topic_words_list:
                
    #             # add this word to list of topic/search terms
    #             lda_top_topic_words_list.append(topic_words[i][0])
                
    #             # also update the string for the search terms
    #             lda_top_topic_words += ' '+topic_words[i][0]
                
    #             # update word added
    #             word_added = 1
    #             break
        
    #     # if no word was added because all supporting words in topic are already
    #     # search terms, then just add the highest prob/first word in topic
    #     if word_added == 0:
    #         # if every word in this topic is already a search term,
    #         # just add the first most probable word and leave the while loop
    #         lda_top_topic_words_list.append(topic_words[0][0])
    #         lda_top_topic_words += ' '+topic_words[0][0]
        

            
    
    # get list of google queries
    queries = []
    
    # quick manual entry
    all_sides_domains = ['nytimes.com','wsj.com']
    all_sides_names = ['nyt','wsj']
    
    for domain in all_sides_domains:
        query = 'site:'+domain+lda_top_topic_words
        queries.append(query)        
    
    # # create dictionary for results of query
    # query_results = {}

    # # loop through queries
    # for ind, query in enumerate(queries):
        
    #     # initialize list for results of query
    #     current_results = []
        
    #     # loop through search results
    #     for j in search(query, tld='com', lang='en', num = 1, start=0, stop = 5, pause = 2.0):
            
    #         # append search result to current list of results
    #         current_results.append(j)
            
    #     # append list of results from query to dictionary for that query
    #     query_results[all_sides_names[ind]] = current_results
    
    # # also create entry in dictionary for the search terms
    # query_results['search_terms'] = lda_top_topic_words
    
    
    # # convert dictionary to json dictionary
    # json_object = json.dumps(query_results, indent = 4)
    
    print(lda_topics)
    print(headline)
    print(combined_article)
    print(lda_top_topic_words)
    print(topic_words)
    return queries

def return_suggested_articles(url):
    """
    returns suggested articles based on topic of one currently being viewed

    Parameters
    ----------
    request : request (flask.Request): The request object..

    Returns
    -------
    articles to read.

    """    
    
    use_bucket = 0
    
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
    
    combined_article = headline+p_tags_text_1string
    
    # # get the requested json for the webpage
    # request_json = request.get_json(silent=True)
    
    # # get the headline and article
    # headline = request_json['headline']
    # article = request_json['article']
    
    # make into one text file
    # combined_article = headline+article
    
    # download stopwords list
    if use_bucket:
        download_from_bucket('debiaser_data', 'sw1k.csv', '/tmp/stop_words_1k.csv')
    
        # load stop words into pandas and then into list
        stop_words = pd.read_csv('/tmp/stop_words_1k.csv')
    
    else:
        stop_words = pd.read_csv('/Users/sagarsetru/Documents/post PhD positions search/insightDataScience/project/debiaser/stop_words_db/news-stopwords-master/sw1k.csv')
        
        # use nltk stopwords
        # stop_words = list(stopwords.words('english'))
        
    # stop_words = stop_words['term']
    # stop_words = [word for word in stop_words]
    
    # # adding some custom words
    # stop_words.append('said')
    # stop_words.append('youre')
    # stop_words.append('mph')
    
    
    # download all_sides_media list
    if use_bucket:
        download_from_bucket('debiaser_data','allsides_final_plus_others_with_domains.csv.csv', '/tmp/all_sides.csv')
    
        # load domain names into dataframe and then get only names and
        all_sides = pd.read_csv('/tmp/all_sides.csv')
    
    # else:
    #     all_sides = pd.read_csv('/Users/sagarsetru/Documents/post PhD positions search/insightDataScience/project/debiaser/all_sides_media_data/allsides_final_plus_others_with_domains.csv')
    
    # all_sides_names = all_sides['name']
    # all_sides_domains = all_sides['domain']
    # all_sides_names_domains = pd.concat([all_sides_names,all_sides_domains],axis=1)
    
    # clean up workspace for memory
    if use_bucket:
        os.remove('/tmp/all_sides.csv')
        os.remove('/tmp/stop_words_1k.csv')
    
    # get dictionary of entities in article
    # entity_dict = entity_recognizer(combined_article,nlp)
    
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
    article_processed = process_all_articles(combined_article,nlp)
    
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
    n_passes = 100
    
    # generate the LDA model
    lda = LdaModel(corpus = bow_corpus,
                   num_topics = num_lda_topics,
                   id2word = processed_dictionary,
                   passes = n_passes)
    
    # get the topics from the lda model
    lda_topics = lda.show_topics(formatted = False)
    
    
    # ALL INTERESTING BUT DEPRECATED FOR NOW
    # WILL FOLLOW SIMPLER APPROACH:
        # Just take top word in each generated topic
        
    # get top words per topic
    lda_top_topic_words = ''
    
    # loop through each list of generated topics
    for topic in lda_topics:
        
        # get the list of topics
        topic_words = topic[1]
        
        lda_top_topic_words += ' '+topic_words[0][0]
    
    print(lda_top_topic_words)
    # get list of google queries
    queries = []
    
    # quick manual entry
    all_sides_domains = ['nytimes.com','wsj.com']
    all_sides_names = ['nyt','wsj']
    
    for domain in all_sides_domains:
        query = 'site:'+domain+lda_top_topic_words
        queries.append(query)        
    
    # create dictionary for results of query
    query_results = {}

    # loop through queries
    for ind, query in enumerate(queries):
        
        # initialize list for results of query
        current_results = []
        
        # loop through search results
        for j in search(query, tld='com', lang='en', num = 1, start=0, stop = 5, pause = 2.0):
            
            # append search result to current list of results
            current_results.append(j)
            
        # append list of results from query to dictionary for that query
        query_results[all_sides_names[ind]] = current_results
    
    # also create entry in dictionary for the search terms
    query_results['search_terms'] = lda_top_topic_words
    
    
    # convert dictionary to json dictionary
    json_object = json.dumps(query_results, indent = 4)
    
    return json_object
    
    # # get the unique topic words
    # # get dictionary of each topic word and their associated probabilities per topic
    # # get dictionary of each topic word and their mean probability
    # # get dictionary of each topic word and their std dev probability
    # # get dictionary of each topic word and the frequency with which the words show up
    # topics, topics_probs_dict, topics_mean_probs_dict, topics_std_probs_dict, topics_frequency_dict = get_topic_words_mean_std_prob_frequency(lda_topics)
    
    # # get the topic mean probs and frequencies, sorted
    # topics_means, means_sorted, std_sorted, topics_freq, freq_sorted = sort_topics_mean_frequency(topics,
    #                                          topics_mean_probs_dict,
    #                                          topics_std_probs_dict,
    #                                          topics_frequency_dict)
    
    
    
    # # get the words to use in google search
    # # (for now, just pick the top 3 probs and top 2 most common)
    # words_to_search = [topics_means[0:2],topics_freq[]
    
    # return json.dumps(combined_article)


# def download_from_bucket(bucket_name,source_data_name,destination_file_name):
#     """
    

#     Parameters
#     ----------
#     bucket_name : name of the bucket on google cloud.
#     source_data_name : name of the data set.
#     destination_file_name : where the data will be stored.

#     Returns
#     -------
#     None.

#     """
    
#     # load instance of the storage client
#     storage_client = storage.Client()
    
#     # get the bucket from the bucket name
#     bucket = storage_client.get_bucket(bucket_name)
    
#     # get the data set/bloc
#     data_set = bucket.blob(source_data_name)
    
#     # save the dataset at the destination file name
#     data_set.download_to_filename(destination_file_name)
    
#     print(f'Blob data {source_data_name} downloaded to {destination_file_name}')
    
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

# url = st.text_input("URL: ", 'https://www.theguardian.com/us-news/2020/sep/25/ruth-bader-ginsburg-us-capitol-lie-in-state')

# st.title('debiaser')

url = 'https://www.theguardian.com/sport/2020/sep/25/la-lakers-denver-nuggets-game-4-recap'
url = 'https://www.nytimes.com/2020/09/25/us/politics/rbg-retirement-obama.html'

queries = return_suggested_articles2(url)

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
