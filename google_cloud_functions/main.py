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

# following https://stackoverflow.com/questions/62209018/any-way-to-import-pythons-nltk-downloadpunkt-into-google-cloud-functions
# import nltk.data as nldata
# root = os.path.dirname(os.path.abspath(__file__))
# download_dir = os.path.join(root, 'nltk_data')
# nldata.load(
#     os.path.join(download_dir, 'tokenizers/punkt/english.pickle')
# )
from nltk import download as nldl
nldl('punkt')

import pickle
# os.environ['NLTK_DATA'] = nltk_data

# from nltk.corpus import stopwords

# for counting frequency of words
# from collections import defaultdict

# import spacy
# nlp  = spacy.load('en_core_web_sm')
nlp = []

# from googlesearch import search

# import requests

# import urllib3
# import urllib



def return_suggested_articles(request):
    """
    returns suggested articles based on topic of one currently being viewed

    Parameters
    ----------
    request : request (flask.Request): The request object

    Returns
    -------
    JSON of google search queries for articles to read

    """    
    
    
    # get the requested json for the webpage
    request_json = request.get_json(silent=True)
    
    
    # get the headline and article
    headline = request_json['headline']
    article = request_json['article']
    print('requested json headline and article text')
    
    # make into one text file
    combined_article = headline+'. '+article
    
    # set to 1 for single doc lda, 0 for tfidf
    do_single_document_LDA = 1
    
    # number of query words to return
    n_search_words = 5
    
    # can identify ngrams, but slows down performance
    do_ngrams = 1
    
    ### SINGLE DOC LDA PARAMS
    
    # set the number of topics to generate (5 seems to work pretty well)
    num_lda_topics = 5
    
    # set the number of passes
    n_passes = 10
    
    # if avoiding repeated words (only relevant if num_lda_topics > 1)
    do_unique_search_words = 0
    
    print('Downloading stop words')
    # download stopwords list
    # if use_bucket:
    download_blob('debiaser_data', 'sw1k.csv', '/tmp/sw1k.csv')
    
    # load stop words into pandas and then into list
    stop_words = pd.read_csv('/tmp/sw1k.csv')
    
    # remove from memory
    os.remove('/tmp/sw1k.csv')
        
    stop_words = stop_words['term']
    stop_words = [word for word in stop_words]
    
    # # adding some custom words
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
    
    print('Downloading news organizations from AllSidesMedia')
    # download all_sides_media list
    # if use_bucket:
    download_blob('debiaser_data','allsides_final_plus_others_with_domains.csv', '/tmp/allsides_final_plus_others_with_domains.csv')
    
    # load domain names into dataframe and then get only names and
    all_sides = pd.read_csv('/tmp/allsides_final_plus_others_with_domains.csv')
    
    # remove from memory
    os.remove('/tmp/allsides_final_plus_others_with_domains.csv')

    # get the domain
    # all_sides_names = all_sides['name']
    all_sides_domains = all_sides['domain']
    # all_sides_names_domains = pd.concat([all_sides_names,all_sides_domains],axis=1)

    # get dictionary of entities in article
    # entity_dict = entity_recognizer(combined_article,nlp)
    
    if do_single_document_LDA:
        
        print('splitting article into sentences')
        # break up into sentences
        combined_article = tokenize.sent_tokenize(combined_article)
        
    else:
        
        # make into one element list for downstream processing
        combined_article = [combined_article]
        
    print('pre processing article text')
    # process article
    article_processed = process_all_articles(combined_article,nlp)
    
    print('removing stopwords')
    # remove stopwords
    article_processed = remove_stopwords(article_processed,stop_words)
        
    
    # floor for the frequency of words to remove
    # word_frequency_threshold = 1
    
    # get corpus, dictionary, bag of words
    # processed_corpus, processed_dictionary, bow_corpus = get_simple_corpus_dictionary_bow(article_processed,
    #                                                                                       word_frequency_threshold)
    
    if do_single_document_LDA:
            
        if do_ngrams:
            # load bigram trigram quadgram models
            bigram_mod_fname = '/tmp/bigram_mod.pkl'
            trigram_mod_fname = '/tmp/trigram_mod.pkl'
            quadgram_mod_fname = '/tmp/quadgram_mod.pkl'
            
            download_blob('debiaser_data','bigram_mod.pkl',bigram_mod_fname)
            download_blob('debiaser_data','trigram_mod.pkl',trigram_mod_fname)
            download_blob('debiaser_data','quadgram_mod.pkl',quadgram_mod_fname)
            
            with open(bigram_mod_fname, 'rb') as pickle_file:
                bigram_mod = pickle.load(pickle_file)
                
            with open(trigram_mod_fname, 'rb') as pickle_file:
                trigram_mod = pickle.load(pickle_file)
                
            with open(quadgram_mod_fname, 'rb') as pickle_file:
                quadgram_mod = pickle.load(pickle_file)
                
            print('FINDING QUADGRAMS')
            
            # make up to quad grams
            article_processed = make_quadgrams(article_processed,bigram_mod,trigram_mod,quadgram_mod)
            
            # remove to free memory
            os.remove(bigram_mod_fname)
            os.remove(trigram_mod_fname)
            os.remove(quadgram_mod_fname)
        
        print('generating dictionary and bag of words vector...')
        start = time.process_time()
        processed_corpus, processed_dictionary, bow_corpus = get_simple_corpus_dictionary_bow(article_processed)
        print('TIME FOR GENERATING DICTIONARY AND BOW VECTOR')
        print(time.process_time() - start)
        
        print('generating lda model...')
        start = time.process_time()
        # generate the LDA model
        lda = LdaModel(corpus = bow_corpus,
                        num_topics = num_lda_topics,
                        id2word = processed_dictionary,
                        passes = n_passes)
        print('TIME FOR GENERATING LDA MODEL')
        print(time.process_time() - start)
        
        
        # get the topics from the lda model
        lda_topics = lda.show_topics(formatted = False)
            
            
        # ALL INTERESTING BUT DEPRECATED FOR NOW
        # WILL FOLLOW SIMPLER APPROACH:
            # Just take top word in each generated topic
            
        # get top words per topic
        lda_top_topic_words_string, lda_top_topic_words_list = get_lda_top_topic_words(lda_topics,num_lda_topics,do_unique_search_words,n_search_words)
        
    # doing tfidf
    else:
        
        # specify file name
        tfidf_matrix_filename = '/tmp/tfidf_matrix.pkl'
        
        # download the tfidf matrix
        print('DOWNLOADING TFIDF MODEL')
        download_blob('debiaser_data','tfidf_matrix.pkl', tfidf_matrix_filename)
        
        with open(tfidf_matrix_filename, 'rb') as pickle_file:
            tfidf = pickle.load(pickle_file)
        
        # remove from memory
        os.remove(tfidf_matrix_filename)
        
        if do_ngrams:
            # load bigram trigram quadgram models
            bigram_mod_fname = '/tmp/bigram_mod.pkl'
            trigram_mod_fname = '/tmp/trigram_mod.pkl'
            quadgram_mod_fname = '/tmp/quadgram_mod.pkl'
            
            download_blob('debiaser_data','bigram_mod.pkl',bigram_mod_fname)
            download_blob('debiaser_data','trigram_mod.pkl',trigram_mod_fname)
            download_blob('debiaser_data','quadgram_mod.pkl',quadgram_mod_fname)
            
            with open(bigram_mod_fname, 'rb') as pickle_file:
                bigram_mod = pickle.load(pickle_file)
                
            with open(trigram_mod_fname, 'rb') as pickle_file:
                trigram_mod = pickle.load(pickle_file)
                
            with open(quadgram_mod_fname, 'rb') as pickle_file:
                quadgram_mod = pickle.load(pickle_file)
            
            # make up to quad grams
            combined_article = make_quadgrams(combined_article,bigram_mod,trigram_mod,quadgram_mod)
        
            # remove to free memory
            os.remove(bigram_mod_fname)
            os.remove(trigram_mod_fname)
            os.remove(quadgram_mod_fname)
            
        # download dictionary
        id2word_fname = '/tmp/id2word.pkl'
        download_blob('debiaser_data','id2word_ec2.pkl',id2word_fname)
        
        with open(id2word_fname, 'rb') as pickle_file:
            processed_dictionary = pickle.load(pickle_file)
        
        # remove to free memory
        os.remove(id2word_fname)
        
        print('GENERATING BOW VECTOR FOR ARTICLE')
        # get bag of words representation
        bow_corpus_article = [processed_dictionary.doc2bow(text) for text in combined_article]
        
        print('GETTING TF IDF SCORE')
        tfidf_vector = tfidf[bow_corpus_article[0]]
    
        # sort the tfidf vector
        tfidf_vector = sorted(tfidf_vector,key=getKey,reverse=True)
        
        # if there are fewer words than search words, then just use how many words there are
        if len(tfidf_vector) < n_search_words:
            n_search_words = len(tfidf_vector)
        
        top_tfidf_values = [tfidf_vector[i][0] for i in range(0,n_search_words)]
        print(top_tfidf_values)
        
        top_words_list = [processed_dictionary[i].replace("_"," ") for i in top_tfidf_values]
        
        top_words_string = ' '
        for word in top_words_list:
            if word not in top_words_string:
                top_words_string += ' '+word
        
    
    # get dictionary of google queries    
    queries_dict = {}
    
    for domain in all_sides_domains:
        
        # if this is single document lda
        if do_single_document_LDA:
            query = 'www.news.google.com/search?q=site:'+domain+lda_top_topic_words_string
        
        # if this is tfidf
        else:
            query = 'www.news.google.com/search?q=site:'+domain+top_words_string
        
        queries_dict[domain] = query
        
    return json.dumps(queries_dict)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """
    Downloads a blob from the bucket.
    From: https://cloud.google.com/storage/docs/downloading-objects#code-samples
    """
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )
    
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
    '''fxn identifies bigrams in text'''
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts,bigram_mod,trigram_mod):
    '''fxn identifies trigrams in text'''
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def make_quadgrams(texts,bigram_mod,trigram_mod,quadgram_mod):
    '''fxn identifies quadrigrams in text'''
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

# def get_lda_top_topic_words(lda_topics,num_topics,do_unique_search_words,n_search_words):
#     """
#     fxn for algorithm to return the top topic words
#     algo varies based on:
#     1) whether only unique words are wanted, and
#     2) whether there is 1 or more topics
    
                
#     if one topic, just take top word in each generated topic
#     else, if do_unique_search_words, get top word in each topic that is unique,
#           else, just get top word in each topic even if it isn't unique
    
#     parameters
#     ----------
#     lda_topics - topic output from lda model
#     num_topic - how many lda topics were generated
#     do_unique_search_words - whether to all repeating words as search terms
#     n_search_words - how many topic words to use as search terms
    
#     outputs
#     -------
#     list and string of search/topic words
#     """
    
#     # string is for final search string
#     lda_top_topic_words_string = ''

#     # list is for checking previous words
#     lda_top_topic_words_list = []
    
#     # if lda model has only one topic
#     if num_topics == 1:

#         for topic in lda_topics:

#             # get the list of topic words
#             topic_words = topic[1]

#             # loop through these words and get the top n number
#             counter = -1
#             for topic_word in topic_words:

#                 counter += 1            

#                 if counter < n_search_words:

#                     if topic_word not in lda_top_topic_words_list:
                        
#                         lda_top_topic_words_string += ' '+topic_word[0].replace("_"," ")
#                         lda_top_topic_words_list.append(topic_word[0].replace("_"," "))

#     # if lda model has more than one topic
#     elif num_topics > 1:
            
#         # this ind is to always get list of tuples of (word, prob)
#         fixed_ind1 = 1

#         # this ind is to always access the word in the tuple (word, prob)
#         fixed_ind2 = 0

#         # if you're okay with topic words repeating (often happens..)
#         if not do_unique_search_words:

#             # loop counter
#             counter = 0
            
#             # index of word within topic
#             ind_use = 0
            
#             # index of topic
#             topic_use = -1
            
#             for i in range(n_search_words):
#                 counter += 1

#                 if counter > num_topics:
#                     ind_use += 1
#                     counter = 1

#                 if topic_use < num_topics-1:
#                     topic_use += 1
#                 else:
#                     topic_use = 0

#                 # access the appropriate topic word
#                 word = lda_topics[topic_use][fixed_ind1][ind_use][fixed_ind2]
                
#                 word = word.replace("_"," ")

#                 lda_top_topic_words_string += ' '+word

#                 lda_top_topic_words_list.append(word)

#         # don't reuse a word if it has already been used
#         else:

#             counter = 0
#             ind_use = 0
#             topic_use = -1
            
#             # do loop over total words across all topics
#             total_topic_words = len(lda_topics)*len(lda_topics[0][fixed_ind1])
#             for i in range(total_topic_words):
#                 counter += 1

#                 if counter > num_topics:
#                     ind_use += 1
#                     counter = 1

#                 if topic_use < num_topics-1:
#                     topic_use += 1
#                 else:
#                     topic_use = 0

#                 # access the appropriate topic word
#                 word = lda_topics[topic_use][fixed_ind1][ind_use][fixed_ind2]
                
#                 word = word.replace("_"," ")

#                 # only add if it is not currently in the top topic words
#                 if word not in lda_top_topic_words_list:

#                     lda_top_topic_words_string += ' '+word

#                     lda_top_topic_words_list.append(word)
                
#                 # if the length of the topic words list is at the number of descired topics
#                 if len(lda_top_topic_words_list) == n_search_words:
#                     break

#     return lda_top_topic_words_string, lda_top_topic_words_list

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
                    
        # remake the string now that word has been removed
        lda_top_topic_words_string = ''
        
        for already_word in lda_top_topic_words_list:
            lda_top_topic_words_string += ' '+already_word

    
    if do_unique_search_words:
        
        for already_word in lda_top_topic_words_list:
                
                if already_word in word:
    
                    lda_top_topic_words_list.remove(already_word)
                    
                    counter -= 1
        
        lda_top_topic_words_string = ''
        
        # remake the string now that word has been removed
        for already_word in lda_top_topic_words_list:
            lda_top_topic_words_string += ' '+already_word
            
        
        # check to confirm this new word isn't a unigram in an ngram
        if word not in lda_top_topic_words_string:
            
            lda_top_topic_words_string += ' '+word.replace("_"," ")
            lda_top_topic_words_list.append(word.replace("_"," "))
    else:
        
        lda_top_topic_words_string += ' '+word.replace("_"," ")
        lda_top_topic_words_list.append(word.replace("_"," "))
        
    return lda_top_topic_words_string, lda_top_topic_words_list, counter

def getKey(item):
    return item[1]