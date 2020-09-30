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
    
    # if avoiding repeated words (only relevant if num_lda_topics > 1)
    unique_topic_words = 0
    
    # only for use when using one topic; this is number of words from that topic
    # that will be used in search
    n_topic_words = 5
    
    # set the number of topics to generate (5 seems to work pretty well)
    num_lda_topics = 1
    
    # set the number of passes
    n_passes = 6
    
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
    # stop_words.append('factset')
    
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
    
    print('splitting article into sentences')
    # break up into sentences
    combined_article = tokenize.sent_tokenize(combined_article)
    
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
    lda_top_topic_words_string, lda_top_topic_words_list = get_lda_top_topic_words(lda_topics,num_lda_topics,unique_topic_words,n_topic_words)
    
        
    # get dictionary of google queries    
    queries_dict = {}
    
    
    for domain in all_sides_domains:
        query = 'www.news.google.com/search?q=site:'+domain+lda_top_topic_words_string
        
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

def get_lda_top_topic_words(lda_topics,num_topic,unique_topic_words,n_topic_words):
    """
    fxn for algorithm to return the top topic words
    algo varies based on:
    1) whether only unique words are wanted, and
    2) whether there is 1 or more topics
    
                
    if one topic, just take top word in each generated topic
    else, if unique_topic_words, get top word in each topic that is unique,
          else, just get top word in each topic even if it isn't unique
    
    parameters
    ----------
    lda_topics - topic output from lda model
    num_topic - how many lda topics were generated
    unique_topic_words - whether to all repeating words as search terms
    n_topic_words - how many topic words to use as search terms
    
    outputs
    -------
    list and string of search/topic words
    """
    
    # string is for final search string
    lda_top_topic_words_string = ''

    # list is for checking previous words
    lda_top_topic_words_list = []

    # if generating more than 1 topic
    if num_topic > 1:

        # if you're okay with topic words repeating (often happens..)
        if not unique_topic_words:

            for topic in lda_topics:

                # get the list of topics
                topic_words = topic[1]

                lda_top_topic_words_string += ' '+topic_words[0][0]

                lda_top_topic_words_list.append(topic_words[0][0])

        # don't reuse a word if it has already been used
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
                        lda_top_topic_words_string += ' '+topic_words[i][0]

                        # update word added
                        word_added = 1
                        break

                # if no word was added because all supporting words in topic are already
                # search terms, then just add the highest prob/first word in topic
                if word_added == 0:
                    # if every word in this topic is already a search term,
                    # just add the first most probable word and leave the while loop
                    lda_top_topic_words_list.append(topic_words[0][0])
                    lda_top_topic_words_string += ' '+topic_words[0][0]

    else:

        for topic in lda_topics:

            # get the list of topic words
            topic_words = topic[1]

            # loop through these words and get the top n number
            counter = -1
            for topic_word in topic_words:

                counter += 1

                if counter < n_topic_words:

                    lda_top_topic_words_string += ' '+topic_word[0]
                    lda_top_topic_words_list.append(topic_word[0])
                    
    return lda_top_topic_words_string, lda_top_topic_words_list