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
    n_passes = 5
    
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
    
        
    
    # get dictionary of google queries    
    queries_dict = {}
    
    # quick manual entry
    # all_sides_domains = ['nytimes.com','wsj.com']
    # all_sides_names = ['nyt','wsj']
    
    for domain in all_sides_domains:
        query = 'www.google.com/search?q=site:'+domain+lda_top_topic_words
        
        queries_dict[domain] = query
    
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
