#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:53:43 2020

@author: sagarsetru
"""

# gensim
import gensim
from gensim.corpora import Dictionary
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict

import string

import spacy

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

def process_text(article_text):
    """
    processes document text by removing stop words, making all lower case,
    and removing punctuation.
    

    Parameters
    ----------
    article_text : document text that you want to preprocess.

    -------
    processed text for document.

    """
    
    # run gensim preprocess
    article_text = gensim.utils.simple_preprocess(article_text, deacc=True)
    
    # # replace weird apostrophes
    # article_text = article_text.replace("`","'")
    # article_text = article_text.replace("’","'")
    # article_text = article_text.replace("'","'")
    # article_text = article_text.replace("“","'")
    # article_text = article_text.replace("”","'")
    
    # # replace long dashes with short dashes
    # article_text = article_text.replace("—","-")
    
    # # replace short dashes with spaces
    # article_text = article_text.replace("-"," ")
    
    # article_text = article_text.replace("…","...")
    
    # # get rid of punctuation
    # article_text = article_text.translate(article_text.maketrans('', '', string.punctuation))
    
    # # Lowercase each document, split it by white space and filter out stopwords
    # article_text = [[word for word in document.lower().split()] for document in [article_text]]
    
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
        