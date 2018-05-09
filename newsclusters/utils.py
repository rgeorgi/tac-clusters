"""
A handful of convenience functions.
"""
import os
import argparse
import re

import yaml
from nltk.tokenize import PunktSentenceTokenizer, WordPunctTokenizer
from gensim.models.doc2vec import TaggedDocument

def text_or_none(item):
    if item is None:
        return None
    else:
        return item.text.strip()

def existsfile(path):
    if os.path.exists(path) and os.path.isfile(path):
        return path
    else:
        raise argparse.ArgumentTypeError('Path "{}" does not exist or is not a file.'.format(path))

def load_config(path: str) -> yaml.YAMLObject:
    """
    Load the config file.
    """
    with open(path, 'r') as y:
        return yaml.load(y)

wt = WordPunctTokenizer()
st = PunktSentenceTokenizer()
# punc_re = re.compile('[^\w]+')

def process_story(doc_text, tags=None):
    """
    Process a story from within a
    :rtype: TaggedDocument
    """
    words = []
    tags = tags if tags else []

    sents = st.tokenize(doc_text)
    for sent_num, sent in enumerate(sents):
        for word in wt.tokenize(sent):
            words.append(word)

    return TaggedDocument(words, tags)
