"""
This module is used to train word
embeddings from the English Gigaword
corpus
"""

import argparse
import os
import re
from collections import defaultdict
from random import Random
import bs4

from newsclusters.utils import existsfile, load_config, process_story
import gzip
from gensim.models import Word2Vec, Doc2Vec
import logging
logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger()


import html

def gather_files(dir_root:str, ext:str=None):
    """
    Generator to list files recursively, with
    an optional extension filter
    """
    for root, dirs, filenames in os.walk(dir_root):
        for fullpath in [os.path.join(root, filename) for filename in filenames]:
            if ext is None or os.path.splitext(fullpath)[1] == ext:
                yield fullpath



def parse_corp(corp_path: str, counts=None, limit=0):
    """
    Given a GW Corpus path, process it
    into a list of TaggedDocuments

    :param corp_path:
    :return:
    """
    counts = defaultdict(int) if counts is None else counts

    # Keep track of current doc text:
    in_text = False
    cur_text = ''
    cur_id = None

    LOG.debug('Opening corpus file "{}"'.format(corp_path))
    with gzip.GzipFile(corp_path, 'r') as corp_file:
        for line in corp_file:

            line = line.decode('utf-8')
            if '<doc' in line.lower():
                LOG.debug('Processing story #{:,d}: "{}" -- {:,d} words'.format(counts['doc'],
                                                                                re.search('id="([^"]+)"', line, flags=re.I).group(1),
                                                                                counts['word'],
                                                                                ))
            elif '<text>' in line.lower():
                in_text = True
            elif '</text' in line.lower():
                if cur_text:
                    story = process_story(cur_text, tags=['SENT_{}'.format(counts['pgph']), 'DOC_{}'.format(counts['doc'])])
                    counts['word'] += len(story.words)
                    yield story
                    if limit != 0 and counts['word'] > limit:
                        raise StopIteration
                    counts['doc'] += 1
                cur_text = ''
                in_text = False
            elif '<p>' in line.lower():
                counts['pgph'] += 1
            elif in_text:
                text = html.unescape(bs4.BeautifulSoup(line).text)
                cur_text += text

def gather_docs(gw_root:str, limit=0, seed=None):
    """
    Return an iterator over the sentences in the GW corpus.

    :param gw_root:
    :return:
    """
    corp_paths = sorted(gather_files(gw_root, '.gz'))
    if seed is not None:
        r = Random(x=seed)
        r.shuffle(corp_paths)

    counts = defaultdict(int)

    for corp_path in corp_paths:
        LOG.debug('Parsing "{}"'.format(corp_path))
        for tagged_doc in parse_corp(corp_path, counts=counts, limit=limit):
            # Escape early if we've hit the doc limit
            yield tagged_doc
            if limit != 0 and counts['word'] > limit:
                raise StopIteration


class DocGatherer(object):
    """
    Create an iterator to feed sentences to Word2Vec
    """
    def __init__(self, corp_path, limit=0, seed=None):
        self._doc_iterator = gather_docs
        self._corp_path = corp_path
        self._limit = limit
        self._seed = seed

    def __iter__(self):
        return self._doc_iterator(self._corp_path, limit=self._limit, seed=self._seed)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dimensions', default=300, type=int, help='Number of dimensions for the embeddings')
    p.add_argument('-o', '--output', required=True, help='Path to save the output.')
    p.add_argument('-c', '--config', default='config.yml', help='Path to the config file.', type=existsfile)
    p.add_argument('-l', '--limit', default=0, help='Word limit', type=int)
    p.add_argument('-r', '--random-seed', default=None, help='Seed the random generator to have reproducible randomization of documents.', type=int)

    args = p.parse_args()

    c = load_config(args.config)

    gw_path = c.get('gw-path')
    w = Doc2Vec(documents=DocGatherer(gw_path, seed=args.random_seed, limit=args.limit), size=args.dimensions, min_count=1)
    w.save(args.output)