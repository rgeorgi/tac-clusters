"""
This module contains an OOP framework for processing the AQUAINT and English
Gigaword corpora, and extracting the relevant documents to create a "nice"
corpus with regular docIDs containing only the documents referenced by
the TAC topic guides.
"""


import os
import re
from collections import defaultdict
from gzip import GzipFile
from abc import abstractmethod
from lxml import etree
from newsclusters.utils import text_or_none
from bs4 import BeautifulSoup

class Guide(object):
    """
    A "Guide" object to represent the collection
    of topics/docsets, categories, and stories
    defined in the TAC XML files.
    """
    def __init__(self):
        self.categories = defaultdict(Category)

    def get_cat(self, cat_id):
        cat = self.categories[cat_id]
        cat.category_id = cat_id
        return cat

    @property
    def docs(self):
        """
        Return a list of all documents in the guide
        :rtype: list[DocumentMetadata]
        """
        return [doc for category in self.categories.values() for doc in category.docs ]

    @property
    def topics(self):
        """
        Return a list of all topics.
        :return: list[Topic]
        """
        return [topic for category in self.categories.values() for topic in category.topics]

    def find_topic(self, topic_id):
        for topic in self.topics:
            if topic.id == topic_id:
                return topic

    def find_doc(self, doc_id):
        for doc in self.docs:
            if doc.id == doc_id:
                return doc

    @classmethod
    def load(cls, guide_path):
        """
        Read the guide file from disk
        """
        g = cls()
        with open(guide_path, 'r') as guide_f:
            guide_entity = BeautifulSoup(guide_f.read(), "lxml")
            for topic_entity in guide_entity.find_all('topic'):
                cat = g.get_cat(topic_entity['category'])
                title = topic_entity.find('title').text.strip()
                t = Topic(topic_entity['id'], cat, title)
                cat.add_topic(t)

                for doc_entity in topic_entity.find_all('doc'):
                    doc = DocumentMetadata(doc_entity['id'], t)
                    t.add_doc(doc)

        return g


class Category(object):
    """
    Container class for topics.
    """
    def __init__(self, cat_id=None):
        self.category_id = cat_id
        self.topics = set([])

    def add_topic(self, t):
        self.topics.add(t)

    @property
    def docs(self):
        """
        Return a list of all documents in the category
        :rtype: list[DocumentMetadata]
        """
        return [doc for topic in self.topics for doc in topic.docs]

class Topic(object):
    """
    Class to represent a topic, and the documents
    that it contains.
    """
    def __init__(self, topic_id, category, title):
        self.category = category
        self.id = topic_id
        self.title = title
        self.docs = set([])

    def __repr__(self):
        return '<t: {} - {}>'.format(self.id, self.title)

    def add_doc(self, d): self.docs.add(d)

class DocumentMetadata(object):
    """
    Class to represent the metadata for a document,
    as described in guide document.

    Provides methods for parsing the doc_id into applicable
    metadata, such as the corpus, its publication date,
    and a regularized format.
    """
    def __init__(self, doc_id, topic):
        self.topic = topic
        self.id = doc_id

    @property
    def category(self)->Category: return self.topic.category if self.topic is not None else None

    @property
    def corp(self): return self.id[0:3]

    @property
    def reg_corp(self):
        if self.corp.lower() == 'xie':
            return 'XIN'
        else:
            return self.corp.upper()

    @property
    def date(self): return re.search('([0-9]{4})([0-9]{2})([0-9]{2})', self.id).groups()

    @property
    def year(self): return self.date[0]

    @property
    def month(self): return self.date[1]

    @property
    def day(self): return self.date[2]

    @property
    def number(self): return self.id.split('.')[1]

    @property
    def reg_id(self):
        """
        Returns a "regularized" ID string, so that
        document names can be normalized in the "nice"
        document collection.
        """
        return '{}_ENG_{}{}{}.{}'.format(self.reg_corp, self.year, self.month, self.day, self.number)


    @property
    def filename(self):
        """
        Figure out the path to this file.
        """
        return '{}_eng_{}{}.gz'.format(self.reg_corp.lower(), self.year, self.month)
        return corp

    def aq1_path(self, aq1_dir):
        return '{5}/{0}/{1}/{1}{2}{3}_{4}_ENG'.format(self.corp.lower(),
                                                      self.year,
                                                      self.month,
                                                      self.day,
                                                      self.corp.upper(),
                                                      aq1_dir)

    def gw_path(self, gw_dir):
        return os.path.join(os.path.join(gw_dir, 'data/{}_eng'.format(self.reg_corp.lower())),
                            self.filename)

    def nice_path(self, nice_dir):
        return os.path.join(nice_dir, self.reg_id)

    def vector_path(self, vector_dir):
        return os.path.join(vector_dir, self.reg_id)


    def __repr__(self): return '<d: {}>'.format(self.id)

class DocContent(object):
    """
    Object for serializing/deserializing stories into a "nice"
    format that is consistent between corpora.
    """
    def __init__(self, headline=None, dateline=None, sents=None, doc_id=None, doc: DocumentMetadata=None):
        self.headline = headline
        self.dateline = dateline
        self.sents = sents
        self.doc_id = doc_id
        self.doc=doc

    @property
    def text(self):
        return ' '.join([s for s in self.sents])


    @classmethod
    def load(cls, path):
        """
        Deserialize a document.
        """
        with open(path, 'r') as f:
            dc = DocContent()
            data = BeautifulSoup(f.read())
            dc.headline = text_or_none(data.find('headline'))
            dc.dateline = text_or_none(data.find('dateline'))
            dc.doc_id = dc.find('doc')['id']
            dc.sents = [p.text for p in data.find_all('p')]
        return dc

    def write(self, path):
        root = etree.Element('DOC', id=self.doc.reg_id)
        if self.headline:
            headline = etree.SubElement(root, 'HEADLINE')
            headline.text = self.headline

        if self.dateline:
            dateline = etree.SubElement(root, 'DATELINE')
            dateline.text = self.dateline

        text = etree.SubElement(root, 'TEXT')
        for sent in self.sents:
            if not sent.strip():
                continue
            p = etree.SubElement(text, 'P')
            p.text = sent

        with open(path, 'wb') as f:
            f.write(etree.tostring(root, pretty_print=True))

# -------------------------------------------
# Classes to represent the different corpora
# -------------------------------------------


class BaseCorpFile(object):
    """
    Abstract class to represent documents from the different
    corpora. (Note that AQUAINT and the English Gigaword both
    have file structures containing multiple stories/docs per
    file.)
    """
    def __init__(self, path):
        self.path = path

    @abstractmethod
    def get_stories(self, story_ids: list):
        """
        Given a list of story_ids, return a list
        of DocContent objects retrieved from within
        this file.

        :rtype: list[DocContent]
        """
        pass


class AQ1FileBase(BaseCorpFile):
    """
    A class representing AQUAINT(-1) documents
    """

    def create_doc_content(self, bs_doc, metadata: DocumentMetadata) -> DocContent:
        """
        Given the contents of an AQ1 doc,
        normalize it to a "DocContent" object

        :param bs_doc: a BeautifulSoup parse of the file.
        """
        dc = DocContent()
        dc.doc = metadata
        dc.sents = bs_doc.find('text').get_text().split('\n\n')
        dc.headline = text_or_none(bs_doc.find('headline'))
        dc.dateline = text_or_none(bs_doc.find('date_time') or bs_doc.find('dateline'))
        return dc

    def get_stories(self, doclist: list):
        """
        AQUAINT-specific implementation for retrieving the
        appropriate document IDs
        """
        found_stories = []

        with open(self.path, 'r', encoding='utf-8') as aq1_f:

            aq_data = BeautifulSoup(aq1_f.read(), 'lxml')

            docs = aq_data.find_all('doc')

            for doc_elt in docs:
                doc_id = doc_elt.find('docno').text.strip()
                matching_docs = [doc for doc in doclist if doc.id == doc_id]
                if matching_docs:
                    dc = self.create_doc_content(doc_elt, matching_docs[0])
                    found_stories.append(dc)

        return found_stories



class GWFileBase(BaseCorpFile):
    """
    A class representing English Gigaword documents
    """
    def get_stories(self, story_ids: list):
        """
        An attempt at creating a (fairly) efficient
        parser for ENG-GW docs to retrieve the set
        of story IDs specified, and terminate once
        the list is exhausted.
        """
        cur_doc = ''
        cur_id = None
        docs = {}

        # ...because ENG-GW docs are GZipped
        with GzipFile(self.path, 'r') as gz:
            doc_data = gz.readlines()
            for line in doc_data:
                linestr = line.decode('utf-8')

                if cur_id is not None:
                    cur_doc += linestr

                if linestr.strip() == '</DOC>' and cur_id is not None:
                    docs[cur_id] = cur_doc
                    cur_doc = ''
                    cur_id = None

                    if not story_ids:
                        break

                elif linestr.startswith('<DOC'):
                    doc_id = re.search('id="([^"]+)"', linestr).group(1)
                    if doc_id in story_ids:
                        cur_doc += linestr
                        cur_id = doc_id
                        del story_ids[story_ids.index(doc_id)]

        return docs

