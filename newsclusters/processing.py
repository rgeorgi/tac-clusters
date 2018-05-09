"""
This module contains the processing scripts
for running the semi-supervised clustering
experiments.
"""

import os
import pickle
import random
import argparse
import sys
from collections import defaultdict


from newsclusters.model import Guide, DocContent, AQ1FileBase, GWFileBase, BaseCorpFile
from newsclusters.utils import text_or_none, load_config, existsfile, process_story

import spacy
import gensim
from gensim.models.doc2vec import Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np

# -------------------------------------------
# Set up Logging
# -------------------------------------------
import logging
logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger()

# -------------------------------------------
# CONSTANTS
# -------------------------------------------
CATEGORY_CLUSTER = 'category'
TOPIC_CLUSTER = 'topic'

GLOVE_VECTORS = 'glove'
TFIDF_VECTORS = 'tfidf'
WORD2VEC_VECTORS = 'word2vec'
DOC2VEC_VECTORS = 'doc2vec'

# -------------------------------------------
# Matrices
#
# Different ways of creating matrices for document vectors
# -------------------------------------------


def tfidf_matrix(doclist, nice_dir):
    """
    Create tf*idf vectors for all documents

    :return: numpy array of shape (|Documents|, |Vocabulary|)
    """
    tfidf = TfidfVectorizer(input='filename',
                            stop_words='english')
    matrix = tfidf.fit_transform([doc.nice_path(nice_dir) for doc in doclist])
    return matrix


def glove_matrix(doclist, nice_dir, vector_dir, spacy_model, overwrite=False):
    """
    Create vectors for documents using pre-trained GloVe embeddings from spacy.

    :return: numpy array of shape (|Documents|, 300)
    """
    # Start by creating a directory to contain the
    # pickled vector files, if it doesn't exist.
    os.makedirs(vector_dir, exist_ok=True)

    # Initialize the empty matrix.
    matrix = np.ndarray(shape=(len(doclist), 300))

    # Populate the matrix with saved vectors, or generate them
    # fresh if "overwrite" is True.
    for i, doc in enumerate(doclist):
        doc_path = doc.nice_path(nice_dir)
        vec_path = doc.vector_path(vector_dir)
        if not os.path.exists(vec_path) or overwrite:
            dc = DocContent.load(doc_path)
            doc_vector = spacy_model(dc.text).vector
            with open(vec_path, 'wb') as vec_f:
                pickle.dump(doc_vector, vec_f)
        else:
            with open(vec_path, 'rb') as vec_f:
                doc_vector = pickle.load(vec_f)

        # Update the matrix row
        matrix[i] = doc_vector

    return matrix


def doc_vector_from_word2vec(model: gensim.models.Word2Vec,
                             td: gensim.models.doc2vec.TaggedDocument):
    """
    Given a word2vec model, create a document vector for the
    :param model:
    :param td:
    :return:
    """
    word_vectors = []
    for word in td.words:
        try:
            word_vectors.append(model[word])
        except KeyError:
            pass

    if not word_vectors:
        return np.zeros(shape=(model.vector_size,))
    else:
        return np.mean(word_vectors, axis=0)


def gensim_matrix(doclist, nice_dir, vector_dir, d2v_model: gensim.models.Doc2Vec, overwrite=False):
    """
    Create vectors for documents using gensim-trained Word2Vec model.

    :return:
    """
    os.makedirs(vector_dir, exist_ok=True)

    # -------------------------------------------
    # Initialize the cluster matrix
    # -------------------------------------------
    vec_size = d2v_model.vector_size
    matrix = np.ndarray(shape=(len(doclist), vec_size),
                        dtype=np.float32)

    for i, path in enumerate(doclist):

        doc_path = path.nice_path(nice_dir)
        dc = DocContent.load(doc_path)
        td = process_story(dc.text, i)

        # If it's a doc2vec model, we can just use
        #  "infer vector"
        if isinstance(d2v_model, gensim.models.Doc2Vec):
            matrix[i] = d2v_model.infer_vector(td.words)
        else:
            matrix[i] = doc_vector_from_word2vec(d2v_model, td)

    return matrix







def run_clustering_experiments(guide: Guide, nice_dir: str, vector_dir: str, overwrite: bool=False,
                               vector_type=TFIDF_VECTORS, cluster_type=TOPIC_CLUSTER,
                               max_samples=None, num_runs=20, vec_path: str=None):
    """
    Run the semi-supervised clustering experiments.

    This consists of:
        *
    """

    # A pseudorandom number generator is created,
    # then seeded, to ensure that the results are
    # replicable from run to run.
    r = random.Random(20)
    sorted_docs = sorted(guide.docs, key=lambda x: x.id)
    r.shuffle(sorted_docs)

    # Set the maximum number of samples to a default for
    # topics (20) or categories (50) if left unspecified
    if max_samples is None:
        max_samples = 20 if cluster_type == TOPIC_CLUSTER else 50


    # Load the spacy language model if we plan to
    # use the GloVe vectors.
    spacy_model = None
    if vector_type == GLOVE_VECTORS:
        spacy_model = spacy.load('/home2/rgeorgi/python3/lib/python3.4/site-packages/en_core_web_lg/en_core_web_lg-2.0.0/')

    w2v_model = None
    if vector_type == WORD2VEC_VECTORS:
        if os.path.splitext(vec_path)[1] in ['.bin', '.gz']:
            w2v_model = gensim.models.KeyedVectors.load_word2vec_format(vec_path, binary=True)
        else:
            w2v_model = Doc2Vec.load(vec_path)
            # w2v_model = gensim.models.Word2Vec.load(vec_path)

    # -------------------------------------------
    # Outer loop
    #
    # To make sure that the clustering results are not a fluke of picking a given couple documents
    # to seed the clusters, a number of runs are performed in which the order of the document set
    # is varied, so that different example docs will be chosen
    # -------------------------------------------
    for run_num in range(1, num_runs):

        ordered_docs = []
        true_labels = []

        # sorted_docs = sorted_docs[window_size:] + sorted_docs[:window_size]
        r.shuffle(sorted_docs)

        # Make sure that the order of documents
        # and their labels is static for evaluation.
        for doc in sorted_docs:
            ordered_docs.append(doc)
            if cluster_type == CATEGORY_CLUSTER:
                true_labels.append(doc.category.category_id)
            else:
                true_labels.append(doc.topic.id)

        if vector_type == TFIDF_VECTORS:
            matrix = tfidf_matrix(ordered_docs, nice_dir)
        elif vector_type == GLOVE_VECTORS:
            matrix = glove_matrix(ordered_docs, nice_dir, vector_dir, spacy_model, overwrite=overwrite)
        elif vector_type == WORD2VEC_VECTORS:
            matrix = gensim_matrix(ordered_docs, nice_dir, vector_dir, w2v_model, overwrite=overwrite)


        # -------------------------------------------
        # Iterate over a different number of supervised
        # samples
        # -------------------------------------------

        for supervised_samples in range(0, max_samples+1):

            # -------------------------------------------
            # Build the initial clusters
            # -------------------------------------------
            inits = init_cluster_dict(matrix.shape[1])
            samples_per_cluster = defaultdict(int)

            # Now, let's pick out some supervised
            # samples.
            for i, doc in enumerate(ordered_docs):
                topic_id = doc.topic.id
                category_id = doc.category.category_id

                label_key = category_id if cluster_type == CATEGORY_CLUSTER else topic_id

                if samples_per_cluster[label_key] <= supervised_samples:
                    v = matrix[i].toarray()[0,:] if not isinstance(matrix[i], np.ndarray) else matrix[i]
                    inits[label_key] += v
                    samples_per_cluster[label_key] += 1

            for key in inits:
                if samples_per_cluster[key] > 1:
                    inits[key] /= samples_per_cluster[key]

            # -------------------------------------------
            # Now, do the clustering.
            # -------------------------------------------

            # If no samples are used, seed the clusters randomly.
            # otherwise, use the generated init vectors.
            if supervised_samples == 0:
                init = 'random'
            else:
                init = np.array([v for v in inits.values()])

            # Set the number of clusters based on the number of clusters
            # defined in the guide
            num_clusters = len(g.categories) if cluster_type == CATEGORY_CLUSTER else len(g.topics)

            k = KMeans(n_clusters=num_clusters,
                       random_state=5,
                       init=init,
                       n_init=1,
                       )
            k.fit(matrix)

            rand_index = adjusted_rand_score(true_labels,
                                             k.labels_)

            # Finally, print out a CSV row for each iteration.
            csv = '{},{},{}'.format(run_num,
                                    supervised_samples,
                                    rand_index)
            print(csv)



def process_corpus(g: Guide, gigaword_dir: str, aq1_dir: str, nice_dir: str, overwrite=False):
    """
    Function to parse the guide, and extract the
    relevant
    :param g:
    :param nice_dir:
    :return:
    """
    filedict = defaultdict(set)

    # First, collect all the unique files that
    # contain articles we are interested in.
    for doc in g.docs:

        # Skip any files we've already processed
        if os.path.exists(doc.nice_path(nice_dir)) and not overwrite:
            continue

        # Check whether it exists in aquaint-1
        aq1_path = doc.aq1_path(aq1_dir)
        if os.path.exists(aq1_path):
            filedict[AQ1FileBase(doc.aq1_path(aq1_dir))].add(doc)
        else:
            filedict[GWFileBase(doc.gw_path(gigaword_dir))].add(doc)

    # Now, let's pull those files out so we only
    # have to unzip each file once.
    for corp_file in filedict.keys():

        remaining_docs = list(filedict[corp_file])

        assert isinstance(corp_file, BaseCorpFile)
        doc_contents = corp_file.get_stories(remaining_docs)

        for doc_content in doc_contents:
            doc_content.write(doc_content.doc.nice_path(nice_dir))







def init_cluster_dict(num_features: int):
    """
    Initialize an empty cluster
    """
    return defaultdict(lambda: np.zeros(shape=(num_features,)))

# -------------------------------------------
# MAIN
# -------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--vector-type', default=TFIDF_VECTORS, choices=[TFIDF_VECTORS, GLOVE_VECTORS, WORD2VEC_VECTORS, DOC2VEC_VECTORS],
                   help='Choose between tfidf or GloVe to represent documents as vectors')
    p.add_argument('--cluster-type', default=TOPIC_CLUSTER, choices=[TOPIC_CLUSTER, CATEGORY_CLUSTER],
                   help='Choose between running experiments for clustering topics or categories as defined in the TREC data')
    p.add_argument('-c', '--config',
                   default='config.yml', type=existsfile,
                   help='Path to the config file that specifies paths for the corpora')
    p.add_argument('-f', '--force', action='store_true',
                   help='Force overwrite of already-generated vectors.')
    p.add_argument('--runs', default=20, type=int,
                   help='Number of test runs to perform')
    p.add_argument('--max-samples', default=None, type=int,
                   help='Maximum number of supervised samples to use.')
    p.add_argument('-i', '--input', help='Input vector file, if using gensim', type=existsfile)

    args = p.parse_args()

    # -------------------------------------------
    # Do some error checking
    # -------------------------------------------
    if args.vector_type in [WORD2VEC_VECTORS, DOC2VEC_VECTORS] and not args.input:
        LOG.critical("An existing gensim vector path must be specified with -i/--input to use the gensim vector type.")
        sys.exit(3)



    config = load_config(args.config)

    g = Guide.load(config.get('topic-path'))
    process_corpus(g, config.get('gw-path'), config.get('aq1-path'), config.get('nice-path'))
    run_clustering_experiments(g, config.get('nice-path'), config.get('vector-path'),
                               vector_type=args.vector_type, cluster_type=args.cluster_type,
                               overwrite=args.force, max_samples=args.max_samples,
                               num_runs=args.runs,
                               vec_path=args.input)
