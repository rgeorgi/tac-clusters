# TAC Semi-Supervised Cluster Experiments

This code gives an example of how a task, in this case document clustering, may be improved using varying amounts of supervised data.

## Experiment Description

Using the AQUAINT and English Gigaword corpora from the [TAC 2009 Summarization Track](https://tac.nist.gov//2009/Summarization/), document clustering is performed on:

* **5** categories
* **46** topic clusters (docsets)

This code uses the documents named as belonging to a particular topic or category by the TAC guide `*_topics.xml` files. With a number of randomized runs, the code shows the effect of varying the number of sample documents used to create "seed" centroids for each [*k*-means](https://en.wikipedia.org/wiki/K-means_clustering) document cluster, from **0** (random initialization), **1** (single document) or any integer up to the maximum number of documents in each topic or category.

The result of the *k*-means algorithm is then compared against the true labels for the documents provided by the `*_topics.xml` file.

The results of running these experiments on 20 random initializations on the devtest data are shown in the graphs below, where the red refers to representing documents as tf*idf vectors using [scikit-learn](http://scikit-learn.org/), and the blue using pre-trained GloVe embeddings via [spacy](https://spacy.io).

## Results

![](doc/images/topics.png)

![](doc/images/categories.png)