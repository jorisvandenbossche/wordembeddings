# wordembeddings


Overview of the content of this repo:

- I used the GloVe pre-trained word vectors. Those are explored and converted to a pickle file for local usage in [GloVe_pretrained_vectors.ipynb](GloVe_pretrained_vectors.ipynb).
- A scikit-learn compatible `Transformer` to convert text features to numerical vectors based on the wordembeddings is included [wordembeddings.py](wordembeddings.py).
- Applied on data
  - Semantic Textual Similarity (STS) Tasks: [sts_tasks.ipynb](sts_tasks.ipynb)
    - In this notebook we try to replicate the experimental results on textual similarity tasks from **A Simple but Tough-to-Beat Baseline for Sentence Embeddings** (Arora *et al*. 2017, https://openreview.net/pdf?id=SyK00v5xx)
    - This just calculates the similarity between two sentences based on the distance between the sentence embeddings (without applying supervised model)
    - Some different methods are used to convert the set of word embeddings (from the sentence) to a single sentence embedding (average, weighted averaged, with SVD reduction)
  - SICK similarity task: [sts_sick_supervized.ipynb](sts_sick_supervized.ipynb)
    - Similarity between two sentences, but now learn similarity with supervised model based on the sentence embeddings (instead of just the distance between the embeddings)
    - Implemented a Keras model similar to the one described in Wieting *et al.*, 2016 (https://arxiv.org/pdf/1511.08198.pdf)
    - Compared with scikit-learn using MLPRegressor, but don't get as good results
  - SICK entailment task: [sts_sick_entailment.ipynb](sts_sick_entailment.ipynb)
    - Classification problem: based on two sentences determine whether those are NEUTRAL, ENTAILMENT, or CONTRADICTION
    - Applied Keras model  similar to the one described in Wieting *et al.*, 2016, sklearn MLPClassifier using word embeddings and using TF-IDF.

