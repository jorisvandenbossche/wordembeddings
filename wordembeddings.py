import six

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD


def _get_word_freq(token):
    import wordfreq
    return wordfreq.word_frequency(token, 'en', wordlist='large') 


class EmbeddingVectorizer(CountVectorizer):
    """Convert a collection of text documents to a matrix of sentence
    embeddings.
    
    Parameters
    ----------
    word_vectors : dict
        Mapping of token -> word vector
    weighted : boolean, default False
        Whether to take a weighted average of the word vectors
    R : boolean, default False
        Whether to remove the common component
        
        
    TODO: 
         - this implementation currently still takes as record a list of
           already tokenized words, instead of one string that needs to be
           tokenized (as the other text Vectorizers do)
    
    """
        
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 word_vectors=None, weighted=False, R=False):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        
        self.word_vectors = word_vectors
        self.weighted = weighted
        self.R = R
        
        self.ngram_range = (1, 1)
        self.analyzer='word',
 
    def _sentence_vectors_avg(self, raw_documents):
        """This just calculates the sentence vector as
        the (weighted) mean of the word vectors"""
        
        #analyze = self.build_analyzer()
        
        values = []
        
        for doc in raw_documents:
            
            word_vecs = []
            
            #for token in analyze(doc):
            for token in doc:
                #TODO integrate this with analyzer
                token = token.lower()
                try:
                    vw = np.array(self.word_vectors[token])
                    if self.weighted:
                        freq = _get_word_freq(token) 
                        vw *= 1e-3 / (1e-3 + freq)
                    word_vecs.append(vw)
                except KeyError:
                    # Ignore out-of-vocabulary items for fixed_vocab=True
                    continue
                    
            vs = np.array(word_vecs).mean(axis=0)
            
            values.append(vs)
        
        return np.array(values)
    
    
    def _calculate_singular_vector(self, X):
        
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0).fit(X)
        u = svd.components_[0]
        
        self.singular_vector_ = u
    
    def _remove_common_component(self, X):
        
        u = self.singular_vector_
        
        return X - np.dot(X, np.outer(u, u))


    def fit_transform(self, raw_documents, y=None):
        """Learn the vocabulary dictionary and return term-document matrix.
        This is equivalent to fit followed by transform, but more efficiently
        implemented.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        Returns
        -------
        X : array, [n_samples, n_features]
            Document-term matrix.
        """
        # We intentionally don't call the transform method to make
        # fit_transform overridable without unwanted side effects in
        # TfidfVectorizer.
        if isinstance(raw_documents, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")

        X = self._sentence_vectors_avg(raw_documents)
        
        if self.R:
            self._calculate_singular_vector(X)
            X = self._remove_common_component(X)

        return X

    def transform(self, raw_documents):
        """Transform documents to document-term matrix.
        Extract token counts out of raw text documents using the vocabulary
        fitted with fit or the one provided to the constructor.
        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Document-term matrix.
        """
        if isinstance(raw_documents, six.string_types):
            raise ValueError(
                "Iterable over raw text documents expected, "
                "string object received.")


        # use the same matrix-building strategy as fit_transform
        X = self._sentence_vectors_avg(raw_documents)
        
        if self.R:
            # now use the previously calculated singular vector
            X = self._remove_common_component(X)

        return X
