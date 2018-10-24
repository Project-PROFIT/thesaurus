"""
Functions for comparison
"""
import numpy as np
import scipy
from scipy.linalg import norm
from scipy.sparse import coo_matrix, csr_matrix
import pickle
import logging
import json
from time import time
from collections import defaultdict, Counter
from functools import lru_cache

from thesaurus.thesaurus import Thesaurus, get_sim_dict


root_logger = logging.getLogger('root')


def permute_and_sparsify_dense_matrix(matrix,
                                      permutation,
                                      sparse_class=csr_matrix):

    sparse_m = coo_matrix(matrix)
    size = matrix.shape
    new_data = np.array([matrix[permutation[sparse_m.row[i]],
                        permutation[sparse_m.col[i]]]
                        for i in range(len(sparse_m.data))])
    sparse_m.data = new_data

    return sparse_class(sparse_m, shape=size)


def soft_squared_norm(d1, features_similarity, outer_fun):
    return (features_similarity.multiply(outer_fun(d1, d1))).sum()


def soft_projection(d1, d2, features_similarity, outer_fun):
    return (features_similarity.multiply(outer_fun(d1, d2))).sum()


def soft_cosine(d1, d2, features_similarity):
    """
    Compare issues using soft cosine.

    :param d1: Vector representing first document
    :param d2: Vector representing second document
    :param features_similarity: A matrix whose i,j entry returns similarity
            between freatures i and j
    :return: float: similarity measure soft cosine (search Wikipedia)
    """
    start = time()
    outer_fun = lambda x, y: np.outer(x, y)
    if scipy.sparse.isspmatrix(d1):
        outer_fun = lambda x, y: x.T.dot(y)
    numerator = soft_projection(d1, d2, features_similarity, outer_fun)
    root_logger.debug('numerator done, time: {:0.3f}'.format(time() - start))
    denominator = np.sqrt(soft_squared_norm(d1, features_similarity, outer_fun)
                          * soft_squared_norm(d2, features_similarity, outer_fun))
    root_logger.debug('denom done, time: {:0.3f}'.format(time() - start))
    if denominator != 0:
        ans = numerator / denominator
        return ans
    else:
        return 0


class SoftCosineCombinedSimilarity:
    def __init__(self,
                 sim_dict_path=None,
                 the=None,
                 **kwargs):

        """
        :param sim_dict_path:
        :param the: Thesaurus
        """

        def make_i2i(from_, to_):
            # A mapping from all the concepts in from_ to those in to_
            assert len( set(to_) & set(from_) ) == 0
            k2i1 = {k: i for i, k in enumerate(from_)}
            k2i2 = {k: i for i, k in enumerate(to_)}
            i2i = {k2i1[k]: k2i2[k] for k in k2i1}
            return i2i

        self.sim_dict, self.all_cpts = get_sim_dict(
            sim_dict_path=sim_dict_path, the=the
        )
        if not isinstance(self.sim_dict, scipy.sparse.csr_matrix):
            self.sim_dict = scipy.sparse.csr_matrix(self.sim_dict)
        self.all_cpts = [str(x) for x in self.all_cpts]
        self.cpt_inds = {cpt: i for i, cpt in enumerate(self.all_cpts)}

    def compute(self, v1, v2):
        """

        :param v1: (concept array, term-multi-set) of document 1
        :param v2: (concept array, term-multi-set) of document 2
        :return:
        """
        start = time()

        cpt_vect1, termset1 = v1
        cpt_vect2, termset2 = v2

        cpt_sim = self.compute_cpts(cpt_vect1, cpt_vect2)

        term_sim_den = sum((termset1 | termset2).values())
        if term_sim_den == 0:
            term_sim = 0
        else:
            term_sim_num = sum((termset1 & termset2).values())
            term_sim = term_sim_num / term_sim_den
        root_logger.info(
            'Terms done, time: {:0.3f}'.format(time() - start))

        return cpt_sim, term_sim

    def compute_cpts(self, cpt_vect1, cpt_vect2):
        start = time()

        cpt_sim = soft_cosine(cpt_vect1, cpt_vect2, self.sim_dict)
        root_logger.debug(
            'Cpts done, time: {:0.3f}'.format(time() - start))
        return cpt_sim

    def transform_cpts(self, cpt_dict):
        """
        :param cpt_dict: {cpt_uri: cpt_freq} all cpt_uris should be contained in self.all_cpts
        :return: scipy.sparse.csr_matrix
        """
        vect = np.zeros(len(self.all_cpts))
        for cpt_uri, cpt_freq in cpt_dict.items():
            cpt_ind = self.cpt_inds[cpt_uri]
            vect[cpt_ind] = cpt_freq
        vect = scipy.sparse.csr_matrix(vect)
        return vect

    @staticmethod
    def transform_terms(term_dict):
        """
        :param term_dict: {term: term_freq}
        :return: collections.Counter
        """
        termset = Counter(term_dict)
        return termset


if __name__ == '__main__':
    pass
