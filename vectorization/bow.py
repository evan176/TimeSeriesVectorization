#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import numpy
import spams
import logging


logger = logging.getLogger(__name__)

class BoWSp():
    """BoWSp Class
    This module provides a vectorization way to represent time series by bow
    model. Bag of words is a common technique in text minig. It provides a
    good result in document representation. We apply BoW model to represent
    time series data. Before it, we must preprocess raw series. The second
    technique we choose is sparse coding. "Spams" provides very efficiently
    dictionary learning packages. After converting series to tokens by this
    method, we can apply BoW model now!
    Attributes:
        w_len(int): the length of window size
        k(int): dictionary size in sparse coding
        lambda1(float): lambda conefficient in sparse coding
            |X-D*a|^2 + lambda*|a|^1
            For more information, see spams packages:
            http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams004.html#sec5
        interval(int): sliding window interval
        batch(bool): online learning or batch learning for sparse coding
            For more information, see spams packages:
            http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams004.html#sec5
        iter1(int): learning iterations
            For more information, see spams packages:
            http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams004.html#sec5
    """
    def __init__(self, w_len=20, k=100, lambda1=None,
                 interval=1, batch=False, iter1=-5):
        self.w_len = w_len
        self.k = k
        self.interval = interval
        self.batch = batch
        self.iter1 = iter1
        self.D = None
        if lambda1 == None:
            self.lambda1 = 1.0 / math.sqrt(w_len)
        else:
            self.lambda1 = lambda1
        # Log object information
        log_msg = "Initial BoWSp object: window length=%d, dictionary size=%d, \
                   lambda=%f, interval=%d" %(self.w_len, self.k, 
                                             self.lambda1, self.interval)
        logger.info(log_msg)

    def fit(self, X):
        """Learn dictionary from data
        Args:
            X(list): Each item is a list which contains a time series instance.
                ex: [[1, 2, 3, 0, 3, 2, 4, 5],
                     [0, 3, 2, 1, 5, 3, 1],
                     [9, 0, 4, 1, 8 ,4, 9, 3, 1, 2],
                     [8, 5, 2, 1, 8]]
        Returns:
            D(numpy 2d-array): learning dictionary from given time series data
                dimensions is w_len * k
        """
        segment_list = self.segment(X, self.w_len, self.interval)
        self.D = self.learn_D(segment_list, self.k, self.lambda1,
                              self.batch, self.iter1)
        return self.D

    def fit_transform(self, X):
        """Learn dictionary from data and represent it as bow model
        Args:
            X(list): Each item is a list which contains a time series instance.
                ex: [[1, 2, 3, 0, 3, 2, 4, 5],
                     [0, 3, 2, 1, 5, 3, 1],
                     [9, 0, 4, 1, 8 ,4, 9, 3, 1, 2],
                     [8, 5, 2, 1, 8]]
        Returns:

        """
        segment_list = self.segment(X, self.w_len, self.interval)
        self.D = self.learn_D(segment_list, self.k, self.lambda1,
                              self.batch, self.iter1)
        bow_representation = self.coding_series(segment_list, self.D,
                                                self.lambda1)
        return bow_representation

    def transform(self, X):
        """Represent data as bow model
        Args:
            X(list): Each item is a list which contains a time series instance.
                ex: [[1, 2, 3, 0, 3, 2, 4, 5],
                     [0, 3, 2, 1, 5, 3, 1],
                     [9, 0, 4, 1, 8 ,4, 9, 3, 1, 2],
                     [8, 5, 2, 1, 8]]
        Returns:

        """
        segment_list = self.segment(X, self.w_len, self.interval)
        bow_representation = self.coding_series(segment_list, self.D,
                                                self.lambda1)
        return bow_representation

    @staticmethod
    def get_index(data_len, w_len, interval):
        """Retrieve index with given length of data and size of window
        Args:
            data_len(int):
            w_len(int):
            interval(int):
        Returns:
            stamp_index(list): list contains location of sliding window in
                given series
            len_index(int): the number of current windows

        """
        # Initial timestamp index
        stamp_index = range(0, data_len-w_len, interval)
        # Get length of index
        len_index = len(stamp_index)
        # Log len_index for debug
        log_msg = "stamp index length: %d" %(len_index)
        logger.debug(log_msg)
        return stamp_index, len_index

    @staticmethod
    def segment(data, w_len=20, interval=1):
        """Segment series data to subsequences by sliding window with
        given length and same interval
        Args:
            data(list): each item is a time series instance
            w_len(int): legnth of window
            interval(int): size of interval
        Returns:
            segment_list(list): each item contains m subsequences which is
                sliced from original time series instance
        """
        segment_list = list()
        # Slice data segment to temp
        for i in range(len(data)):
            stamp_index, len_index = BoWSp.get_index(len(data[i]),
                                                     w_len, interval)
            temp = numpy.zeros([w_len, len_index])
            for count, j in enumerate(stamp_index):
                temp[:, count] = data[i][j:j+w_len]
            segment_list.append(temp)

        return segment_list

    @staticmethod
    def learn_D(segment_list, k, lambda1=None, batch=False, iter1=-5):
        """Learn dictionary from given series with input parameters
        Args:
            segment_list(list): each item contains m subsequences which is
                sliced from original time series instance
            k(int): size of dictionary
            lambda1(float): lambda conefficient in sparse coding,
                |X-D*a|^2 + lambda*|a|^1
                For more information, see spams packages:
                http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams004.html#sec5
            batch(bool): online learning or batch learning for sparse coding
                For more information, see spams packages:
                http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams004.html#sec5
            iter1(int): learning iterations
                For more information, see spams packages:
                http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams004.html#sec5
        Returns:
            D(numpy 2d-array): learning dictionary
        """
        # Horizontal train list
        temp = numpy.hstack(segment_list)
        if lambda1 == None:
            lambda1 = 1.0 / math.sqrt(temp.shape[0])
        # Log learning informatino
        log_msg = "learning dictionary with lambda: %f" %(lambda1)
        logger.info(log_msg)
        # Learn dictionary
        D = spams.trainDL(
                      numpy.asfortranarray(temp),
                      K=k, lambda1=lambda1, batch=batch,
                      iter=iter1, posAlpha=True
                  )
        return D

    @staticmethod
    def coding_series(segment_list, D, lambda1):
        """Represent given series with learned dictionary
        Args:
            segment_list(list): each item contains m subsequences which is
                sliced from original time series instance
            D(numpy 2d-array): learning dictionary
            lambda1(float): lambda conefficient in sparse coding,
                |X-D*a|^2 + lambda*|a|^1
                For more information, see spams packages:
                http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams005.html#sec15
        Returns:
            bow_data(numpy array): transformed data
        """
        k = D.shape[1]
        bow_data = numpy.zeros([len(segment_list), k])

        # BoW for data
        for index, item in enumerate(segment_list):
            # Log lasso information
            log_msg = "Solve lasso problem for %d series" %(index)
            logger.info(log_msg)
            code = spams.lasso(numpy.asfortranarray(item), D,
                               lambda1=lambda1, pos=True)
            code = numpy.sum(code.todense(), axis=1)
            bow_data[index:index+1, :] += code.reshape([1, k])
            div = numpy.linalg.norm(bow_data[index, :])
            if div > 0:
                bow_data[index, :] = bow_data[index, :] / div
            # Log bow result for debug
            log_msg = "%d series: " %(index)
            logger.debug(log_msg + str(bow_data[index, :]))
        return bow_data


if __name__ == "__main__":
    pass
