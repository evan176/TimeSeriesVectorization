import os
import sys
import math
import numpy
import spams
    
class BoW_sp():
    def __init__(self, w_len=20, k=100, a=None, interval=1, batch=False, iter=-5):
        self.w_len = w_len
        self.k = k
        self.interval = interval
        self.batch = batch
        self.iter = iter
        if a == None:
            self.a = 1.0 / math.sqrt(w_len)
        else:
            self.a = a
        
    def fit(self, X):
        segment_list = self.segment(X, self.w_len, self.interval)
        self.D = self.learn_D(segment_list, self.k, self.a, self.batch, self.iter)
        return self.D
    
    def fit_transform(self, X):
        segment_list = self.segment(X, self.w_len, self.interval)
        self.D = self.learn_D(segment_list, self.k, self.a, self.batch, self.iter)
        bow_representation = self.coding_series(segment_list, self.D, self.a, self.batch, self.iter)
        return bow_representation
    
    def transform(self, X):
        segment_list = self.segment(X, self.w_len, self.interval)
        bow_representation = self.coding_series(segment_list, self.D, self.a, self.batch, self.iter)
        return bow_representation
      
    #def get_params(self, ):
    
    #def set_params(self, **params):
      
    @classmethod
    def get_index(self, data_len, w_len, interval):
        # Initial timestamp index
        stamp_index = range(0, data_len - w_len, interval)
        # Get length of index
        len_index = len(stamp_index)
        return stamp_index, len_index
        
    @classmethod
    def segment(self, data, w_len=20, interval=1):

        segment_list = list()
        
        # Slice data segment to temp
        for i in range(len(data)):
            stamp_index, len_index = self.get_index(len(data[i]), w_len, interval)
            temp = numpy.zeros([w_len,  len_index])
            for count, j in enumerate(stamp_index):
                temp[:, count] = data[i][j:j+w_len]
            segment_list.append(temp)
            
        return segment_list

    @classmethod
    def learn_D(self, segment_list, k, a=None, batch=False, iter=-5):
               
        # Horizontal train list
        temp = numpy.hstack(segment_list)
        if a == None:
            a = 1.0 / math.sqrt(temp.shape[0])
        # Learn dictionary
        D = spams.trainDL(
                      numpy.asfortranarray(temp), 
                      K=k, lambda1=a, batch=batch, 
                      iter=iter, posAlpha=True
                  )
        return D

    @classmethod
    def coding_series(self, segment_list, D, a=None, batch=False, iter=-5):
        k = D.shape[1]
        bow_data = numpy.zeros([len(segment_list), k])

        # BoW for data
        for index, item in enumerate(segment_list):
            code = spams.lasso(numpy.asfortranarray(item), D, lambda1=a, pos=True)
            code = numpy.sum(code.todense(), axis=1)
            bow_data[index:index+1, :] += code.reshape([1, k])
            div = numpy.linalg.norm(bow_data[index, :])
            if div > 0:
                bow_data[index, :] = bow_data[index, :] / div
        return bow_data
