eriesVectorization
This toolbox provides some time series vectorization methods which could give better representation for classification / clustering or other analysis.

## BoWSp
Bag of words is a common technique in text mining for document representation. Since this method shows good result in computer vision, it is also used to represent time series data. The subsequences from raw series were extracted as local patterns for learning codebook. Consequently, a time series data instance is encoded by the codebook, which describes different local patterns of time series data. With the learned codebook, each original time series data instance could be represented by BoW histogram.

Requires:
+  Numpy == 1.8
+  SPArse Modeling Software(http://spams-devel.gforge.inria.fr/downloads.html)
