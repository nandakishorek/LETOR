# Linear regression - Learning to Rank using Microsoft LETOR

Search engines have become increasingly relevant when it comes to our
daily lives. Whether we want to search for latest news or flight itinerary,
we just search it on google, bing or yahoo.
To do this search engines have to display the most relevant results on the first
few pages. The relevancy depends on a lot of parameters, which we call
features. In this project, we use the benchmark dataset - QueryLevelNorm
version of LETOR 4.0 package provided by Microsoft. This dataset contains
69923 vectors of 46 dimensions each. Each dimension corresponds to a
feature namely IDF of terms in the body, anchor, title, etc., stream length of
the body, anchor, title, etc.

#####MODEL:
We model the linear regression using M basis functions, where M is the
complexity of the model. The basis functions are non-linear functions of the
input variables. So these basis functions bring non-linearity to our model with
the d input variables, where d is the number of features in our input dataset.

######1. Choice of the basis function:
We overcome the limitations of a single polynomial basis function, the
function being global and changes in one region affecting other regions, by
choosing M basis functions for different regions. One such basis function is
Gaussian Radial Basis Function (RBF). We choose M Gaussian kernels for our input data. 

######2. Partitioning the data set:
The input data consists of 69623 vectors of 46 dimensions each. This dataset
was partitioned into three sets namely training set, validation set and test set.
Training set – 80% = 55700 samples.
Validation set – 10% = 6961 samples.
Test set – 10% = 6961 samples.
The training set is used to learn the weights or the parameters.
The validation set is used to validate the learned weights and the test set is
used to test how the learned weights generalizes.

######3. The spread of each basis function:
To choose the spread for each of the basis functions the histogram of the
whole dataset was taken. We see that most of the points have frequency peaks between 0 and 0.1
So we choose 0.1 as the spread for each of the basis functions.

######4. Centers for the basis functions:
We decided to use M basis functions instead of one polynomial function, by
dividing the feature space into M regions and applying M basis functions on
the input vectors.
We have to determine the centers for each of these Gaussians kernels.
To do this we divide the space into M clusters. These clusters are found using
the k-means algorithm.
The centroid or the mean of one cluster is taken as the center for one basis
function. That way we have M centroids forming the centers of these basis
functions.

######5. Model complexity and lambda:
We find the weights or the parameters using the closed form maximum
likelihood solution.
