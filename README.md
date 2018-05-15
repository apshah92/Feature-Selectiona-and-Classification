# Feature-Selectiona-and-Classification
This is a machine learning application I developed in Fall 2017. The program works with 8000 rows of trianing data each having about 30000 feature columns. The goal of the application is to use it for classification of test data of 2000 rows of data.
It first runs a feature selection algorithm using pearson correlation method and selects 15 most relevant features and reshapes training data with these 15 columns.
It then runs a classfication algorithm using Linear SVC model from SciKit Learn module svm. 
Once the model is trained we can use it for classification of test data. The program has been tested on 2000 rows of data and achieves 63-64% accuracy. accuracy is measured using 10 fold cross validation method.
The data used here is UCI dataset of single nucleotide polymorphism (SNP) genotype data 
containing 29623 SNPs (total features).
The training dataset can be accessed via this link :https://web.njit.edu/~usman/courses/cs675_fall17/traindata.gz
For test data: https://web.njit.edu/~usman/courses/cs675_fall17/testdata.gz
