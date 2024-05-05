# DawgsML
A machine learning package for models and utilities.  
This project initially started as a simple assignment for an undergraduate course in AI at Butler University, and that  
is why is named Dawgs.  
For this assignment we had to build a simple perceptron to be used for a binary classification of a non-linearly  
separable problem, without using any packages other than the Python standard library.  

Here there is a data structure for tabular data representation that it kind of simulates pandas, with file I/O and tons  
of methods to be used for data preprocessing like remove/add columns or rows, shuffling, column type conversion,  
but also operations on data structure itself like iterations, indexing, etc.  
This data structure is used in all the machine learning algorithms and utilities, and is meant to give to user a package  
that do not rely on any other dependencies.  

Still you can access the data as list by calling the proper .data property on the columns so that is still possible to  
use graphic libraries such as matplotlib or seaborn.  
The purpose of this project was not to build an efficient tabular data structure. In fact, this is nowhere even close  
to pandas efficiency, and literally any efforts was made to optimize any of the methods of the Dataframe data structure.  
However, contributors, are very welcome to have their hand dirty by using this as a playground.

AS of now, there is only one ML algorithm, a Perceptron, and a bunch of ML utilities like splitting, k-Fold, and a  
grid-search, but more will be added soon.  



