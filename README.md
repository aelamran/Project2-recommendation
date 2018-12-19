# Project2-recommender system @EPFL

**Team Members:**
<br />
Ayyoub EL Amrani (ayyoub.elamrani@epfl.ch)
<br />
Fatine Benhsain (fatine.benhsain@epfl.ch)
<br />
Tabish Qureshi (tabish.qureshi@epfl.ch)

**Instructions to Run:**
<br />

* Besire usual packages, spotlight package have to be installed using the following command line :
```
conda install -c maciejkula -c pytorch spotlight=0.1.5
```
* To reproduce the results of our code and create a the submission results, type **'python3 run.py'**. This generates our predictions by going through matrix factorization using pytorch layers and adam optimization algorithm. 



**Important Files**
* 'run.py' contains all of our code for parsing command line arguments and running the selected models. If no choice of models is given, the default option is to run all 8 methods and generate predictions. This file also performs the operations to read in training and testing data and generate splits on the training dataset.

* 'helpers_pj.py' contains all of the helpers provided to us for this project. 

* The 'models' directory contains each of the following files:

	* 'adam.py' contains the implementations of adam optimization algorithm and generates 
	* 'als.py' contains the implementation and generates predictions for the Alternating Least Squares algorithm
	* 'sgd.py' contains the implementation and generates predictions for the Stochastic Gradient Descent algorithm


**Libraries:**

We have used the following libraries to implement our methods:
* [Numpy](http://www.numpy.org/)
* [Scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html)
* [Sklearn](http://scikit-learn.org/stable/)
* [Spotlight]https://maciejkula.github.io/spotlight/