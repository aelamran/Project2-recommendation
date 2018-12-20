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

Make sure to create a new environment to avoid any problems. (using Python 3.6.7).
* Spotlight package have to be installed using the following command line :
```
conda install -c maciejkula -c pytorch spotlight=0.1.5
```
* You will also need to install pandas:
```
pip install pandas
```

* To reproduce the results of our code and create a the submission results, type **'python3 run.py'**. This generates our predictions by going through matrix factorization using pytorch layers and adam optimization algorithm. 



**Important Files**
* 'run.py' contains all of our code for parsing command line arguments and running the selected models. If no choice of models is given, the default option is to run all 8 methods and generate predictions. This file also performs the operations to read in training and testing data and generate splits on the training dataset.



* Our project contain the following files:

	* 'helpers_pj.py' contains all of the helpers provided to us for mainly SGD and ALS. 
	* 'mf_spotlight_model.py' contains the all the useful implementations to run adam optimization algorithm using the __Spotlight__ library 
	* 'ALS.py' contains the implementation and generates predictions for the Alternating Least Squares algorithm
	* 'SGD.py' contains the implementation and generates predictions for the Stochastic Gradient Descent algorithm
	* 'run.py' contains the implementation of our main file that runs and make the submission of our model
	* 'report.pdf' contains our project report

* And the folder `data` containing :
	
	* 'data_train.csv'      : the training csv file provided by the challenge.
	* 'data_test.csv'       : the test csv file provided by the challenge.
	* 'final_submission.csv': our final submission giving our crowdAI score.

** Final score **
The final score obtained was 1.032, the submission id on CrowdAI is : 25099 and the username is : FatineB.
**Libraries:**

We have used the following libraries to implement our methods:
* [Numpy](http://www.numpy.org/)
* [Scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html)
* [Sklearn](http://scikit-learn.org/stable/)
* [Spotlight](https://maciejkula.github.io/spotlight/)