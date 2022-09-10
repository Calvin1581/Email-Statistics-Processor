'''naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
Calvin Whitley
CS 251 Data Analysis Visualization, Fall 2021
'''
import numpy as np


class NaiveBayes:
    '''Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any
     number of classes)'''
    def __init__(self, num_classes):
        '''Naive Bayes constructor

        TODO:
        - Add instance variable for `num_classes`
        '''
        self.num_classes = num_classes
        # class_priors: ndarray. shape=(num_classes,).
        #   Probability that a training example belongs to each of the classes
        #   For spam filter: prob training example is spam or ham
        self.log_class_priors = None
        # class_likelihoods: ndarray. shape=(num_classes, num_features).
        #   Probability that each word appears within class c
        self.log_class_likelihoods = None

    def train(self, data, y):
        '''Train the Naive Bayes classifier so that it records the "statistics" of the training set:
        the log of the class priors (i.e. how likely an email is in the training set to be spam or ham?) and the
        log of the class likelihoods (the probability of a word appearing in each class — spam or ham)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        TODO:
        - Compute the instance variables self.log_class_priors and self.log_class_likelihoods needed for
        Bayes Rule. See equations in notebook.
        '''
        num_samps, num_features = data.shape
        uniques, counts = np.unique(y,return_counts=True)
        self.log_class_priors = np.ndarray(counts.size)
        for i in range(counts.size):
            self.log_class_priors[i] = (counts[i]/num_samps)
        self.log_class_priors = np.log(self.log_class_priors)
        
        
        self.log_class_likelihoods = np.ndarray((counts.size,num_features))
        for x in range(uniques.size):
            nc = 0
            for z in range(num_features):
                ncw = 0
                for f in range(num_samps):
                    if y[f] == x:
                        ncw += data[f,z]
                        nc += data[f,z]          
                self.log_class_likelihoods[x,z] = ncw + 1
            for b in range(num_features):
                self.log_class_likelihoods[x,b] = self.log_class_likelihoods[x,b]/(nc + num_features)

        self.log_class_likelihoods = np.log(self.log_class_likelihoods)

    def predict(self, data):
        '''Combine the class likelihoods and priors to compute the posterior distribution. The
        predicted class for a test sample from `data` is the class that yields the highest posterior
        probability.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each test data sample.

        TODO:
        - For the test samples, we want to compute the log of the posterior by evaluating
        the the log of the right-hand side of Bayes Rule without the denominator (see notebook for
        equation). This can be done without loops using matrix multiplication or with a loop and
        a series of dot products.
        - Predict the class of each test sample according to the class that produces the largest
        log(posterior) probability (hint: can be done without loops).

        NOTE: Remember that you are computing the LOG of the posterior (see notebook for equation).
        NOTE: The argmax function could be useful here.
        '''
        num_test_samps, num_features = data.shape
        predictions = np.ndarray(num_test_samps,dtype=int)
        for i in range(num_test_samps):
            probs = np.ndarray(self.log_class_priors.size)
            for m in range(self.log_class_priors.size):
                n = 0
                for j in range(num_features):
                    s = data[i,j]*self.log_class_likelihoods[m,j]
                    n += s
                n += self.log_class_priors[m]
                probs[m] = n
            predictions[i] = np.argmax(probs)
        return predictions


    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''
        n = y_pred.shape[0]
        flot = np.sum(y_pred == y)/n
        # print(flot)
        # o = 0
        # for i in range(n):
        #     if y[i] == y_pred[i]:
        #         o += 1
        return flot

    def confusion_matrix(self, y, y_pred):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).

        Recall: the rows represent the "actual" ground truth labels, the columns represent the
        predicted labels.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_samps,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        ndarray. shape=(num_classes, num_classes).
            Confusion matrix
        '''
        # To get the number of classes, you can use the np.unique
        # function to identify the number of unique categories in the
        # y matrix.

        c = np.zeros((self.num_classes,self.num_classes))
        for a,p in zip(y,y_pred):
            c[int(a),p] += 1

        # for i in range(self.num_classes):
        #     for j in range(self.num_classes):
        #         c[i,j] = np.count_nonzero(np.and(y==i,y_pred==j))
        
        # for i in range(y.size):
        #     c[y[i],y_pred[j]] += 1
        
        return c

