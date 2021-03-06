"""Base and mixin classes for instance reduction techniques"""
# Author: Dayvid Victor <dvro@cin.ufpe.br>
# License: BSD Style
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors.classification import KNeighborsClassifier

from sklearn.utils import safe_asarray, atleast2d_or_csr, check_arrays
from sklearn.externals import six


class InstanceReductionWarning(UserWarning):
    pass

# Make sure that NeighborsWarning are displayed more than once
warnings.simplefilter("always", InstanceReductionWarning)


class InstanceReductionBase(six.with_metaclass(ABCMeta, BaseEstimator)):

    """Base class for instance reduction estimators."""

    @abstractmethod
    def __init__(self):
        pass


class InstanceReductionMixin(InstanceReductionBase, ClassifierMixin):

    """Mixin class for all instance reduction techniques"""


    def set_classifier(self):
        """Sets the classified to be used in the instance reduction process
            and classification.

        Parameters
        ----------
        classifier : classifier, following the KNeighborsClassifier style
            (default = KNN)

        y : array-like, shape = [n_samples]
            Labels for X.

        Returns
        -------
        P : array-like, shape = [indeterminated, n_features]
            Resulting training set.

        q : array-like, shape = [indertaminated]
            Labels for P
        """

        self.classifier = classifier


    def reduce_data(self, X, y):
        """Perform the instance reduction procedure on the given training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training set.0

        y : array-like, shape = [n_samples]
            Labels for X.

        Returns
        -------
        X_ : array-like, shape = [indeterminated, n_features]
            Resulting training set.

        y_ : array-like, shape = [indertaminated]
            Labels for X_
        """
        pass
    
    def get_prototypes(self):
        return self.X_, self.y_

    def fit(self, X, y, reduce_data=True):
        """
        Fit the InstanceReduction model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
            Note that centroid shrinking cannot be used with sparse matrices.
        y : array, shape = [n_samples]
            Target values (integers)
        reduce_data : bool, flag indicating if the reduction would be performed
        """
        self.X = X
        self.y = y
        self.labels = set(y)
        self.prototypes = None
        self.prototypes_labels = None
        self.reduction_ratio = 0.0

        if reduce_data:
            self.reduce_data(X, y)

        return self

    def predict(self, X, n_neighbors=1):
        """Perform classification on an array of test vectors X.

        The predicted class C for each sample in X is returned.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]

        Notes
        -----
        The default prediction is using KNeighborsClassifier, if the
        instance reducition algorithm is to be performed with another
        classifier, it should be explicited overwritten and explained
        in the documentation.
        """
        X = atleast2d_or_csr(X)
        if not hasattr(self, "X_") or self.X_ is None:
            raise AttributeError("Model has not been trained yet.")

        if not hasattr(self, "y_") or self.y_ is None:
            raise AttributeError("Model has not been trained yet.")

        if self.classifier == None:
            self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

        self.classifier.fit(self.X_, self.y_)
        return self.classifier.predict(X)


    def predict_proba(self, X):
        """Return probability estimates for the test data X.
        after a given prototype selection algorithm.
    
        Parameters
        ----------
        X : array, shape = (n_samples, n_features)
            A 2-D array representing the test points.
        
        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
        of such arrays if n_outputs > 1.
        The class probabilities of the input samples. Classes are ordered
        by lexicographic order.
        """
        self.classifier.fit(self.X_, self.y_)
        return self.classifier.predict_proba(X)

