import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    num_trains, dim = X.shape
    _, num_classes = W.shape
    for i in range(num_trains):
        scores = np.dot(X[i], W) # 1,C Matrix prediction of X[i] over the different classes
        scores -= np.max(scores) # Numeric stability - shift the values inside the scores vector so that the highest value is zero
        probabilities = np.exp(scores) / np.sum(np.exp(scores)) # 1,C / 1 = 1,C
        loss += -np.log(probabilities[y[i]])
        
        probabilities[y[i]] -= 1 # calculate p-1 and later we'll put the negative back
        for j in range(num_classes):
            dW[:, j] += X[i] * probabilities[j] # 1,C * scalar = 1,C
    
    loss /= num_trains
    dW /= num_trains
    
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    num_trains, dim = X.shape
    _, num_classes = W.shape
    
    scores = np.dot(X, W) # N,C Matrix prediction of X[i] over the different classes
    scores -= np.max(scores, axis=1, keepdims=True) # Numeric stability N,1
    probabilities = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True) # N,C / C = N,C
    loss = np.sum(-np.log(probabilities[np.arange(num_trains), y]))
        
    probabilities[np.arange(num_trains), y] -= 1 # calculate p-1 and later we'll put the negative back
    dW = np.dot(X.T, probabilities) # D,N * N,C = D,C
    
    loss /= num_trains
    dW /= num_trains
    
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

    return loss, dW
