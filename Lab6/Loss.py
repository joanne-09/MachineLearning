import numpy as np

def compute_CCE_loss(AL, Y):
    n = Y.shape[0]

    # Compute loss from aL and y.
    ### START CODE HERE ### (≈ 1 line of code)
    loss = -np.sum(Y * np.log(AL + 1e-5)) / n
    ### END CODE HERE ###

    loss = np.squeeze(loss)  # To make sure your loss's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (loss.shape == ())

    return loss
    
# compute_MSE_loss (MSE)
def compute_MSE_loss(AL, Y):
    m = Y.shape[0]
    loss = (1 / m) * np.sum(np.square(AL - Y))
    return loss