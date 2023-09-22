# Lab 1: Linear Regression (corresponding to lecture handout 1)
import numpy as np

# This function computes the polynomial of order 'order' corresponding to a least-squares fit
# to the data (x, y), where 'y' contains the observed values and 'x' contains the x-coordinate
# of each observed value.
#
# The normal equation is solved in the function 'linear regression'.
def LS_poly(x, y, order, eps = 0):
    # First build the polynomial design matrix (relies only x-ordinates, not observed values)
    X = polynomial_design_matrix(x, order);
    # Then find the polynomial using this matrix and the values 'y'.
    w = linear_regression(X, y, eps=eps);
    return w

# Computes the polynomial design matrix.
#
# For a vector 'x', this contains all powers up to 'order'
# of each element of 'x'.  This kind of matrix is also called
# a Vandermonde matrix.
#
# The numpy array 'x' contains the x-ordinates (x-axis
# values) which we are analyzing.
def polynomial_design_matrix(x, order=1):
    # Create a matrix of zeros, with 'length-of-x' rows and 'order+1' cols
    X = np.zeros(shape=(x.size,order+1))

    # EXERCISE 1: fill the body of this function.
    # See slide 26 of the lecture 1 handout.
    # The exponentiation (power) operator in Python is '**'.
    # Assign to the element (row,col) of a numpy matrix with: M[r,c] = <expression>

    # Hint:
    # Outer loop: iterating over columns; each column gets a higher power
    # for p in range(0, order+1):
    # Inner loop: iterating over rows: each row corresponds to an element of 'x'
    # for i in range(x.size):
    # Element (i,p) of X should be the ith element of 'x' to the power p:
    for p in range(0, order+1):
        for i in range(x.size):
            X[i, p] = x[i]**p

    return X


# Given values 'y' and the polynomial design matrix for the x-ordinates of those
# values in 'X', find the polynomial having the best fit:
#
# theta = ((X'X + I)^(-1))*X'y
#
# This uses numpy to solve the normal equation (see slide 16 of handout 1)
def linear_regression(X, y, eps=0):
    order = X.shape[1] - 1;
    M = np.dot(X.transpose(), X)

    # EXERCISE 2: implement Tikhonov regularisation.
    # See lecture handout 1, slide 38.
    print("Eps: " + str(eps))
    #
    # <add 'eps' times the identity matrix to M>
    # Hints:
    # There is a function 'identity' in numpy to generate an identity matrix
    # The 'identity' function takes an integer parameter: the size of the (square) identity matrix
    # The shape of a numpy matrix 'A' is accessed with 'A.shape' (no parentheses); this is a tuple
    # The number of rows in a matrix 'A' is then 'A.shape[0]' (or 'len(A)')
    # You can add matrices with '+' -- so you will update 'M' with 'M = M + <amount> * <identity>'
    # Note that the amount of regularization is denoted 'alpha' in the slides but here it's 'eps'.

    identity_matrix = np.identity(X.shape[1])

    M = M + eps*identity_matrix

    theta = np.dot(np.linalg.inv(M), np.dot(X.transpose(), y))
    return theta

# EXERCISE 3: implement computation of mean squared error between two vectors
def mean_squared_error(y1, y2):
    # You can use '-' to compute the elementwise difference of numpy vectors (i.e. y1 - y2).
    # You can use '**' for elementwise exponentiation of a numpy vector.
    # You can use the numpy function 'mean' to compute the mean of a vector.
    vector_diff = y1 - y2

    vector_diff_squred = vector_diff**2

    sum_of_squared_diff = sum(vector_diff_squred)

    mse = sum_of_squared_diff / len(y1)

    return mse  # replace this with your answer.

# EXERCISE 4: return the number of the best order for the supplied
# data (see the notebook).
def question_4():
    # Ok so to get the best order for the supplied data, we need to:
    # TODO: use `Lab 1.ipynb` to work and actually find the best fit order, then just hard code it back to here

    # Note: this was just got by observing the graphs in the notebook and finding the lowest loss on the test set
    # and the corresponding polynomial order
    return 3


def q4_data():
    # Load the data
    data = np.loadtxt('lab_1.data', delimiter=',')

    # Split it into two equal sets, for training and test.
    num_points = data.shape[1]
    set_size = num_points // 2
    training_x = data[0, 0:set_size]
    training_y = data[1, 0:set_size]
    test_x = data[0, set_size:]
    test_y = data[1, set_size:]
    return training_x, training_y, test_x, test_y

def print_q4_data() -> None:
    trainingx, trainingy, testx, testy = q4_data()
    print(f"trainingx: {trainingx}")
    print(f"tx: {testx}")
    print(f"ty: {testy}")
    print(f"trainingx.shape: {trainingx.shape}")
    print(f"tx.shape: {testx.shape}")
    print(f"ty.shape: {testy.shape}")


def q5_data():
    pass


def main():
    pass

if __name__ == '__main__':
    main()
