import random
import time
import numpy as np
import matplotlib.pyplot as plot


# Function to swap two given row indices of a matrix
def m_row_swap(matrix, r1, r2):
    if len(matrix) > r1 and len(matrix) > r2:
        for i in range(len(matrix[0])):
            temp = matrix[r1][i]
            matrix[r1][i] = matrix[r2][i]
            matrix[r2][i] = temp


# Function to swap two given indices of a vector
def v_row_swap(b, r1, r2):
    if len(b) > r1 and len(b) > r2:
        temp = b[r1]
        b[r1] = b[r2]
        b[r2] = temp


# A function to determine if the provided matrix is symmetric
def is_symmetric(matrix):
    if len(matrix) != len(matrix[0]):
        print("ERROR: NON-SQUARE MATRIX")
        exit(-1)
    n = len(matrix)
    for i in range(n):
        for j in range(i+1, n):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True


# A function to carry out the cholesky method to see if a given matrix is positive definite
#   Time complexity = O(n³)
def cholesky_method(matrix):

    if len(matrix) != len(matrix[0]):
        print("ERROR: MATRIX IS NOT IN THE FORM nxn")
        exit(1)

    n = len(matrix)

    for i in range(n):
        for k in range(i):
            matrix[i][i] -= pow(matrix[k][i], 2)

        if matrix[i][i] <= 0:
            return False

        matrix[i][i] = np.sqrt(matrix[i][i])

        if i != n:
            for j in range(i+1, n):
                for k in range(i):
                    matrix[i][j] -= matrix[k][i]*matrix[k][j]
                matrix[i][j] /= matrix[i][i]

    return True


# A function that uses gaussian elimination with partial pivoting to solve for x in Ax = b
#   Time complexity = O(n³)
def gaussian_elimination(matrix, b):

    #   Conditional to ensure we have the dimensions of our matrix and vector match
    if len(matrix) != len(matrix[0]) or len(matrix) != len(b):
        exit(-1)

    n = len(matrix)
    x = [0 for _ in range(n)]

    for j in range(n-1):

        largest_row = j

        for i in range(j, n):
            if abs(matrix[i][j]) > abs(matrix[largest_row][j]):
                largest_row = i

        m_row_swap(matrix, j, largest_row)
        v_row_swap(b, j, largest_row)

        for i in range(j+1, n):
            matrix[i][j] /= matrix[j][j]
            for k in range(j+1, n):
                matrix[i][k] -= matrix[i][j] * matrix[j][k]
            b[i] -= matrix[i][j] * b[j]

    for i in range(n):

        x[n-i-1] = b[n-i-1]

        for j in range(i):
            x[n-i-1] -= matrix[n-i-1][n-j-1]*x[n-j-1]

        if(matrix[n-i-1][n-i-1]) == 0:
            exit(5)

        x[n-i-1] /= matrix[n-i-1][n-i-1]

    return x


# Tests the Cholesky method for integers from 1,...,n
#   Saves a graph depicting time vs matrix size to 'cholesky.png'
def test_cholesky(n):
    # We will test this for matrices of size 1,...,50
    for i in range(1, n):

        while True:
            # Generating a Positive Definite Matrix named "matrix"
            #    First we generate a matrix of random numbers, then multiply that matrix by its transpose
            initial_matrix = np.random.rand(i, i)
            matrix = np.dot(initial_matrix, initial_matrix.transpose())

            # This gives us a symmetric matrix which is most likely positive definite
            #   Just to make sure, we verify that the matrix is positive definite, if not we redo this process
            if is_symmetric(matrix) and np.linalg.det(matrix) != 0:
                break

        # Setting our datetime before the cholesky_method is run
        start = time.time()

        # If the process fails to verify that a matrix is pos. def, we print it out
        if not cholesky_method(matrix):
            print("The matrix is not positive definite")

        # Setting our datetime after the cholesky_method has run
        end = time.time()

        # Add this datapoint to the scatter plot to verify that it is n^3
        plot.scatter(i, end-start, marker='.')

    # Labeling axes and saving to 'cholesky.png'
    plot.xlabel('Matrix Size')
    plot.ylabel('Time (s)')
    plot.savefig('cholesky.png')
    plot.clf()


# Tests the Gaussian Elimination Method matrix size 1,...,n
#   Saves graph depicting time vs matrix size to 'gaussian.png'
def test_gaussian(n):
    # We will test this for matrices of size 1,...,n
    for i in range(1, n):

        while True:
            # Generating a Positive Definite Matrix named "matrix"
            #    First we generate a matrix of random numbers
            matrix = np.random.rand(i, i)
            # Checking that the above code produced a symmetric, pos def matrix
            if np.linalg.det(matrix) != 0:
                break

        # Generate a vector, b, of random numbers
        b = [random.random() for _ in range(i)]

        # Store the datetime value before we begin gaussian_elimination in start
        start = time.time()

        # Running gaussian_elimination on our matrix and b vector,
        #   The function returns the solution x, but we don't need it to test efficiency
        gaussian_elimination(matrix, b)

        # Store the datetime value just after gaussian_elimination finishes in end
        end = time.time()

        # Append the datapoint to our scatter plot
        plot.scatter(i, end - start, marker='.')

    # Plot the data and save it to 'gaussian.png'
    plot.xlabel('Matrix Size')
    plot.ylabel('Time (s)')
    plot.savefig('gaussian.png')
    plot.clf()


# Driver code
if __name__ == '__main__':
    test_cholesky(50)
    test_gaussian(50)
