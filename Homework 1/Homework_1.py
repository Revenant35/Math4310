import random
import time
import numpy as np
import matplotlib.pyplot as plot


# Function to print a matrix in the form [a1, a2, a3...an]
def matrix_print(matrix):
    for row in range(len(matrix)):
        print("[", end='')
        for column in range(len(matrix[0])):
            print(matrix[row][column], end='')
            if column != len(matrix[0])-1:
                print(",", end='')
        print("]")


# Function that multiplies two nxn matrices
# Time complexity of O(n^3)
def matrix_multiply(matrix_a, matrix_b):
    # Checking that we have two valid nxn matrices
    if len(matrix_a) != len(matrix_b) or\
            len(matrix_a) != len(matrix_b[0]) or\
            len(matrix_a) != len(matrix_a[0]):
        print("ERROR, INVALID MATRICES RECEIVED")
        exit(1)

    n = len(matrix_a)
    product_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for row in range(n):
        for column in range(n):
            for index in range(n):
                product_matrix[row][column] += matrix_a[row][index] * matrix_b[index][column]
    return product_matrix


def l_tri_matrix_solve(G, b):
    # Conditional to ensure we have an nxn matrix and a n-tall vector
    if len(G) != len(G[0]) or len(G) != len(b):
        print("ERROR, MATRIX/VECTOR MISMATCH")
        exit(1)

    n = len(G)
    for i in range(n):
        if i != 0:
            for j in range(i):
                b[i] -= G[i][j]*b[j]

        if G[i][i] == 0:
            print("ERROR: G[i][i] should never equal zero")
            print("(This is most likely due to the matrix being invalid)")
            exit(2)

        b[i] /= G[i][i]


def matrix_multiply_vector(G, y):
    if len(G[0]) != len(y):
        print("ERROR, MATRIX/VECTOR MISMATCH")
        exit(1)

    n = len(G)
    m = len(G[0])
    b = [0 for _ in range(n)]
    for i in range(n):
        for j in range(m):
            b[i] += G[i][j] * y[j]
        b[i] = round(b[i])
    return b


if __name__ == '__main__':
    n = 100
    for i in range(1, n):
        A = [[random.randint(0, 1000) for _ in range(i)] for _ in range(i)]
        B = [[random.randint(0, 1000) for _ in range(i)] for _ in range(i)]
        start = time.time()
        C = matrix_multiply(A, B)
        end = time.time()
        plot.scatter(i, end - start)
    plot.xlabel('Matrix Size')
    plot.ylabel('Time (s)')
    plot.savefig('output.png')
    plot.show()

    while True:
        G = [[0 for _ in range(n)] for _ in range(n)]
        for row in range(n):
            for column in range(row+1):
                G[row][column] = random.randint(1, 100)

        if np.linalg.det(G) != 0:
            break

    b = [random.randint(1, 100) for _ in range(n)]
    print(b)
    l_tri_matrix_solve(G, b)
    print("y = ", b)
    print(matrix_multiply_vector(G, b))


