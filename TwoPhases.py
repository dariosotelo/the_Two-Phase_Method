import numpy as np
from SimplexAlgorithm import simplex

def generate_simplex_table_first_phase(matrix_A, vector_b):
    raise NotImplementedError

def TwoPhases(matrix_A, vector_b, vector_c):
    n,m = matrix_A.shape()
    #First phase
    table_0_1p = generate_simplex_table_first_phase(matrix_A, vector_b)
    final_table = simplex(table_0_1p)