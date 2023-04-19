import numpy as np
from SimplexAlgorithm import simplex

def generate_simplex_table_first_phase(matrix_A, vector_b):
    raise NotImplementedError
def from_first_phase_generate_simplex_table_second_phase():
    raise NotImplementedError
def get_solutions(final_table_2p):
    raise NotImplementedError

def TwoPhases(matrix_A, vector_b, vector_c):
    n,m = matrix_A.shape()
    #First phase
    table_0_1p = generate_simplex_table_first_phase(matrix_A, vector_b)
    final_table_1p = simplex(table_0_1p)
    # Second phase
    if final_table_1p[-1][-1] != 0:
        raise Exception
    table_0_2p = from_first_phase_generate_simplex_table_second_phase(final_table_1p)
    final_table_2p = simplex(table_0_2p)
    vector_sol, min_func_obj=get_solutions(final_table_2p)
    return vector_sol, min_func_obj