#%%
import numpy as np
from SimplexAlgorithm import simplex
#%%
def generate_simplex_table_first_phase(matrix_A, vector_b):
    n,m =  np.shape(matrix_A)
    table_0 = np.zeros((n+1,m+n+1))
    # Filling the matrix A's part of the table
    for i in range(n):
        for j in range(m):
           table_0[i,j] =  matrix_A[i,j]
    # Filling the identity part of the table
    for i in range(n):
        table_0[i,m+i] = 1
    # Filling the vector b's part of the table
    for i in range(n):
        table_0[i,-1] =  vector_b[i]
    # Filling the vector cost part of the table
    for i in range(n):
        table_0[-1,m+i] = 1
    return table_0
#%%
## Pruebas
A = np.matrix([[1,2],[1,2]])
b = np.array([7,8])

generate_simplex_table_first_phase(A,b)
#%%
def from_first_phase_generate_simplex_table_second_phase(final_table_1p, vector_c):
    n1P,m1P =  np.shape(final_table_1p)
    n, m = n1P - 1, m1P-n1P
    table_0 = np.zeros((n+1,m+1))
    # Filling the matrix A's part of the table
    for i in range(n):
        for j in range(m):
           table_0[i,j] =  final_table_1p[i,j]
    print(table_0)
    # Filling the vector b's part of the table
    for i in range(n):
        table_0[i,-1] =  final_table_1p[i,-1] 
    print(table_0)
    # Filling the vector cost part of the table
    for i in range(m):
        table_0[-1,i] = vector_c[i]
    print(table_0)
    return table_0
#%% Pruebas
A =np.matrix([[1.,0.,1.,2.5,0.,7.],
               [0.,1.,0.,1.,1.5,8.],
               [0.,0., 0,5., 5. ,0.]])
c = [1,2,3]
from_first_phase_generate_simplex_table_second_phase(A,c)
#%%
def get_solutions_simplex(final_table):
    n2P,m2P =  np.shape(final_table)
    n,m = n2P-1,m2P-1
    sol_n = 0
    vector_sol = np.zeros(m)
    min_func_obj = - final_table[n,m]
    # Filling the matrix A's part of the table
    for i in range(m):
        # only find the solution with the first n zeros of the basis
        if final_table[-1,i] == 0 and sol_n != n:
            j = 0
            ## search for the 1 of the basis
            while j < n:
                if final_table[j,i] == 1:
                    break
                j+=1
            ## save the value of b of the 1 finded
            vector_sol[i] = final_table[j,-1]
            sol_n +=1
        else:
            vector_sol[i] = 0
    return vector_sol, min_func_obj
A =np.matrix([[1.,0.,1.,2.5,0.,7.],
               [0.,1.,0.,1.,1.5,8.],
               [0.,0., 1,5., 5. ,5.]])
get_solutions_simplex(A)
#%%
def TwoPhases(matrix_A, vector_b, vector_c):
    #First phase
    table_0_1p = generate_simplex_table_first_phase(matrix_A, vector_b)
    final_table_1p = simplex(table_0_1p)
    # Second phase
    if final_table_1p[-1][-1] != 0:
        raise Exception
    table_0_2p = from_first_phase_generate_simplex_table_second_phase(final_table_1p,vector_c)
    final_table_2p = simplex(table_0_2p)
    ## return of a tuple with vector_sol, min_func_obj
    return get_solutions_simplex(final_table_2p)
    
# %%
