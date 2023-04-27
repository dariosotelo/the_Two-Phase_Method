#%%
# Import required libraries
import numpy as np
from SimplexAlgorithm import *
np.seterr(divide='ignore', invalid='ignore')
#%%
# Define a function to generate the simplex table for the first phase of the two-phase simplex algorithm
def generate_simplex_table_first_phase(matrix_A, vector_b):
    # Get the dimensions of matrix_A
    n,m = np.shape(matrix_A)
    # Create a table of zeros with the appropriate dimensions
    table_0 = np.zeros((n+1,m+n+1))
    # Fill in the matrix A's part of the table
    for i in range(n):
        for j in range(m):
            table_0[i,j] = matrix_A[i,j]
    # Fill in the identity part of the table
    for i in range(n):
        table_0[i,m+i] = 1
    # Fill in the vector b's part of the table
    for i in range(n):
        table_0[i,-1] = vector_b[i]
    # Fill in the vector cost part of the table
    for i in range(n):
        table_0[-1,m+i] = 1
    # Return the completed table
    return table_0
#%% Tests
A = np.matrix([[1,2],[1,2]])
b = np.array([7,8])

generate_simplex_table_first_phase(A,b)
#%%
def from_first_phase_generate_simplex_table_second_phase(final_table_1p, vector_c):
    # Get the dimensions of the first-phase final table
    n1P, m1P = np.shape(final_table_1p)
    # Calculate the dimensions of the second-phase table
    n, m = n1P - 1, m1P - n1P
    # Create a table of zeros with the appropriate dimensions
    table_0 = np.zeros((n+1, m+1))
    # Fill in the matrix A's part of the table
    for i in range(n):
        for j in range(m):
            table_0[i,j] = final_table_1p[i,j]
    # Fill in the vector b's part of the table
    for i in range(n):
        table_0[i,-1] = final_table_1p[i,-1] 
    # Fill in the vector cost part of the table
    for i in range(m):
        table_0[-1,i] = vector_c[i]
    # Return the completed table
    return table_0
#%% Tests
A =np.matrix([[1.,0.,1.,2.5,0.,7.],
               [0.,1.,0.,1.,1.5,8.],
               [0.,0., 0,5., 5. ,0.]])
c = [1,2,3]
from_first_phase_generate_simplex_table_second_phase(A,c)
#%%
#This method returns the canon vector that was found and it's corresponding position
#The format of return is a list of lists where the first element of the lists is the
#corresponding canon vector, i. e., it is the value of n where En is the nth canon vector
#The second value is the position inside of the matrix
def canonVectorAndPosition(matrix):
    n,m=matrix.shape
    i=0
    j=0
    list=[]
    for j in range(m):
        canonVecVar=canonVector(matrix[:,j], n-1)
        if canonVecVar>=0:
            list.append([j,canonVecVar])
    return list

def get_solutions_simplex(final_table):
    # Get the dimensions of the final simplex table
    n2P, m2P = np.shape(final_table)
    # Get the number of variables and constraints in the problem
    n, m = n2P - 1, m2P - 1
    # Initialize the solution vector
    vector_sol = np.zeros(m)
    pos_solutions = canonVectorAndPosition(final_table)
    # Compute the minimum value of the objective function (the negative of the bottom-right entry)
    min_func_obj = -final_table[n, m]
    # Iterate over the columns of the final table to extract the solution vector
    for pos in pos_solutions:
        vector_sol[pos[0]] = final_table[pos[1], -1]
    # Return the solution vector and the minimum value of the objective function
    return vector_sol, min_func_obj

def TwoPhases(matrix_A, vector_b, vector_c):
    # First phase: generate simplex table for the first phase and solve it using simplex algorithm
    table_0_1p = generate_simplex_table_first_phase(matrix_A, vector_b)
    final_table_1p = simplex(table_0_1p)

    # Check if the problem has a feasible solution; if not, raise an exception
    if is_unbounded(final_table_1p):
        print("The problem is unbounded")
        print("The problem has no feasible solution")
        return 
    if abs(final_table_1p[-1][-1]) >= 10**-5:
        print("The problem has no feasible solution")
        return 

    # Second phase: generate simplex table for the second phase and solve it using simplex algorithm
    table_0_2p = from_first_phase_generate_simplex_table_second_phase(final_table_1p, vector_c)
    final_table_2p = simplex(table_0_2p)
    if is_unbounded(final_table_2p):
        print("the problem is unbounded")
        print("The problem has no feasible solution")
        return 
    # Return the optimal solution (vector_sol) and the minimum function objective value (min_func_obj)
    sol, z_op = get_solutions_simplex(final_table_2p)
    print("The solution of the first LPP is: ")
    for i in range(len(sol)):
        print("Variable",i+1, "is:",sol[i])
    print("With the value of the objective function:", z_op)
    return 

    
# %%
