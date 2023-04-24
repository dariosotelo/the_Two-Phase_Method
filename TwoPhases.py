#%%
# Import required libraries
import numpy as np
from SimplexAlgorithm import simplex
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
def get_solutions_simplex(final_table):
    # Get the dimensions of the final simplex table
    n2P, m2P = np.shape(final_table)
    # Get the number of variables and constraints in the problem
    n, m = n2P - 1, m2P - 1
    # Initialize the number of non-zero solutions found and the solution vector
    sol_n = 0
    vector_sol = np.zeros(m)
    # Compute the minimum value of the objective function (the negative of the bottom-right entry)
    min_func_obj = -final_table[n, m]
    
    # Iterate over the columns of the final table to extract the solution vector
    for i in range(m):
        # If the variable is basic and the maximum number of non-zero solutions hasn't been found yet
        if final_table[-1, i] == 0 and sol_n != n:
            j = 0
            # Search for the 1 in the column corresponding to the basic variable
            while j < n:
                if final_table[j, i] == 1:
                    break
                j += 1
            # Save the value of the corresponding entry in the b vector
            vector_sol[i] = final_table[j, -1]
            # Increment the number of non-zero solutions found
            sol_n += 1
        else:
            vector_sol[i] = 0
    
    # Return the solution vector and the minimum value of the objective function
    return vector_sol, min_func_obj
#%% tests
A =np.matrix([[1.,0.,1.,2.5,0.,7.],
               [0.,1.,0.,1.,1.5,8.],
               [0.,0., 1,5., 5. ,5.]])
get_solutions_simplex(A)
#%%
def TwoPhases(matrix_A, vector_b, vector_c):
    # First phase: generate simplex table for the first phase and solve it using simplex algorithm
    table_0_1p = generate_simplex_table_first_phase(matrix_A, vector_b)
    final_table_1p = simplex(table_0_1p)

    # Check if the problem has a feasible solution; if not, raise an exception
    if abs(final_table_1p[-1][-1]) >= 10**-5:
        raise Exception("The problem has no feasible solution")

    # Second phase: generate simplex table for the second phase and solve it using simplex algorithm
    table_0_2p = from_first_phase_generate_simplex_table_second_phase(final_table_1p, vector_c)
    final_table_2p = simplex(table_0_2p)

    # Return the optimal solution (vector_sol) and the minimum function objective value (min_func_obj)
    return get_solutions_simplex(final_table_2p)

    
# %%
