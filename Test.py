#%%
# Import required libraries
import numpy as np
#%%
def first_negative_cost(vector_c):
    n = len(vector_c)
    j = 0
    while j < n:
        if vector_c[j] >= 0:
            j += 1
        else:
            return j
#%%
def find_the_first_positive(vector):
    n = len(vector)
    j = 0
    while j < n:
        if vector[j] <= 0:
            j += 1
        else:
            return j
def lowest_positive_ratio(vector_xj,vector_b):
    ratios = vector_b/vector_xj
    ratios_sort = np.sort(ratios)
    result_sort = np.where(ratios_sort >= 0)
    result = np.where(ratios == ratios_sort[result_sort[0][0]])
    # Return lowest_positive_ratio
    return result[0][0]
#%%
def row_to_np_array(matrix_A,row):
    n,m = np.shape(matrix_A)
    vect = np.zeros(m)
    for i in range(m):
        vect[i] = matrix_A[row,i]
    return vect
def col_to_np_array(matrix_A,col):
    n,m = np.shape(matrix_A)
    vect = np.zeros(n)
    for i in range(n):
        vect[i] = matrix_A[i,col]
    return vect

#%% 
def simplex(table):
    """
    Solves a linear programming problem in standard form using the simplex algorithm.

    Args:
    - table: A numpy array representing the simplex table of the problem in standard form.
    
    Returns:
    - A numpy array representing the final simplex table after the algorithm has converged.
    """
    # Initialize the iteration counter
    iterations = 0
    table = identityInsideMatrix(table)
    # Iterate until the objective function coefficients are all non-negative
    while np.any(table[-1, :-1] < 0) and iterations < 1000:
        vec_c = row_to_np_array(table,-1)
        vec_b = col_to_np_array(table,-1)
        
        j = first_negative_cost(vec_c[:-1])
        
        col_j = col_to_np_array(table,j)
        
        i = lowest_positive_ratio(col_j[:-1],vec_b[:-1])
        # Update the simplex table using the pivot row i and pivot column j
        table[i, :] /= table[i, j]
        for k in range(table.shape[0]):
            if k != i:
                table[k, :] -= table[k, j] * table[i, :]
        # Increment the iteration counter
        iterations += 1
    return table

# %%
#This section of the code is used to ask the user a matrix and it 
#This method receives a matrix as a paramether 
#and adds canonic vectors which are missing in order to build an identity matrix inside of our matrix

#this code returns a negative value if it is not a canon vector.
#it returns the position of the canon vector.
def canonVector(vector, n):
    if len(vector)==0:
        return -99
    i=0
    sum = 0
    pos=-1
    while(i<n):
        sum+=abs(vector[i])
        if (vector[i]==1):
            pos=i
        i+=1
    if sum==1:
        return pos
    else:
        return -1

#This function turns the whole column into 0s except the value given in row and column, it turns that one into 1
def turnToCanonColumn(matrix, row, column):
    matrix[row,:] /= matrix[row, column]
    for i in range(matrix.shape[0]):
        if i!=row:
            matrix[i,:]-=matrix[row,:]*matrix[i, column]
    return matrix

def identityInsideMatrix(table):
    n,m=table.shape
    i=0
    j=0
    canonColumns=[]
    canonList=[]
    missingCanonVectors=[]
    #We build a list which contains the canon vectors inside the matrix
    #this is used to increase efficiency
    while j<m:
        pos = canonVector(table[:, j],n-1)
        if (pos>=0):
            canonList.append(pos)
            canonColumns.append(j)
        j+=1
    #This list is used to check which canon vectors are needed
    #it is filled with the canon vectors that are not in the matrix
    aux=list(range(0,n-1))
    missingCanonVectors = [x for x in aux if x not in canonList]
    #Si algo quiebra, es esta variable i
    i=0
    while i<m-1 and len(missingCanonVectors)!=0:
        if i not in canonColumns:
            newCanonPosition=missingCanonVectors.pop(0)
            turnToCanonColumn(table, newCanonPosition, i)
        i+=1
    for canonVectorVar in canonColumns:
        if table[-1,canonVectorVar]!=0:
            pos = canonVector(table[:, canonVectorVar],n-1)
            table[-1,:]-=table[pos,:]*table[-1, canonVectorVar]
    return table

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

A = np.matrix([
    [3.,4.,1.,0.],
    [2.,-1.,0.,-1.]
])
b = np.array([20.,2.])
c = np.array([1.,2.,0.,0.])
print("Sol. Prueba Libro: ",TwoPhases(A,b,c))


#A trial
# 1-19 x+20 x-20 X20a e1
matrix_a = np.matrix([
#   [1., 2., 3., 4., 5., 6., 7., 8., 9., 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  -20,  a.,  e1, h1,  e2, h2, h3]
    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.,   0.,  0.,  0., 0.,  0., 0., 0.],
    [4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 0., 0.,   0.,  0., -1., 0.,  0., 0., 0.],
    [2., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,  -1.,  0.,  0., 1.,  0., 0., 0.],
    [0., 0., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 4., 0., 0., 1.,  -1.,  0.,  0., 1., -1., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,   0., -1.,  0., 0.,  0., 1., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,  -1., -1.,  0., 0.,  0., 0., 1.],
])
#               [1., 2., 3., 4., 5., 6., 7., 8., 9., 10, 11, 12, 13, 14, 15, 16, 17, 18,  19, 20,-20,  a, e1, h1, e2, h2, h3]
vector_cost_a = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., -1., 0., 0., -1, 0., 0., 0., 0., 0.]
vector_obj_a = np.array([2.,2., 2., 0., 0., 0.])
try:
    print("Sol. PPL A Modelado: ",TwoPhases(matrix_a,vector_obj_a,vector_cost_a))
except:
    print("Sol. PPL A Modelado: No tiene solucion factible ")
#B trial:
matrix_b = np.matrix([
    [1., 1., -1., 0., -1., 0., 0., 0.],
    [1., 1., 2., 3., 0., 1., 0., 0.],
    [1., 2., -1., 2., 0., 0., 1., 0.],
    [0., 1., 0., 2., 0., 0., 0., -1.]
])
vector_obj_b = np.array([2.,10., 6., 5.])
vector_cost_b = np.array([3., 6., -1., 2., 0., 0., 0., 0.,])
try:
    print("Sol. PPL B Modelado: ",TwoPhases(matrix_b,vector_obj_b,vector_cost_b))
except:
    print("Sol. PPL B Modelado: No tiene solucion factible ")
#C trial:
matrix_c = np.matrix([
    [1., 1., -1., 0., 0., -1., 0., 0.],
    [1., 1., 2., 3., 0., 0., 1., 0.],
    [3., 0., 0., 1., -1., 0., 0., 0.],
    [0., 1., 2., 0., 0., 0., 0., 1.],
])
vector_obj_c = np.array([2.,10.,5., 2. ])
vector_cost_c = np.array([3., 6., -1., 2., 7., 0., 0., 0.])
try:
    print("Sol. Ejercicio C Modelado: ",TwoPhases(matrix_c,vector_obj_c,vector_cost_c))
except:
    print("Sol. PPL C Modelado: No tiene solucion factible ")

# %%
