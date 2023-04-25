#%%
# Import required libraries
import numpy as np

def first_negative_cost(vector_c):
    n = len(vector_c)
    j = 0
    while j < n:
        if vector_c[j] >= 0:
            j += 1
        else:
            return j

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
    while np.any(table[-1, :-1] < 0) and iterations < 100000:
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
            list.append([j,canonVecVar-1])
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
    if abs(final_table_1p[-1][-1]) >= 10**-5:
        raise Exception("The problem has no feasible solution")

    # Second phase: generate simplex table for the second phase and solve it using simplex algorithm
    table_0_2p = from_first_phase_generate_simplex_table_second_phase(final_table_1p, vector_c)
    final_table_2p = simplex(table_0_2p)

    # Return the optimal solution (vector_sol) and the minimum function objective value (min_func_obj)
    return get_solutions_simplex(final_table_2p)

