#%%
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

def make_canonical_basis(table):
    n,m=table.shape
    basis = {i  for i in range(n-1)}
    while basis:
        for e_i in basis :
            if (table[e_i,e_i] != 0):
                table[e_i,:]/=table[e_i,e_i]
                for k in range(n):
                    if k!= e_i:
                        table[k,:]-=table[k,e_i]*table[e_i,:]
                basis.remove(e_i)
                break
    return table

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

#This section of the code is used to ask the user a matrix and it 
#Thi method receives a matrix as a paramether 
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
            print(table[-1,:])
    return table


# %%
