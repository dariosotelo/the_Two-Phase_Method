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
## Test
lista = np.array([-1,2.,-2.,1.,6.,-9.])
b = np.array([1,1.,1.,1.,1.,1.])
lowest_positive_ratio(lista,b)
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
    # Iterate until the objective function coefficients are all non-negative
    while np.any(table[-1, :-1] < 0) and iterations < 1000:
        vec_c = row_to_np_array(table,-1)
        j = first_negative_cost(vec_c[:-1])
        col_j = col_to_np_array(table,j)
        vec_b = col_to_np_array(table,-1)
        i = lowest_positive_ratio(col_j[:-1],vec_b[:-1])
        # Update the simplex table using the pivot row i and pivot column j
        table[i, :] /= table[i, j]
        for k in range(table.shape[0]):
            if k != i:
                table[k, :] -= table[k, j] * table[i, :]
        # Increment the iteration counter
        iterations += 1
    return table
##Tests
A = np.matrix([
    [0.,5.,50.,1.,1.,0.,10.],
    [1.,-15.,2.,0.,0.,0.,2.],
    [0.,1.,1.,0.,1.,1.,6.],
    [0.,-10.,-2.,0.,1.,0.,-6.],
])
simplex(A)
# %%
A = np.matrix([
    [0.,5.,50.,1.,1.,0.,10.],
    [1.,-15.,2.,0.,0.,0.,2.],
    [0.,1.,1.,0.,1.,1.,6.],
    [0.,-10.,-2.,0.,1.,0.,-6.],
])

A[:-1, 2]
print(col_to_np_array(A,-1))




# %%
import numpy as np
#This section of the code is used to ask the user a matrix and it 
#Thi method receives a matrix as a paramether 
#and adds canonic vectors which are missing in order to build an identity matrix inside of our matrix

#this code returns a negative value if it is not a canon vector.
#it returns the position of the canon vector.
def canonVector(vector):
    if len(vector)==0:
        return -99
    i=0
    sum=0
    por=-1
    while(i<len(vector)):
        if (vector[i]==1):
            sum+=vector[i]
            pos=i
        i+=1
    if sum==1:
        return pos
    else:
        return -1


def identityInsideMatrix(table):
    n,m=table.shape
    i = 0
    basis = 0
    while i < m and basis < n - 1:
        if (table[i,i]!=0):
            table[i,:]/=table[i%n,i]
            for k in range(n):
                if k!= i%n:
                    table[k,:]-=table[k,i]*table[i%n,:]
            basis += 1
        i += 1
    return table

A = np.matrix([
    [0.,5.,50.,1.,1.,0.,10.],
    [1.,-15.,2.,0.,0.,0.,2.],
    [0.,1.,1.,0.,1.,1.,6.],
    [0.,-10.,-2.,0.,1.,0.,-6.],
])

num1=canonVector(A[:,0])

print(A[:,0])
print(num1)
# %%
