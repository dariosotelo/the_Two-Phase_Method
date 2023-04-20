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
def to_numpy_array(vector):
    raise NotImplementedError
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
    print(table[-1, :-1].reshape(-1,1))
    # Iterate until the objective function coefficients are all non-negative
    while np.any(table[-1, :-1] < 0) and iterations < 1000:
        
        j = first_negative_cost(table[-1, :-1])
        i = lowest_positive_ratio(table[:-1, j],table[:-1, -1])
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
vec = A[-1, :-1]
arr = np.shape(vec)[1]
vec1 = vec.flatten(order='F')
print(vec1)
# %%
