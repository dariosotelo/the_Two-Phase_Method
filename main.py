#%%
from SimplexAlgorithm import *
from TwoPhases import *

if __name__=='__main__':
    #These are three linear programming problems which are solved by the code in the SimplexAlgorithm.py and TwoPhases.py files:
        
    # A trial
    matrix_a = np.matrix([
    #   [1., 2., 3., 4., 5., 6., 7., 8., 9., 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  20,  -20,  a.,  e1, h1,  e2, h2, h3]
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,  0.,   0.,  0.,  0., 0.,  0., 0., 0.],
        [4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 0.,  0.,   0.,  0., -1., 0.,  0., 0., 0.],
        [2., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,  1.,  -1.,  0.,  0., 1.,  0., 0., 0.],
        [0., 0., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 4., 0., 0.,  1.,  -1.,  0.,  0., 0., -1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,  1.,   0., -1.,  0., 0.,  0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,  0.,  -1., -1.,  0., 0.,  0., 0., 1.],
    ])
    #               [1., 2., 3., 4., 5., 6., 7., 8., 9., 10, 11, 12, 13, 14, 15, 16, 17, 18,  19, 20,-20,  a, e1, h1, e2, h2, h3]
    vector_cost_a = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., -1., 0., 0., -1, 0., 0., 0., 0., 0.]
    vector_obj_a = np.array([2.,2., 2., 0., 0., 0.])
    print("Sol. PPL A Modelado: ")
    TwoPhases(matrix_a,vector_obj_a,vector_cost_a)

    # #B trial:
    matrix_b = np.matrix([
        [1., 1., -1., 0., -1., 0., 0., 0.],
        [1., 1., 2., 3., 0., 1., 0., 0.],
        [1., 2., -1., 2., 0., 0., 1., 0.],
        [0., 1., 0., 2., 0., 0., 0., -1.]
    ])
    vector_obj_b = np.array([2.,10., 6., 5.])
    vector_cost_b = np.array([3., 6., -1., 2., 0., 0., 0., 0.,])

    print("Sol. PPL B Modelado: ")
    TwoPhases(matrix_b,vector_obj_b,vector_cost_b)
    #C trial:
    matrix_c = np.matrix([
        [1., 1., -1., 0., 0., -1., 0., 0.],
        [1., 1., 2., 3., 0., 0., 1., 0.],
        [3., 0., 0., 1., -1., 0., 0., 0.],
        [0., 1., 2., 0., 0., 0., 0., 1.],
    ])
    vector_obj_c = np.array([2.,10.,5., 2. ])
    vector_cost_c = np.array([3., 6., -1., 2., 7., 0., 0., 0.])
    print("Sol. Ejercicio C Modelado: ")
    TwoPhases(matrix_c,vector_obj_c,vector_cost_c)

    A=np.array([
            [1., 0., 0., 0., 1., 0., 0., 0.],
            [20., 1., 0., 0., 0., 1., 0., 0.],
            [200., 20., 1., 0., 0., 0., 1., 0.],
            [2000., 200., 20., 1., 0., 0., 0., 1.]
            ])
        
    b_vector=np.array([1., 100., 10000., 1000000.])
    c_vector=np.array([-1000., -100., -10., -1., 0., 0., 0., 0.])
    print("Sol. Ejercicio A clase Modelado: ")
    TwoPhases(A,b_vector,c_vector)
    B=np.array([
        [1., 1., 0.],
        [2., 0., -1.]    
    ])
    b_vector=np.array([4., 2.])
    c_vector=np.array([1., -1., 0.])
    print("Sol. Ejercicio B clase Modelado: ")
    TwoPhases(B,b_vector,c_vector)

    C=np.array([
        [1., 1., 1., 0., 0., -1., 0., 0., 0., 0.],
        [-2., -1., 0., -1., 1., 0., -1., 0., 0., 0.],
        [1., 1., 2., 3., 0., 0., 0., -1., 0., 0.],
        [0., 1., 0., 2., 0., 0., 0., 0., 0., 1.]
    ])
    b_vector=np.array([2., 1., 10., 6., 5.])
    c_vector=np.array([8., -2., 1., 2., 5., 0., 0., 0., 0., 0.])
    print("Sol. Ejercicio C clase Modelado: ")
    TwoPhases(C,b_vector,c_vector)    
# %%
