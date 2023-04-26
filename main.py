#%%
from SimplexAlgorithm import *
from TwoPhases import *

if __name__=='__main__':
    #These are three linear programming problems which are solved by the code in the SimplexAlgorithm.py and TwoPhases.py files:
        
    #A trial
    print("")
    print("First trial: ")
    matrix_a = np.matrix([
    #   [X1., X2., X3., X4., X5., X6., X7., X8., X9., X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, X20,  -20,  a.,  e1, h1,  e2, h2, h3]
        [1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,   0.,  0.,  0., 0.,  0., 0., 0.],
        [4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  0.,  0.,   0.,  0., -1., 0.,  0., 0., 0.],
        [2.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  -1.,  0.,  0., 1.,  0., 0., 0.],
        [0.,  0.,  4.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  4.,  0.,  0.,  1.,  -1.,  0.,  0., 1., -1., 0., 0.],
        [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,   0., -1.,  0., 0.,  0., 1., 0.],
        [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  -1., -1.,  0., 0.,  0., 0., 1.],
    ])
    #               [X1., X2., X3., X4., X5., X6., X7., X8., X9., X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, X20, -20,  a, e1, h1, e2, h2, h3]
    vector_cost_a = [1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., -1.,  0.,   0., -1, 0., 0., 0., 0., 0.]
    vector_obj_a = np.array([2.,2., 2., 0., 0., 0.])
    try:
        auxprint=TwoPhases(matrix_a, vector_obj_a, vector_cost_a)
        print("The solution of the first LPP is: ")
        for i in range(len(auxprint[0])):
            print("Variable",i+1, "is:",auxprint[0][i])
        print("With the value of the objective function:", auxprint[1])
    except:
        print("The LPP has no feasible solution")
    
    #B trial:
    print("")
    print("Second trial: ")
    matrix_b = np.matrix([
        [1., 1., -1., 0., -1., 0., 0., 0.],
        [1., 1., 2., 3., 0., 1., 0., 0.],
        [1., 2., -1., 2., 0., 0., 1., 0.],
        [0., 1., 0., 2., 0., 0., 0., -1.]
    ])
    vector_obj_b = np.array([2.,10., 6., 5.])
    vector_cost_b = np.array([3., 6., -1., 2., 0., 0., 0., 0.,])
    try:
        auxprint=TwoPhases(matrix_b, vector_obj_b, vector_cost_b)
        print("The solution of the second LPP is: ")
        for i in range(len(auxprint[0])):
            print("Variable",i+1,"is:",auxprint[0][i])
        print("With the value of the objective function:", auxprint[1])
    except:
        print("The LPP has no feasible solution")
        
    #C trial:
    print("")
    print("Third trial: ")
    matrix_c = np.matrix([
        [1., 1., -1., 0., 0., -1., 0., 0.],
        [1., 1., 2., 3., 0., 0., 1., 0.],
        [3., 0., 0., 1., -1., 0., 0., 0.],
        [0., 1., 2., 0., 0., 0., 0., 1.],
    ])
    vector_obj_c = np.array([2.,10.,5., 2. ])
    vector_cost_c = np.array([3., 6., -1., 2., 7., 0., 0., 0.])
    try:
        auxprint=TwoPhases(matrix_c, vector_obj_c, vector_cost_c)
        print("The solution of the third LPP is: ")
        for i in range(len(auxprint[0])):
            print("Variable", i+1, "is:", auxprint[0][i])
        print("With the value of the objective function:", auxprint[1])
    except:
        print("The LPP has no feasible solution")

    #--------------------------------------------------------------
    A=np.array([
        [1., 0., 0., 0., 1., 0., 0., 0.],
        [20., 1., 0., 0., 0., 1., 0., 0.],
        [200., 20., 1., 0., 0., 0., 1., 0.],
        [2000., 200., 20., 1., 0., 0., 0., 1.]
        ])
    
    b_vector=np.array([1., 100., 10000., 1000000.])
    c_vector=np.array([-1000., -100., -10., -1., 0., 0., 0., 0.])

    print("")
    print("----------------------------------------------")
    try:
        auxprint=TwoPhases(A, b_vector, c_vector)
        print("The solution of the fourth LPP is: ")
        for i in range(len(auxprint[0])):
            print("Variable", i+1, "is:", auxprint[0][i])
        print("With the value of the objective function:", auxprint[1])
    except:
        print("The LPP has no feasible solution")
    
    print("")
    B=np.array([
        [1., 1., 0.],
        [2., 0., -1.]    
    ])
    c_vector=np.array([1., -1., 0.])
    b_vector=np.array([4., 2.])
    '''''
    B=np.array([
        [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0],
        [2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 0., -1]
    ])

    b_vector=np.array([4., 2.])
    c_vector=np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., -1., 0.])
    '''
    try:
        auxprint=TwoPhases(B, b_vector, c_vector)
        print("The solution of the fifth LPP is: ")
        for i in range(len(auxprint[0])):
            print("Variable", i+1, "is:", auxprint[0][i])
        print("With the value of the objective function:", auxprint[1])
    except:
        print("The fifth LPP has no feasible solution")
    

    print("")
    C=np.array([
        [1., 1., 1., 0., 0., -1., 0., 0., 0., 0.],
        [-2., -1., 0., -1., 1., 0., -1., 0., 0., 0.],
        [1., 1., 2., 3., 0., 0., 0., -1., 0., 0.],
        [0., 1., 0., 2., 0., 0., 0., 0., 0., 1.]
    ])

    b_vector=np.array([2., 1., 10., 6., 5.])
    c_vector=np.array([8., -2., 1., 2., 5., 0., 0., 0., 0., 0.])

    try:
        auxprint=TwoPhases(C, b_vector, c_vector)
        print("The solution of the sixth LPP is: ")
        for i in range(len(auxprint[0])):
            print("Variable", i+1, "is:", auxprint[0][i])
        print("With the value of the objective function:", auxprint[1])
    except:
        print("The sixth LPP has no feasible solution")

    
# %%
