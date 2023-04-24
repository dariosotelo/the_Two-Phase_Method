from SimplexAlgorithm import *
from TwoPhases import *
# %%
if __name__=='__main__':
    #First test
    A = np.matrix([
        [0.,5.,50.,1.,1.,0.,10.],
        [1.,-15.,2.,0.,0.,7.,2.],
        [0.,1.,1.,0.,1.,1.,6.],
        [0.,-10.,-2.,0.,1.,0.,-6.],
    ])

    #B trial:
    matrix_b = np.matrix([
        [1., 1., -1., 0., -1., 0., 0., 0.],
        [1., 1., 2., 3., 0., 1., 0., 0.],
        [1., 2., -1., 2., 0., 0., 1., 0.],
        [0., 1., 0., 2., 0., 0., 0., -1.]
    ])
    vector_obj_b = np.array([2.,10., 6., 5.])
    vector_cost_b = np.array([3., 6., -1., 2., 0., 0., 0., 0.,])
    print(TwoPhases(matrix_b,vector_obj_b,vector_cost_b))
    #C trial:
    matrix_c = np.matrix([
        [1., 1., -1., 0., 0., -1., 0., 0.],
        [1., 1., 2., 3., 0., 0., 1., 0.],
        [3., 0., 0., 1., -1., 0., 0., 0.],
        [0., 1., 2., 0., 0., 0., 0., 1.],
    ])
    vector_obj_c = np.array([2.,10.,5., 2. ])
    vector_cost_c = np.array([3., 6., -1., 2., 7., 0., 0., 0.])

    print(TwoPhases(matrix_c,vector_obj_c,vector_cost_c))


# %%
