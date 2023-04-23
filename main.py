from SimplexAlgorithm import *
from TwoPhases import *

if __name__=='__main__':
    A = np.matrix([
        [0.,5.,50.,1.,1.,0.,10.],
        [1.,-15.,2.,0.,0.,7.,2.],
        [0.,1.,1.,0.,1.,1.,6.],
        [0.,-10.,-2.,0.,1.,0.,-6.],
    ])
    print(A)
    A=identityInsideMatrix(A)
    print(A)