# The Two-Phase Method
This code uses the Two-Phase method to solve linear programming problems using the simplex algorithm.

The two-phase method involves two phases. In the first phase, an auxiliary problem is solved to find a feasible solution.
This problem involves adding artificial variables to the objective function and solving for a minimum value.
The objective is to find a feasible solution for the original problem by minimizing the sum of the artificial variables.

If a feasible solution is found in the first phase, the algorithm proceeds to the second phase.
In this phase, the artificial variables are removed, and the original objective function is used to find the optimal solution.
If the optimal solution of the original problem has a non-zero value for the artificial variables, then the problem is infeasible.
