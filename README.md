# The Two-Phase Method
This code uses the Two-Phase method to solve linear programming problems using the simplex algorithm.

The two-phase method involves two phases. In the first phase, an auxiliary problem is solved to find a feasible solution.
This problem involves adding artificial variables to the objective function and solving for a minimum value.
The objective is to find a feasible solution for the original problem by minimizing the sum of the artificial variables.

If a feasible solution is found in the first phase, the algorithm proceeds to the second phase.
In this phase, the artificial variables are removed, and the original objective function is used to find the optimal solution.
If the optimal solution of the original problem has a non-zero value for the artificial variables, then the problem is infeasible.

The methods that solve the problems are in the files "SimplexAlgorithm.py" and "TwoPhases.py". 
The "main.py" file contains tests for three linear programming problems, and there is a Rich Text Format file that has the solutions to the three problems. 
All methods and tests are combined in the same file called "Test.py". There is a folder "LingoSolutions" which contains LINGO code programmed to check on the solutions. The objective of these files is to ensure the real objective function value is calculated.
