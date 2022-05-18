# iLQR (iterative Linear Quadratic Control

Implemented an iLQR algorithm for a particular trajectory to find an optimal version of the trajectory. 

This method employed a runge-kutta 4th order integral method to estimate the next steps of the trajectory based on the control signal and input. It also employed gradient descent to find the most optimal trajectory, with the armijo line search to determine how far along the line to move during descent. 
