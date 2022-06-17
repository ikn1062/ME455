# Active Learning and Control Applications in Robotics

This repository contains some of the control and active learning assignments as part of the Northwestern Active Learning in Robotics course. 

A description of each of the assignments is shown below:

Optimal Trajectories:
* Generates optimal trajectories by trying to minimize error through an objective function
* Solves a two point boundary problem for a similar system, plotting the generated trajectories
* Computes directional derivatives for a given system 

iLQR:
* Implementing an iterative linear quadratic regulator control algorithm with gradient descent and armijo line search

Particle and Kalman Filters
* Created state estimations for observations along a given trajectory through a particle filter
* Implemented an augmented version of a kalman filter for a particular dynamical system

Ergodicity and infotaxis
* Created an algorithm to calculate how ergodic a given trajectory was
* Applied infotaxis (search method for a source) to allow an object find a door in an enclosed environment

Ergodic exploration
* Generated ergodic trajectories surrounding the origin point
