# Deep Reinforcement Learning Nanodegree - Project 3

This repository contains my submission for the third project of the DRL Nanodegree

## Assignment
The challenge is the solve Unity's 'Tennis' environemnt. This environment consists of 2 tennis rackets positioned on
either side of a net. The challenge is to succesfully bounce a ball back and forth over the net.
Each time a racket succesfully bounces the ball across the net 0.1 points are rewarded to that racket. 
If the ball touches the ground or goes out of bounds the racket is penalized with -0.01 points.
At the end of a game the maximum of the points awarded to each of the rackets is used as score. 
The challenge is solved if an average score is obtained of 0.5 over 100 episodes.

At each timestep an 8-dimensional observation is available for each racket, including the position and velocity of 
the ball and racket. The ML agent has the observations of three consequetive timesteps at its disposal to learn from. 
Hence a 24 dimensional input for each racket.
The agent can provide to a racket two continuous actions, ranging in value between -1 and 1, corresponding to moving 
back and forth and jumping.

## Solution
The report.ipynb notebook contains an explanation of my solution.


## Replicating the solution
In order to run the files yourself, please follow these instructions:

- Create a python 3.6 environment
- Clone the repo into this environment
- Install the dependencies in the requirements.txt file in this environment
- Download the necessary Unity Environment from one of the following links and install it in the same directory as the code:
  - Linux: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
  - Mac OSX: click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
  - Windows (32-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)  
  - Windows (64-bit): click [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
  

After the installation has been completed, open the Tennis.ipynb notebook and follow the code from there.
