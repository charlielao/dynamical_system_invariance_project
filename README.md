# dynamical_system_invariance_project
An implementation of my master thesis "Learning Invariances in Dynamical Systems" supervised by Dr Mark van der Wilk and Dr Andrew Duncan. 
The codes are built on GPflow and provide several kernels and mean functions that incoporates the inductive bias of invariances and symmetries underlying a dynamical system. 
Specifically the conservation of energy in this case. The model is also able to enforce soft invariance to account for dissipative systems. 
More importantly, the model is able to recover the form of invariance or physics laws automatically by maximising the marginal likelihood of the data.

The methods are then experimentally tested on several 1D and 2D, conservative and dissipative, linear and nonlinear systems; 
namely simple harmonic motion, pendulum, damped version of these as well as 2D simple harmonic motion and double pendulum.
The results shown that our methods beat the baseline bench mark and is able to recover the physics of dynamical system successfully.
