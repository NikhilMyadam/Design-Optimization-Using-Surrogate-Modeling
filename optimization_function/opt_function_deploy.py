#######################################################################################################################################################################
# Created by Nikhil Myadam
#######################################################################################################################################################################

# Multi-objective optimization function - Pymoo

# Tips to keep in mind - 

# Pymoo considers pure minimization problems for optimization
# All constraint functions need to be formulated as a less-than-equal-to constraint
# Normalization of constraints recommended to give equal importance to each one of them

#######################################################################################################################################################################
# Import necessary modules and libraries
#######################################################################################################################################################################

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib as mpl
import matplotlib.pyplot as plt

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

font = {'family': 'Arial', 'size': 11}  # text in graphic, font and size
mpl.rc('font', **font)

#######################################################################################################################################################################
# Load trained XGB models
#######################################################################################################################################################################

# specify filepaths for trained XGB models
# Note: Update <PATH_TO_MODEL> to the 'Models' directory where the optimal XGB estimators for IPCF, Energy and Total deformation are saved 

f_filepath = r'<PATH_TO_MODEL>\f_xgb_regressor'
e_filepath = r'<PATH_TO_MODEL>\e_xgb_regressor'
td_filepath = r'<PATH_TO_MODEL>\td_xgb_regressor'

# load the trained XGB models
model_f = xgb.XGBRegressor()
model_f.load_model(os.path.join(f_filepath, 'f_xgb_regressor_model.json'))

model_e = xgb.XGBRegressor()
model_e.load_model(os.path.join(e_filepath, 'e_xgb_regressor_model.json'))

model_td = xgb.XGBRegressor()
model_td.load_model(os.path.join(td_filepath, 'td_xgb_regressor_model.json'))

#######################################################################################################################################################################
# Problem definition
#######################################################################################################################################################################

class biobjoptimization_std(Problem):
    # defining the problem and objective functions using respective pretrained surrogate models for the optimization process
    # optimize both the objectives seperately using the total deformation as a constraint

    def __init__(self, model_f, model_e, model_td, epsilon1):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_ieq_constr=1,
                         xl=np.array([95, 0.0008]), # lower bounds of the 2 input variables
                         xu=np.array([100, 0.0024]), # upper bounds of the 2 input variables
                         vtype=np.array([int, float])) # specifying vtype to indicate that 'x1' is 'int' type and 'x2' is 'float' type
        self.model_f = model_f
        self.model_e = model_e
        self.model_td = model_td
        self.epsilon1 = epsilon1 # to constrain the total deformation values

    def _evaluate(self, x, out, *args, **kwargs):
        # Predict the IPCF and Energy values using the XGB models
        # calculating -IPCF and -Energy values as Pymoo uses 'minimize' method for optimization
        f1 = -(self.model_f.predict(x))
        f2 = -(self.model_e.predict(x))

        # giving the Total deformation value limit as constraint
        g1 = self.model_td.predict(x) - self.epsilon1
        
        # Writing objective function(s) and constraint values to dictionaries
        out["F"] = [f1, f2]
        out["G"] = [g1]

#######################################################################################################################################################################
# Algorithm initialization
#######################################################################################################################################################################
# defining the optimization algorithm to solve the problem

algorithm = NSGA2(
    pop_size=100,
    n_offsprings=25,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True)

#######################################################################################################################################################################
# Algorithm termination
#######################################################################################################################################################################
# define the termination criteria to specify when the algorithm stops function evaluations

termination = get_termination("n_gen", 100)

#######################################################################################################################################################################
# Problem initialization
#######################################################################################################################################################################

problem = biobjoptimization_std(model_f, model_e, model_td, epsilon1=0.170)

#######################################################################################################################################################################
# Optimization problem
#######################################################################################################################################################################
# perform the optimization using minimize method

res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               save_history=True,
               verbose=True)

# access design solutions, optimal objective function values, and corresponding total deformation values
X = res.X
F = res.F
G = res.G

# initialize numpy arrays to write data i.e. input features and output features
angle_sols = []
thickness_sols = []
ipcf_values = []
energy_values = []
tdef_values = []

print('The dtype of angle_sols is: ', type(angle_sols))
print('The dtype of thickness_sols is: ', type(thickness_sols))
print('The dtype of ipcf_values is: ', type(ipcf_values))
print('The dtype of energy_values is: ', type(energy_values))
print('The dtype of tdef_values is: ', type(tdef_values))

# access epsilon value
epsilon_value = problem.epsilon1

# retrieve angle solutions data
angle_sols = X[:, 0]
angle_sols = angle_sols.reshape(-1, 1)
print('The dtype of angle_sols is: ', type(angle_sols))
print('The shape of angle_sols is: ', angle_sols.shape)

# retrieve thickness solutions data
thickness_sols = X[:, 1]
thickness_sols = thickness_sols.reshape(-1, 1)
print('The dtype of thickness_sols is: ', type(thickness_sols))
print('The shape of thickness_sols is: ', thickness_sols.shape)

# retrieve optimal peak crushing forces data
ipcf_values = np.negative(F[:, 0])
ipcf_values = ipcf_values.reshape(-1, 1)
print('The dtype of ipcf_values is: ', type(ipcf_values))
print('The shape of ipcf_values is: ', ipcf_values.shape)

# retrieve optimal energy absorption data
energy_values = np.negative(F[:, 1])
energy_values = energy_values.reshape(-1, 1)
print('The dtype of energy_values is: ', type(energy_values))
print('The shape of energy_values is: ', energy_values.shape)

# retrieve total deformation constraint values
tdef_values = G + epsilon_value
print('The dtype of tdef_values is: ', type(tdef_values))
print('The shape of tdef_values is: ', tdef_values.shape)

# stacking all arrays along the columns
Opt_data = []
Opt_data = np.hstack((angle_sols,thickness_sols,ipcf_values,energy_values,tdef_values))

# save the data in a single file
np.savetxt('opt_prob_std' '.csv', Opt_data, delimiter = ',', fmt = '%f', header = 'Angle_sols,Thickness_Sols,obj1_IPCF,obj2_Energy,g_tdef')

#######################################################################################################################################################################
# Visualization
#######################################################################################################################################################################
# plot the design space
cm = 1/2.54
fig, ax = plt.subplots(figsize=(16*cm, 10*cm))
xl, xu = problem.bounds()
plt.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.xlim(xl[0], xu[0])
plt.ylim(xl[1], xu[1])
plt.xlabel('Angle (deg)')
plt.ylabel('Thickness (m)')
plt.title('Design Space')
fig.tight_layout()
plt.savefig('Design_Space.png', dpi=500)
plt.show()

# plot the feasible objective space
cm = 1/2.54
fig, ax = plt.subplots(figsize=(16*cm, 10*cm))
plt.scatter(np.negative(F[:, 0]), np.negative(F[:, 1]), s=30, facecolors='none', edgecolors='blue')
plt.xlabel('IPCF (kN)')
plt.ylabel('Energy (J)')
plt.title('Objective Space')
fig.tight_layout()
plt.savefig('Objective_Space.png', dpi=500)
plt.show()

#######################################################################################################################################################################
# Pareto front visualization
#######################################################################################################################################################################

# Note: Update <PATH_TO_PROJECT> to the optimization function directory with locally saved XGB_MODELS_DATABASE

file_path = r'<PATH_TO_PROJECT>\XGB_MODELS_DATABASE.csv'
data = pd.read_csv(file_path)
ipcf = data[['Peak_Impact_Force_SillAssm_N']]
ipcf = ipcf/1000.0
ipcf = ipcf.to_numpy()
energy = data[['Energy_Absorbed_J']]
energy = energy.to_numpy()

cm = 1/2.54
fig, ax = plt.subplots(figsize=(16*cm, 10*cm))
f_values = np.negative(F[:, 0])
e_values = np.negative(F[:, 1])
plt.scatter(f_values, e_values, s=30, facecolors='none', edgecolors='blue', label='Pareto solutions')
plt.scatter(ipcf, energy, c='red', label='FE Data')
plt.xlabel('IPCF (kN)')
plt.ylabel('Energy (J)')
plt.title('IPCF-Energy distributions')
plt.legend()
plt.savefig('Pareto front visualization.png')
plt.show()