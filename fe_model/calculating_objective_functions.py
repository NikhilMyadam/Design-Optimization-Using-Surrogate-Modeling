# File to automate the data parsing of output files (i.e. excel files of energies and contact/reaction forces) and calculate necessary output parameters

import pandas as pd
import matplotlib.pyplot as plt
from numpy import trapz
import numpy as np
import math

M = 17 # number of models/iterations

# calculate the peak impact force, total deformation and energy absorbed for each FE model from every job file

# considering only the right sill part for calculation of peak impact forces
maxRS_CFN1 = [] # max because the reaction forces from right sill are in +ve X direction
maxRS_CFN3 = [] # max because the reaction forces from right sill change in Z direction when the KE becomes 0
t_impactf = [] # to store the total i.e. peak impact force
minU1_RP = [] # min because the impactor moves in -ve X direction until the KE becomes 0
maxU3_RP = [] # max because the impactor moves in +ve Z direction
t_deformation = [] # to store the total i.e. maximum deformation
energy = []

for j in range(1, M+1):

	data = pd.read_csv("Dynamic_Job-%d_Energies_CFNs.csv" %(j))
	t = data["Time_steps"][:]
	AE = data["Artificial_Energy"][:]
	IE = data["Internal_Energy"][:]
	CFN1_RS = data["CFN1_RS_SOI"][:]
	CFN3_RS = data["CFN3_RS_SOI"][:]
	U1_RP = data["U1_Disp_Imp"][:]
	U3_RP = data["U3_Disp_Imp"][:]
	
	# calculating necessary quantities
	
	# peak impact/reaction forces
	MAXRS_N1 = round(CFN1_RS[:90].max(),5) # considering the first 90 elements of the array - closely corresponding to 40ms when KE is 0
	maxRS_CFN1.append(MAXRS_N1)
	MAXRS_N3 = round(CFN3_RS[:90].max(),5) # considering the first 90 elements of the array - closely corresponding to 40ms when KE is 0
	maxRS_CFN3.append(MAXRS_N3)
	R_force = ((MAXRS_N1) * (MAXRS_N1)) + ((MAXRS_N3) * (MAXRS_N3))
	P_force = math.sqrt(R_force)
	t_impactf.append(P_force) # calculating the total peak impact force from the right sill part i.e. the sill assembly

	# maximum displacement of the impactor i.e. deformation of the sill assembly
	MINU1_RP = round(U1_RP.min(),4)-0.01 # subtracting the initial gap between the impactor and sill assembly considering the impact angle
	minU1_RP.append(MINU1_RP)
	MAXU3_RP = round(U3_RP.max(),4)-0.002 # subtracting the initial gap between the impactor and sill assembly considering the impact angle
	maxU3_RP.append(MAXU3_RP)
	R_deformation = ((MINU1_RP) * (MINU1_RP)) + ((MAXU3_RP) * (MAXU3_RP))
	max_deformation = math.sqrt(R_deformation)
	t_deformation.append(max_deformation) # calculating the total deformation of the sill assembly

	# energy absorbed by the sill assembly
	EA = round((IE[200]-AE[200]),5)
	energy.append(EA)

# update the calculated values to the generated file of model and their parameters

# read the file
df = pd.read_csv('model_database_95degrees_dynamic_impact_sims.csv')
# Add columns in the right to add the peak impact force, deformation of the assembly, energy absorbed
df["Peak_Impact_Force_RS_CFN1_N"] = maxRS_CFN1
df["Peak_Impact_Force_RS_CFN3_N"] = maxRS_CFN3
df["Peak_Impact_Force_SillAssm_N"] = t_impactf
df["Maximum_Deformation_X_m"] = minU1_RP
df["Maximum_Deformation_Z_m"] = maxU3_RP
df["Maximum_Deformation_m"] = t_deformation
df["Energy_Absorbed_J"] = energy

# make sure that the above columns/quantities are added to the previously generated file

print(isinstance(df, dict))
print(df.keys())

# save the new dataframe/file to excel and csv files with a new name

df.to_excel("MODELS_DATABASE_95degrees.xlsx")
df.to_csv("MODELS_DATABASE_95degrees.csv")