# File to automate the data parsing of output files (i.e. excel files of energies and contact/reaction forces) and create the required graphs

import pandas as pd
import matplotlib.pyplot as plt
from numpy import trapz
import numpy as np

M = 17 # number of models/iterations

# read the job output files (i.e. csv files) and create the necessary graphs for energies and contact/reaction forces

for i in range(1, M+1):
	data = pd.read_csv("Dynamic_Job-%d_Energies_CFNs.csv" %(i))
	t = data["Time_steps"][:]
	AE = data["Artificial_Energy"][:]
	IE = data["Internal_Energy"][:]
	KE = data["Kinetic_Energy"][:]
	TE = data["Total_Energy"][:]
	CFN1_LS = data["CFN1_LS_SOI"][:]
	CFN1_RS = data["CFN1_RS_SOI"][:]
	CFN3_LS = data["CFN3_LS_SOI"][:]
	CFN3_RS = data["CFN3_RS_SOI"][:]
	U1_RP = data["U1_Disp_Imp"][:]
	U3_RP = data["U3_Disp_Imp"][:]
	
	# plot energy
	plt.plot(t, AE, color='b', label='Artificial Strain Energy')
	plt.plot(t, IE, color='g', label='Internal Energy')
	plt.plot(t, KE, color='r', label='Kinetic Energy')
	plt.plot(t, TE, color='k', label='Total Energy')
	plt.legend()
	plt.title("Energies - Time")
	plt.xlabel('Time (s)')
	plt.ylabel('Energy (J)')
	plt.savefig('Job-%d_Energies.png' %(i))
	plt.close()    # close the figure window
	
	# plot contact normal forces
	plt.plot(t, CFN1_LS, color='violet', label='Contact Force N1 of LS')
	plt.legend()
	plt.title("Contact Force - Time")
	plt.xlabel('Time (s)')
	plt.ylabel('Contact Force (N)')
	plt.savefig('Job-%d_CFN1_LS_Time.png' %(i))
	plt.close()    # close the figure window
	
	plt.plot(t, CFN1_RS, color='violet', label='Contact Force N1 of RS')
	plt.legend()
	plt.title("Contact Force - Time")
	plt.xlabel('Time (s)')
	plt.ylabel('Contact Force (N)')
	plt.savefig('Job-%d_CFN1_RS_Time.png' %(i))
	plt.close()    # close the figure window
	
	plt.plot(t, CFN3_LS, color='violet', label='Contact Force N3 of LS')
	plt.legend()
	plt.title("Contact Force - Time")
	plt.xlabel('Time (s)')
	plt.ylabel('Contact Force (N)')
	plt.savefig('Job-%d_CFN3_LS_Time.png' %(i))
	plt.close()    # close the figure window
	
	plt.plot(t, CFN3_RS, color='violet', label='Contact Force N3 of RS')
	plt.legend()
	plt.title("Contact Force - Time")
	plt.xlabel('Time (s)')
	plt.ylabel('Contact Force (N)')
	plt.savefig('Job-%d_CFN3_RS_Time.png' %(i))
	plt.close()    # close the figure window
	
	# plot displacement i.e. deformation of the sill assembly
	plt.plot(t, U1_RP, color='violet', label='X-Displacement of Impactor')
	plt.legend()
	plt.title("Displacement(X) - Time")
	plt.xlabel('Time (s)')
	plt.ylabel('Displacement (m)')
	plt.savefig('Job-%d_U1_RP_Time.png' %(i))
	plt.close()    # close the figure window
	
	plt.plot(t, U3_RP, color='violet', label='Z-Displacement of Impactor')
	plt.legend()
	plt.title("Displacement(Z) - Time")
	plt.xlabel('Time (s)')
	plt.ylabel('Displacement (m)')
	plt.savefig('Job-%d_U3_RP_Time.png' %(i))
	plt.close()    # close the figure window