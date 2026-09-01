# -*- coding: mbcs -*-
"""
Created by Nikhil Myadam
"""

# My FE MODEL
# Create the Leftsill_shell, Rightsill_shell, and a rigid pole impactor to simulate side pole impact simulation with explicit dynamics solver 

# MODELING LIBRARIES

from abaqus import *
from abaqusConstants import *
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
from odbAccess import *

# Libraries for data collection and visualization in Python

import numpy as np
import csv
import time

# Functions

# Part definition

def Create_Part_3D_LeftSill(point1, point2, point3, point4, point5, point6, length, part, model):
	mdb.models[model].ConstrainedSketch(name='__profile__', sheetSize=5.0)
	mdb.models[model].sketches['__profile__'].Spot(point1)
	mdb.models[model].sketches['__profile__'].Spot(point2)
	mdb.models[model].sketches['__profile__'].Spot(point3)
	mdb.models[model].sketches['__profile__'].Spot(point4)
	mdb.models[model].sketches['__profile__'].Spot(point5)
	mdb.models[model].sketches['__profile__'].Spot(point6)
	mdb.models[model].sketches['__profile__'].Line(point1, point2)
	mdb.models[model].sketches['__profile__'].VerticalConstraint(addUndoState=False, entity=
	mdb.models[model].sketches['__profile__'].geometry[2])
	mdb.models[model].sketches['__profile__'].Line(point2, point3)
	mdb.models[model].sketches['__profile__'].Line(point3, point4)
	mdb.models[model].sketches['__profile__'].VerticalConstraint(addUndoState=False, entity=
	mdb.models[model].sketches['__profile__'].geometry[4])
	mdb.models[model].sketches['__profile__'].Line(point4, point5)
	mdb.models[model].sketches['__profile__'].Line(point5, point6)
	mdb.models[model].sketches['__profile__'].VerticalConstraint(addUndoState=False, entity=
	mdb.models[model].sketches['__profile__'].geometry[6])
	mdb.models[model].Part(dimensionality=THREE_D, name=part, type=DEFORMABLE_BODY)
	mdb.models[model].parts[part].BaseShellExtrude(depth=length, sketch=mdb.models[model].sketches['__profile__'])
	del mdb.models[model].sketches['__profile__']

def Create_Set_LeftSill_Region(model, part, set_name):
	mdb.models[model].parts[part].Set(faces=mdb.models[model].parts[part].faces[:], name=set_name)

def Create_Set_LeftSill_Edges_Z0(point1, point2, point3, point4, point5, point6, model, part, set_name):
	# if using the findAt() method, specify the mid-points lying on the edges; in the function, mention these points in tuples.
	ptZ0_1 = ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2, 0.0)
	ptZ0_2 = ((point2[0] + point3[0]) / 2, (point2[1] + point3[1]) / 2, 0.0)
	ptZ0_3 = ((point3[0] + point4[0]) / 2, (point3[1] + point4[1]) / 2, 0.0)
	ptZ0_4 = ((point4[0] + point5[0]) / 2, (point4[1] + point5[1]) / 2, 0.0)
	ptZ0_5 = ((point5[0] + point6[0]) / 2, (point5[1] + point6[1]) / 2, 0.0)
	edgesZ0 = mdb.models[model].parts[part].edges.findAt(
		((ptZ0_1[0], ptZ0_1[1], ptZ0_1[2]),),
		((ptZ0_2[0], ptZ0_2[1], ptZ0_2[2]),),
		((ptZ0_3[0], ptZ0_3[1], ptZ0_3[2]),),
		((ptZ0_4[0], ptZ0_4[1], ptZ0_4[2]),),
		((ptZ0_5[0], ptZ0_5[1], ptZ0_5[2]),),
	)
	mdb.models[model].parts[part].Set(edges=edgesZ0, name=set_name)

def Create_Set_LeftSill_Edges_ZLength(point1, point2, point3, point4, point5, point6, length, model, part, set_name):
	ptZLength_1 = ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2, length)
	ptZLength_2 = ((point2[0] + point3[0]) / 2, (point2[1] + point3[1]) / 2, length)
	ptZLength_3 = ((point3[0] + point4[0]) / 2, (point3[1] + point4[1]) / 2, length)
	ptZLength_4 = ((point4[0] + point5[0]) / 2, (point4[1] + point5[1]) / 2, length)
	ptZLength_5 = ((point5[0] + point6[0]) / 2, (point5[1] + point6[1]) / 2, length)
	edgesZLength = mdb.models[model].parts[part].edges.findAt(
		((ptZLength_1[0], ptZLength_1[1], ptZLength_1[2]),),
		((ptZLength_2[0], ptZLength_2[1], ptZLength_2[2]),),
		((ptZLength_3[0], ptZLength_3[1], ptZLength_3[2]),),
		((ptZLength_4[0], ptZLength_4[1], ptZLength_4[2]),),
		((ptZLength_5[0], ptZLength_5[1], ptZLength_5[2]),),
	)
	mdb.models[model].parts[part].Set(edges=edgesZLength, name=set_name)

def Create_LeftSill_Top_Surface(top_face_pt, model, part, surface_name):
	top_face = mdb.models[model].parts[part].faces.findAt((top_face_pt,))
	mdb.models[model].parts[part].Surface(name=surface_name, side1Faces=top_face)

def Create_LeftSill_Bottom_Surface(bottom_face_pt, model, part, surface_name):
	bottom_face = mdb.models[model].parts[part].faces.findAt((bottom_face_pt,))
	mdb.models[model].parts[part].Surface(name=surface_name, side1Faces=bottom_face)

def Create_LeftSill_Surface_of_Interest(model, part, surface_name):
	# Method - creating the surface/region of interest by selecting all faces or the entire region of the part without partitioning.
	mdb.models[model].parts[part].Surface(name=surface_name, side2Faces=mdb.models[model].parts[part].faces[:])

def Create_Set_Top_Welds(model, part, top_face_pt, thk, set_name):
	# using Attachment Points along direction method, create points i.e. weld locations on the surface
	top_face = mdb.models[model].parts[part].faces.findAt((top_face_pt,))
	mdb.models[model].parts[part].AttachmentPointsAlongDirection(endPoint=(-(thk/2), 0.085, 1.9), name=set_name,
																 pointCreationMethod=AUTO_FIT, projectOnFaces=top_face,
																 setName=set_name, spacing=0.09,
																 startPoint=(-(thk/2), 0.085, 0.1))

def Create_Set_Bottom_Welds(model, part, bottom_face_pt, thk, set_name):
	# using Attachment Points along direction method, create points i.e. weld locations on the surface
	bottom_face = mdb.models[model].parts[part].faces.findAt((bottom_face_pt,))
	mdb.models[model].parts[part].AttachmentPointsAlongDirection(endPoint=(-(thk/2), -0.085, 1.9), name=set_name,
																 pointCreationMethod=AUTO_FIT,
																 projectOnFaces=bottom_face, setName=set_name,
																 spacing=0.09, startPoint=(-(thk/2), -0.085, 0.1))

def Create_Part_3D_RightSill(point7, point8, point9, point10, point11, point12, length, part, model):
	mdb.models[model].ConstrainedSketch(name='__profile__', sheetSize=5.0)
	mdb.models[model].sketches['__profile__'].Spot(point7)
	mdb.models[model].sketches['__profile__'].Spot(point8)
	mdb.models[model].sketches['__profile__'].Spot(point9)
	mdb.models[model].sketches['__profile__'].Spot(point10)
	mdb.models[model].sketches['__profile__'].Spot(point11)
	mdb.models[model].sketches['__profile__'].Spot(point12)
	mdb.models[model].sketches['__profile__'].Line(point7, point8)
	mdb.models[model].sketches['__profile__'].VerticalConstraint(addUndoState=False, entity=
	mdb.models[model].sketches['__profile__'].geometry[2])
	mdb.models[model].sketches['__profile__'].Line(point8, point9)
	mdb.models[model].sketches['__profile__'].Line(point9, point10)
	mdb.models[model].sketches['__profile__'].VerticalConstraint(addUndoState=False, entity=
	mdb.models[model].sketches['__profile__'].geometry[4])
	mdb.models[model].sketches['__profile__'].Line(point10, point11)
	mdb.models[model].sketches['__profile__'].Line(point11, point12)
	mdb.models[model].sketches['__profile__'].VerticalConstraint(addUndoState=False, entity=
	mdb.models[model].sketches['__profile__'].geometry[6])
	mdb.models[model].Part(dimensionality=THREE_D, name=part, type=DEFORMABLE_BODY)
	mdb.models[model].parts[part].BaseShellExtrude(depth=length, sketch=mdb.models[model].sketches['__profile__'])
	del mdb.models[model].sketches['__profile__']

def Create_Set_RightSill_Region(model, part, set_name):
	mdb.models[model].parts[part].Set(faces=mdb.models[model].parts[part].faces[:], name=set_name)

def Create_Set_RightSill_Edges_Z0(point7, point8, point9, point10, point11, point12, model, part, set_name):
	ptZ0_1 = ((point7[0] + point8[0]) / 2, (point7[1] + point8[1]) / 2, 0.0)
	ptZ0_2 = ((point8[0] + point9[0]) / 2, (point8[1] + point9[1]) / 2, 0.0)
	ptZ0_3 = ((point9[0] + point10[0]) / 2, (point9[1] + point10[1]) / 2, 0.0)
	ptZ0_4 = ((point10[0] + point11[0]) / 2, (point10[1] + point11[1]) / 2, 0.0)
	ptZ0_5 = ((point11[0] + point12[0]) / 2, (point11[1] + point12[1]) / 2, 0.0)
	edgesZ0 = mdb.models[model].parts[part].edges.findAt(
		((ptZ0_1[0], ptZ0_1[1], ptZ0_1[2]),),
		((ptZ0_2[0], ptZ0_2[1], ptZ0_2[2]),),
		((ptZ0_3[0], ptZ0_3[1], ptZ0_3[2]),),
		((ptZ0_4[0], ptZ0_4[1], ptZ0_4[2]),),
		((ptZ0_5[0], ptZ0_5[1], ptZ0_5[2]),),
	)
	mdb.models[model].parts[part].Set(edges=edgesZ0, name=set_name)

def Create_Set_RightSill_Edges_ZLength(point7, point8, point9, point10, point11, point12, length, model, part,
									   set_name):
	ptZLength_1 = ((point7[0] + point8[0]) / 2, (point7[1] + point8[1]) / 2, length)
	ptZLength_2 = ((point8[0] + point9[0]) / 2, (point8[1] + point9[1]) / 2, length)
	ptZLength_3 = ((point9[0] + point10[0]) / 2, (point9[1] + point10[1]) / 2, length)
	ptZLength_4 = ((point10[0] + point11[0]) / 2, (point10[1] + point11[1]) / 2, length)
	ptZLength_5 = ((point11[0] + point12[0]) / 2, (point11[1] + point12[1]) / 2, length)
	edgesZLength = mdb.models[model].parts[part].edges.findAt(
		((ptZLength_1[0], ptZLength_1[1], ptZLength_1[2]),),
		((ptZLength_2[0], ptZLength_2[1], ptZLength_2[2]),),
		((ptZLength_3[0], ptZLength_3[1], ptZLength_3[2]),),
		((ptZLength_4[0], ptZLength_4[1], ptZLength_4[2]),),
		((ptZLength_5[0], ptZLength_5[1], ptZLength_5[2]),),
	)
	mdb.models[model].parts[part].Set(edges=edgesZLength, name=set_name)

def Create_RightSill_Top_Surface(top_face_pt, model, part, surface_name):
	top_face = mdb.models[model].parts[part].faces.findAt((top_face_pt,))
	mdb.models[model].parts[part].Surface(name=surface_name, side2Faces=top_face)

def Create_RightSill_Bottom_Surface(bottom_face_pt, model, part, surface_name):
	bottom_face = mdb.models[model].parts[part].faces.findAt((bottom_face_pt,))
	mdb.models[model].parts[part].Surface(name=surface_name, side2Faces=bottom_face)

def Create_RightSill_Surface_of_Interest(model, part, surface_name):
	mdb.models[model].parts[part].Surface(name=surface_name, side2Faces=mdb.models[model].parts[part].faces[:])

def Create_Part_3D_RigidImpactor(point13, point14, revolution, model, part, set_name):
	mdb.models[model].ConstrainedSketch(name='__profile__', sheetSize=5.0)
	mdb.models[model].sketches['__profile__'].Spot(point=(0.0, 0.0))
	mdb.models[model].sketches['__profile__'].ConstructionLine(point1=(0.0, -5.0), point2=(0.0, 5.0))
	mdb.models[model].sketches['__profile__'].FixedConstraint(
		entity=mdb.models[model].sketches['__profile__'].geometry[2])
	mdb.models[model].sketches['__profile__'].rectangle(point13, point14)
	mdb.models[model].Part(dimensionality=THREE_D, name=part, type=DISCRETE_RIGID_SURFACE)
	mdb.models[model].parts[part].BaseShellRevolve(angle=revolution, flipRevolveDirection=OFF,
												   sketch=mdb.models[model].sketches['__profile__'])
	del mdb.models[model].sketches['__profile__']
	mdb.models[model].parts[part].ReferencePoint(point=(0.0, 0.0, 0.0))
	mdb.models[model].parts[part].Set(name=set_name,
									  referencePoints=(mdb.models[model].parts[part].referencePoints[2],))

def Create_Set_RigidImpactor_Region(model, part, set_name):
	mdb.models[model].parts[part].Set(faces=mdb.models[model].parts[part].faces[:], name=set_name)

def Create_RigidImpactor_Inertia(point13, point14, model, mass_value, part, set_name):
	# inertia values calculated assuming a cylinder of uniform density, i.e. a mass 'm' with a radius 'r' and height 'h'
	# values calculated considering the setup of the simulation; central rotating axis is Y-axis with impact direction in X-axis
	impactor_radius = point14[0] - point13[0]
	impactor_height = point13[1] - point14[1]
	i_xx = 0.25 * (mass_value) * (3 * (impactor_radius ** 2) + (impactor_height ** 2))
	i_yy = 0.5 * (mass_value) * (impactor_radius ** 2)
	i_zz = 0.25 * (mass_value) * (3 * (impactor_radius ** 2) + (impactor_height ** 2))
	mdb.models[model].parts[part].engineeringFeatures.PointMassInertia(alpha=0.0, composite=0.0, i11=i_xx, i22=i_yy,
																	   i33=i_zz, mass=mass_value, name='ImpactorMass',
																	   region=mdb.models[model].parts[part].sets[
																		   set_name])

# Property definition

# Material definition

def Create_Material_Data(model):
	mdb.models[model].Material(name='DP780 Steel')
	mdb.models[model].materials['DP780 Steel'].Density(table=((7890.0,),))
	mdb.models[model].materials['DP780 Steel'].Elastic(table=((200E9, 0.3),))
	mdb.models[model].materials['DP780 Steel'].Plastic(scaleStress=None, table=((500E6, 0.0),))

# Section definition

def Create_Section_Data(model, thk_value):
	mdb.models[model].HomogeneousShellSection(idealization=NO_IDEALIZATION, integrationRule=SIMPSON,
											  material='DP780 Steel', name='SillSection', nodalThicknessField='',
											  numIntPts=5, poissonDefinition=DEFAULT, preIntegrate=OFF,
											  temperature=GRADIENT, thickness=thk_value, thicknessField='',
											  thicknessModulus=None, thicknessType=UNIFORM, useDensity=OFF)

# Property assignment/Section assignment

def Create_Property_Data(model, part1, part2, set_name1, set_name2):
	mdb.models[model].parts[part1].SectionAssignment(offset=0.0, offsetField='', offsetType=MIDDLE_SURFACE,
													 region=mdb.models[model].parts[part1].sets[set_name1],
													 sectionName='SillSection', thicknessAssignment=FROM_SECTION)
	mdb.models[model].parts[part2].SectionAssignment(offset=0.0, offsetField='', offsetType=MIDDLE_SURFACE,
													 region=mdb.models[model].parts[part2].sets[set_name2],
													 sectionName='SillSection', thicknessAssignment=FROM_SECTION)

# Assembly definition

def Create_Assembly(model, part1, part2, part3, instance1, instance2, instance3, thick):
	mdb.models[model].rootAssembly.DatumCsysByDefault(CARTESIAN)
	mdb.models[model].rootAssembly.Instance(dependent=ON, name=instance1, part=mdb.models[model].parts[part1])
	mdb.models[model].rootAssembly.Instance(dependent=ON, name=instance2, part=mdb.models[model].parts[part2])
	mdb.models[model].rootAssembly.Instance(dependent=ON, name=instance3, part=mdb.models[model].parts[part3])
	# defining a gap of 10mm between impactor and sill assembly
	mdb.models[model].rootAssembly.translate(instanceList=(instance3,), vector=((0.187 + (thick/2)), 0.0, 0.0))
	mdb.models[model].rootAssembly.translate(instanceList=(instance3,), vector=(0.0, 0.0, 1.25))

# Step and Output requests' definition

def Create_Analysis_Step(model, step, duration):
	mdb.models[model].ExplicitDynamicsStep(improvedDtMethod=ON, name=step, previous='Initial', timePeriod=duration)

	# defining field output request for the whole model in the loading step
	mdb.models[model].fieldOutputRequests['F-Output-1'].setValues(numIntervals=20, variables=(
	'S', 'SVAVG', 'MISES', 'E', 'PE', 'PEVAVG', 'PEEQ', 'PEEQVAVG', 'LE', 'UT'))

	# defining history output requests in the loading step
	mdb.models[model].HistoryOutputRequest(createStepName='Loading', name='H-Output-1', variables=PRESELECT)
	mdb.models[model].HistoryOutputRequest(createStepName='Loading', name='H-Output-2', rebar=EXCLUDE,
										   region=mdb.models[model].rootAssembly.allInstances['Rigid_Impactor'].sets[
											   'RP'], sectionPoints=DEFAULT, variables=('U1', 'U2', 'U3', 'A1', 'A2', 'A3'))
	mdb.models[model].HistoryOutputRequest(createStepName='Loading', name='H-Output-3', rebar=EXCLUDE,
										   region=mdb.models[model].rootAssembly.allInstances['Left_Sill'].surfaces[
											   'LS-SOI'], sectionPoints=DEFAULT,
										   variables=('CFNM', 'CFN1', 'CFN2', 'CFN3'))
	mdb.models[model].HistoryOutputRequest(createStepName='Loading', name='H-Output-4', rebar=EXCLUDE,
										   region=mdb.models[model].rootAssembly.allInstances['Right_Sill'].surfaces[
											   'RS-SOI'], sectionPoints=DEFAULT,
										   variables=('CFNM', 'CFN1', 'CFN2', 'CFN3'))

# Interaction definition

def Create_Interaction_Property(model, property_name, f_value):
	mdb.models[model].ContactProperty(property_name)
	mdb.models[model].interactionProperties[property_name].TangentialBehavior(dependencies=0, directionality=ISOTROPIC,
																			  elasticSlipStiffness=None,
																			  formulation=PENALTY, fraction=0.005,
																			  maximumElasticSlip=FRACTION,
																			  pressureDependency=OFF,
																			  shearStressLimit=None,
																			  slipRateDependency=OFF,
																			  table=((f_value,),),
																			  temperatureDependency=OFF)
	mdb.models[model].interactionProperties[property_name].NormalBehavior(allowSeparation=ON,
																		  constraintEnforcementMethod=DEFAULT,
																		  pressureOverclosure=HARD)

def Create_Interaction(model, step, interaction_name, property_name):
	mdb.models[model].ContactExp(createStepName=step, name=interaction_name)
	mdb.models[model].interactions[interaction_name].includedPairs.setValuesInStep(stepName=step, useAllstar=ON)
	mdb.models[model].interactions[interaction_name].contactPropertyAssignments.appendInStep(
		assignments=((GLOBAL, SELF, property_name),), stepName=step)

def Create_Top_Fasteners_RigidMPCs(model, fastener_name, instance1, instance2, welds_set, face1, face2):
	mdb.models[model].rootAssembly.engineeringFeatures.PointFastener(adjustOrientation=OFF, connectionType=BEAM_MPC,
																	 influenceRadius=0.01, name=fastener_name,
																	 physicalRadius=0.01, region=
																	 mdb.models[model].rootAssembly.instances[
																		 instance1].sets[welds_set], targetSurfaces=(
		mdb.models[model].rootAssembly.instances[instance1].surfaces[face1],
		mdb.models[model].rootAssembly.instances[instance2].surfaces[face2]), unsorted=OFF)

def Create_Bottom_Fasteners_RigidMPCs(model, fastener_name, instance1, instance2, welds_set, face1, face2):
	mdb.models[model].rootAssembly.engineeringFeatures.PointFastener(adjustOrientation=OFF, connectionType=BEAM_MPC,
																	 influenceRadius=0.01, name=fastener_name,
																	 physicalRadius=0.01, region=
																	 mdb.models[model].rootAssembly.instances[
																		 instance1].sets[welds_set], targetSurfaces=(
		mdb.models[model].rootAssembly.instances[instance1].surfaces[face1],
		mdb.models[model].rootAssembly.instances[instance2].surfaces[face2]), unsorted=OFF)

# Load definition

def Create_BC_Fixed_Support(model, instance1, instance2, ls_setZ0, ls_setZL, rs_setZ0, rs_setZL, fs_1, fs_2, fs_3,
							fs_4):
	# fixing the edges of both the sill parts to constrain the entire assembly at its ends
	mdb.models[model].EncastreBC(createStepName='Initial', localCsys=None, name=fs_1,
								 region=mdb.models[model].rootAssembly.instances[instance1].sets[ls_setZ0])
	mdb.models[model].EncastreBC(createStepName='Initial', localCsys=None, name=fs_2,
								 region=mdb.models[model].rootAssembly.instances[instance1].sets[ls_setZL])
	mdb.models[model].EncastreBC(createStepName='Initial', localCsys=None, name=fs_3,
								 region=mdb.models[model].rootAssembly.instances[instance2].sets[rs_setZ0])
	mdb.models[model].EncastreBC(createStepName='Initial', localCsys=None, name=fs_4,
								 region=mdb.models[model].rootAssembly.instances[instance2].sets[rs_setZL])

def Create_BC_Displacement_Rotation(model, step, bc_name, instance3, rp_set):
	# defining a displacement/rotation BC to specify the state of impact
	mdb.models[model].DisplacementBC(amplitude=UNSET, createStepName=step, distributionType=UNIFORM, fieldName='',
									 fixed=OFF, localCsys=None, name=bc_name,
									 region=mdb.models[model].rootAssembly.instances[instance3].sets[rp_set], u1=UNSET,
									 u2=0.0, u3=UNSET, ur1=0.0, ur2=0.0, ur3=0.0)

def Create_Predefined_Field_Velocity(model, field_name, instance3, rp_set):
	# defining the initial velocity of the rigid impactor
	mdb.models[model].Velocity(distributionType=MAGNITUDE, field='', name=field_name, omega=0.0,
							   region=mdb.models[model].rootAssembly.instances[instance3].sets[rp_set],
							   velocity1=-12.56, velocity3=3.36)

# Mesh definition 	

def Create_Mesh_LeftSill(model, part, seed_size):
	# mdb.models[model].parts[part].Set(faces=mdb.models[model].parts[part].faces[:], name=set_name) - for selecting the entire region of the part for mesh
	mdb.models[model].parts[part].setElementType(elemTypes=(
	ElemType(elemCode=S4R, elemLibrary=EXPLICIT, secondOrderAccuracy=OFF, hourglassControl=DEFAULT),
	ElemType(elemCode=S3R, elemLibrary=EXPLICIT)), regions=(mdb.models[model].parts[part].faces[:],))
	mdb.models[model].parts[part].seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=seed_size)
	mdb.models[model].parts[part].generateMesh()

def Create_Mesh_RightSill(model, part, seed_size):
	mdb.models[model].parts[part].setElementType(elemTypes=(
	ElemType(elemCode=S4R, elemLibrary=EXPLICIT, secondOrderAccuracy=OFF, hourglassControl=DEFAULT),
	ElemType(elemCode=S3R, elemLibrary=EXPLICIT)), regions=(mdb.models[model].parts[part].faces[:],))
	mdb.models[model].parts[part].seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=seed_size)
	mdb.models[model].parts[part].generateMesh()

def Create_Mesh_RigidImpactor(model, part, seed_size):
	mdb.models[model].parts[part].setElementType(
		elemTypes=(ElemType(elemCode=R3D4, elemLibrary=EXPLICIT), ElemType(elemCode=R3D3, elemLibrary=EXPLICIT)),
		regions=(mdb.models[model].parts[part].faces[:],))
	mdb.models[model].parts[part].seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=seed_size)
	mdb.models[model].parts[part].generateMesh()

# Setup the working directory

# Replace <PATH_TO_PROJECT> with the appropriate local project path.

import os
print("Current working directory: {0}".format(os.getcwd()))
if os.path.isdir('Model_Data_99degrees') == True:
  print("Directory already exist")
else:
    os.mkdir('Model_Data_99degrees') #  desired directory name
    print("Created a directory - Model_Data_99degrees")

print("Current working directory: {0}".format(os.getcwd()))
os.chdir(r"<PATH_TO_PROJECT>/Model_Data_99degrees")
print("Current working directory: {0}".format(os.getcwd()))

# Start the loop for the modeling database i.e. myModel

Iterations = 17 # number of models with flange angle 99 degrees

# Dictionaries/list to store the model name and parameters

sim_models = {"Impact_Model": [], "Length": [], "Angle_degrees": [], "Thickness": [], "Mesh_Part": [], "Mesh_Impactor": [], "Mass_Impactor": [], "Dynamic_Job": [], "Time_req_s": []}
dynamic_jobs = [] # Add job id in the list jobs

# parameters to iterate for the database
thickness_values = list(np.linspace(0.0008, 0.0024, num=17)) # models generated with thicknesses from 0.8mm to 2.4mm

def fe_model(i):

	start_time = time.time()

    # add the model id in the dictionary 'sim_models' 
	sim_models["Impact_Model"].append('Dy_Model-%d' %(i + 1))
	myThickness = thickness_values[i] 

	# variables for the impact simulation

	myLength = 2.0
	myAngle = 99.0
	myMass = 100.0
	mySeed_part = 0.006 # seed size 6mm from the mesh sensitivity study
	mySeed_impactor = 0.015

	# Insert the model name and parameters in the dictionary 'sim_models'

	sim_models["Length"].append('%f' %(myLength))
	sim_models["Angle_degrees"].append('%f' %(myAngle))
	sim_models["Thickness"].append('%f' %(myThickness))
	sim_models["Mass_Impactor"].append('%f' %(myMass))
	sim_models["Mesh_Part"].append('%f' %(mySeed_part))
	sim_models["Mesh_Impactor"].append('%f' %(mySeed_impactor))

	# Model creation

	myString = sim_models["Impact_Model"][i]
	myModel = mdb.Model(name=myString)

	# geometry points for left sill creation
	myPoint1 = (-(myThickness/2), 0.1)
	myPoint2 = (-(myThickness/2), 0.07)
	myPoint3 = (-0.05, 0.06216)
	myPoint4 = (-0.05, -0.06216)
	myPoint5 = (-(myThickness/2), -0.07)
	myPoint6 = (-(myThickness/2), -0.1)
	# geometry points for right sill creation
	myPoint7 = ((myThickness/2), 0.1)
	myPoint8 = ((myThickness/2), 0.07)
	myPoint9 = (0.05, 0.06216)
	myPoint10 = (0.05, -0.06216)
	myPoint11 = ((myThickness/2), -0.07)
	myPoint12 = ((myThickness/2), -0.1)
	# geometry points for rigid impactor creation
	myPoint13 = (0.0, 0.15)
	myPoint14 = (0.127, -0.15)
	myRevolution = 360.0
	# points for surface locations to model spot welds
	LS_Top_face_pt = (-(myThickness/2), 0.085, 1.0)
	LS_Bottom_face_pt = (-(myThickness/2), -0.085, 1.0)
	RS_Top_face_pt = ((myThickness/2), 0.085, 1.0)
	RS_Bottom_face_pt = ((myThickness/2), -0.085, 1.0)
	myPart1 = 'LeftSill'
	myPart2 = 'RightSill'
	myPart3 = 'RigidImpactor'

	# assembly parameters

	myInstance1 = 'Left_Sill'
	myInstance2 = 'Right_Sill'
	myInstance3 = 'Rigid_Impactor'

	# analysis parameters

	myStep = 'Loading'
	myTimePeriod = 0.09

	# interaction parameters

	myInteraction = 'Impact'
	myProperty = 'Frictional' 
	friction_coefficient = 0.20

	# Creating the Finite Element Model

	# Left Sill
	Create_Part_3D_LeftSill(myPoint1, myPoint2, myPoint3, myPoint4, myPoint5, myPoint6, myLength, myPart1, myString)
	Create_Set_LeftSill_Region(myString, myPart1, 'LS-Region')
	Create_Set_LeftSill_Edges_Z0(myPoint1, myPoint2, myPoint3, myPoint4, myPoint5, myPoint6, myString, myPart1, 'LSEdges-Z0')
	Create_Set_LeftSill_Edges_ZLength(myPoint1, myPoint2, myPoint3, myPoint4, myPoint5, myPoint6, myLength, myString, myPart1, 'LSEdges-ZLength')
	Create_LeftSill_Top_Surface(LS_Top_face_pt, myString, myPart1, 'LS-TopSurface')
	Create_LeftSill_Bottom_Surface(LS_Bottom_face_pt, myString, myPart1, 'LS-BottomSurface')
	Create_LeftSill_Surface_of_Interest(myString, myPart1, 'LS-SOI')
	Create_Set_Top_Welds(myString, myPart1, LS_Top_face_pt, myThickness, 'TopWelds')
	Create_Set_Bottom_Welds(myString, myPart1, LS_Bottom_face_pt, myThickness, 'BottomWelds')

	# Right Sill
	Create_Part_3D_RightSill(myPoint7, myPoint8, myPoint9, myPoint10, myPoint11, myPoint12, myLength, myPart2, myString)
	Create_Set_RightSill_Region(myString, myPart2, 'RS-Region')
	Create_Set_RightSill_Edges_Z0(myPoint7, myPoint8, myPoint9, myPoint10, myPoint11, myPoint12, myString, myPart2, 'RSEdges-Z0')
	Create_Set_RightSill_Edges_ZLength(myPoint7, myPoint8, myPoint9, myPoint10, myPoint11, myPoint12, myLength, myString, myPart2, 'RSEdges-ZLength')
	Create_RightSill_Top_Surface(RS_Top_face_pt, myString, myPart2, 'RS-TopSurface')
	Create_RightSill_Bottom_Surface(RS_Bottom_face_pt, myString, myPart2, 'RS-BottomSurface')
	Create_RightSill_Surface_of_Interest(myString, myPart2, 'RS-SOI')

	# Rigid Impactor
	Create_Part_3D_RigidImpactor(myPoint13, myPoint14, myRevolution, myString, myPart3, 'RP')
	Create_Set_RigidImpactor_Region(myString, myPart3, 'RI-Region')
	Create_RigidImpactor_Inertia(myPoint13, myPoint14, myString, myMass, myPart3, 'RP')

	# Material definition
	Create_Material_Data(myString)

	# Section definition
	Create_Section_Data(myString, myThickness)
	Create_Property_Data(myString, myPart1, myPart2, 'LS-Region', 'RS-Region')

	# Assembly creation
	Create_Assembly(myString, myPart1, myPart2, myPart3, myInstance1, myInstance2, myInstance3, myThickness)

	# Analysis steps
	Create_Analysis_Step(myString, myStep, myTimePeriod)

	# Interaction definition
	Create_Interaction_Property(myString, myProperty, friction_coefficient)
	Create_Interaction(myString, myStep, myInteraction, myProperty)
	Create_Top_Fasteners_RigidMPCs(myString, 'TopFasteners', myInstance1, myInstance2, 'TopWelds', 'LS-TopSurface', 'RS-TopSurface')
	Create_Bottom_Fasteners_RigidMPCs(myString, 'BottomFasteners', myInstance1, myInstance2, 'BottomWelds', 'LS-BottomSurface', 'RS-BottomSurface')

	# Load definition
	Create_BC_Fixed_Support(myString, myInstance1, myInstance2, 'LSEdges-Z0', 'LSEdges-ZLength', 'RSEdges-Z0', 'RSEdges-ZLength', 'FS-LSZ0', 'FS-LSZL', 'FS-RSZ0', 'FS-RSZL')
	Create_BC_Displacement_Rotation(myString, myStep, 'ImpactCondition', myInstance3, 'RP')
	Create_Predefined_Field_Velocity(myString, 'ImpactVelocity', myInstance3, 'RP')

	# Mesh
	Create_Mesh_LeftSill(myString, myPart1, mySeed_part)
	Create_Mesh_RightSill(myString, myPart2, mySeed_part)
	Create_Mesh_RigidImpactor(myString, myPart3, mySeed_impactor)

	# Job and Calculation

	# add the job id in the list jobs
	sim_models["Dynamic_Job"].append('Dynamic_Job-%d' %(i + 1))
	dynamic_jobs.append('Dynamic_Job-%d' %(i + 1))

	# Job definition

	mdb.Job(activateLoadBalancing=False, atTime=None, contactPrint=OFF, description='', echoPrint=OFF, explicitPrecision=SINGLE, historyPrint=OFF, memory=90, memoryUnits=PERCENTAGE, model=myString, modelPrint=OFF, multiprocessingMode=DEFAULT, name=dynamic_jobs[i], nodalOutputPrecision=SINGLE, numCpus=1, numDomains=1, numThreadsPerMpiProcess=1, parallelizationMethodExplicit=DOMAIN, queue=None, resultsFormat=ODB, scratch='', type=ANALYSIS, userSubroutine='', waitHours=0, waitMinutes=0)
	# run the job created
	mdb.jobs[dynamic_jobs[i]].submit(consistencyChecking=OFF, datacheckJob=True)
	# to not return control till the job is finished
	mdb.jobs[dynamic_jobs[i]].waitForCompletion()

	# Save model database as CAE file

	mdb.saveAs('Model_99degrees.cae')
	mdb.jobs[dynamic_jobs[i]].submit(consistencyChecking=OFF)
	mdb.jobs[dynamic_jobs[i]].waitForCompletion()
	    
	end_time = time.time()
	time_req_s = (end_time - start_time)
	sim_models['Time_req_s'].append('%f' %(time_req_s))

	# writing the model data to the keys/lists in the dictionary 'sim_models'
	writer.writerow([sim_models[x][i] for x in key_list])
	print([sim_models[x] for x in key_list])

	# OUTPUT DATA PARSING
	# Create seperate output file with energies and reaction/contact forces for every job 
	
	# Post-processing

	# Open new viewport
	# setting viewport width and height to defaults values
	session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=120.0, height=80.0)
	session.viewports['Viewport: 1'].makeCurrent()
	session.viewports['Viewport: 1'].maximize()

	# specify the directory address and search for the necessary odb file(s) - to change as needed
	Job_id = ('Dynamic_Job-%d' %(i + 1))
	# creating an odb object by opening the output database whose path is provided as an argument
	odb = session.openOdb(name='Dynamic_Job-%d.odb' %(i + 1))
	# setting the display/viewport to the selected output database
	session.viewports['Viewport: 1'].setValues(displayedObject=odb)

	# setting the display rendering style

	# statements to change the viewport display to the deformed model and setting the plot state
	session.viewports['Viewport: 1'].odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF, ))
	session.viewports['Viewport: 1'].view.setValues(session.views['Iso'])
	session.viewports['Viewport: 1'].odbDisplay.commonOptions.setValues(deformationScaling=UNIFORM)

	# creating screenshots of results of Von-Mises stress and Displacements/deformations
	# printing only the current viewport with its background
	session.printOptions.setValues(vpDecorations=OFF, vpBackground=ON)
	session.printToFile(fileName='Job-VM Stress-%d' %(i + 1), format=PNG, canvasObjects=(session.viewports['Viewport: 1'], ))
	session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(variableLabel='UT', outputPosition=NODAL, refinement=(INVARIANT, 'Magnitude'))
	session.printToFile(fileName='Job-Total Deformations-%d' %(i + 1), format=PNG, canvasObjects=(session.viewports['Viewport: 1'], ))

	# New variables for data storage and history outputs
	# accessing history outputs from the last step of the simulation and for the entire model
	step = 'Loading'
	region = 'Assembly ASSEMBLY'

	# access the entire history output data and store the necessary variables in a single array
	HO = []
	# Artificial strain energy: ALLAE for Whole Model
	ALLAE = np.array(odb.steps[step].historyRegions[region].historyOutputs['ALLAE'].data)
	# Internal energy: ALLIE for Whole Model
	ALLIE = np.array(odb.steps[step].historyRegions[region].historyOutputs['ALLIE'].data)
	# Kinetic energy: ALLKE for Whole Model
	ALLKE = np.array(odb.steps[step].historyRegions[region].historyOutputs['ALLKE'].data)
	# Total energy of the output set: ETOTAL for Whole Model
	ETOTAL = np.array(odb.steps[step].historyRegions[region].historyOutputs['ETOTAL'].data)
	
	# NOTE: The rigid impactor RP node number is model-dependent.
	# Verify and update the RP node number corresponding to the generated FE model.

	# X-Displacement of the rigid impactor RP node set
	U1_PI = np.array(odb.steps[step].historyRegions['Node RIGID_IMPACTOR.1599'].historyOutputs['U1'].data)
	# Z-Displacement of the rigid impactor RP node set
	U3_PI = np.array(odb.steps[step].historyRegions['Node RIGID_IMPACTOR.1599'].historyOutputs['U3'].data)
	# Extracting contact normal forces from both the surfaces of interactions
	# Access the history regions in the whole model for the specified step
	history_regions = odb.steps[step].historyRegions
	# Getting the number of keys or history regions in the specified step
	num_history_regions = len(history_regions)
	# Create a list to specify the variables of interest
	variables_of_interest = [
	    'CFN1 on surface ASSEMBLY_LEFT_SILL_LS-SOI',
	    'CFN3 on surface ASSEMBLY_LEFT_SILL_LS-SOI',
	    'CFN1 on surface ASSEMBLY_RIGHT_SILL_RS-SOI',
	    'CFN3 on surface ASSEMBLY_RIGHT_SILL_RS-SOI'
	]
	# Initialize the variables to store data
	CFN1_LS_SOI = None
	CFN3_LS_SOI = None
	CFN1_RS_SOI = None
	CFN3_RS_SOI = None
	# Iterate through variables of interest
	for variable in variables_of_interest:
	# Iterate through history regions
	    for j in range(num_history_regions):
	# Accessing the history region through key
		history_region_key = list(odb.steps[step].historyRegions.keys())[j]
		history_region = odb.steps[step].historyRegions[history_region_key]
	# Check if the variable exists in the history output
		if variable in history_region.historyOutputs.keys():
	# Access the data of the variable
		    variable_data = np.array(history_region.historyOutputs[variable].data)
	# Assign the data to the corresponding variables
		    if variable == 'CFN1 on surface ASSEMBLY_LEFT_SILL_LS-SOI':
		        CFN1_LS_SOI = variable_data
		    elif variable == 'CFN3 on surface ASSEMBLY_LEFT_SILL_LS-SOI':
		        CFN3_LS_SOI = variable_data
		    elif variable == 'CFN1 on surface ASSEMBLY_RIGHT_SILL_RS-SOI':
		    	CFN1_RS_SOI = variable_data
		    elif variable == 'CFN3 on surface ASSEMBLY_RIGHT_SILL_RS-SOI':
		    	CFN3_RS_SOI = variable_data
		    break
	    else:
		print(r"Error: {variable} not found for any history region.")

	# stacking all the arrays in a horizontal sequence (i.e. along the columns) - use np.hstack()

	HO = np.hstack((ALLAE,ALLIE,ALLKE,ETOTAL,U1_PI,U3_PI,CFN1_LS_SOI,CFN1_RS_SOI,CFN3_LS_SOI,CFN3_RS_SOI))

	# Save the history data in a single file i.e. a text file for every job

	np.savetxt(Job_id + '_Energies_CFNs' '.csv', HO, delimiter = ',', fmt = '%f', header = 'Time_steps,Artificial_Energy,Time_steps,Internal_Energy,Time_steps,Kinetic_Energy,Time_steps,Total_Energy,Time_steps,U1_Disp_Imp,Time_steps,U3_Disp_Imp,Time_steps,CFN1_LS_SOI,Time_steps,CFN1_RS_SOI,Time_steps,CFN3_LS_SOI,Time_steps,CFN3_RS_SOI')


with open('model_database_99degrees_dynamic_impact_sims.csv','wb') as testfile:
	writer = csv.writer(testfile)
	key_list = ["Impact_Model", "Length", "Angle_degrees", "Thickness", "Mesh_Part", "Mesh_Impactor", "Mass_Impactor", "Dynamic_Job", "Time_req_s"]
	writer.writerow(key_list)

	for i in range(Iterations):
		fe_model(i)    
		        
testfile.close()