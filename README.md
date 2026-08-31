# Multi-Objective Design Optimization of a Vehicle Rocker Sill Assembly

## Table of Contents

[1. Project Description](#1-project-description)<br><br>
[2. Problem Statement](#2-problem-statement)<br><br>
[3. Numerical Simulation](#3-numerical-simulation)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.1 Rocker Sill Assembly and Design Variables](#31-rocker-sill-assembly-and-design-variables)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.2 Crashworthiness Metrics](#32-crashworthiness-metrics)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.3 FE Model](#33-fe-model)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[3.4 Automated FE Simulation and Database Generation](#34-automated-fe-simulation-and-database-generation)<br><br>
[4. Surrogate Model](#4-surrogate-model)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.1 Correlation Analysis](#41-correlation-analysis)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.2 XGBoost Regression Models](#42-xgboost-regression-models)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[4.3 Surrogate Model Deployment](#43-surrogate-model-deployment)<br><br>
[5. Multi-Objective Design Optimization](#5-multi-objective-design-optimization)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.1 Optimization Problem Formulation](#51-optimization-problem-formulation)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.2 Optimization Function](#52-optimization-function)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[5.3 Pareto-Optimal Solutions](#53-pareto-optimal-solutions)<br><br>
[6. Project Outcome](#6-project-outcome)<br><br>
[7. Acknowledgements](#7-acknowledgements)<br><br>
[8. References](#8-references)

---

## 1. Project Description

This project investigates the optimal design configurations of a **vehicle rocker sill assembly** under the **European New Car Assessment Programme (Euro NCAP) oblique pole side impact crashworthiness requirements**.

The dynamic bending performance of the rocker sill assembly is investigated using a **multi-objective optimization approach** that combines physics-based **finite element (FE) simulations** with **surrogate modeling techniques**.

A surrogate-based optimization function is formulated using engineering constraints, and evolutionary algorithms are used to explore the design space and generate a diverse set of solutions that simultaneously optimize multiple, conflicting crashworthiness objectives.

## 2. Problem Statement

Vehicle design is a multidisciplinary process in which numerical simulations play a central role. Among the various disciplines involved, **crashworthiness** is critical to vehicle development because it evaluates the structural performance of the **body-in-white (BIW)** and its ability to protect occupants during a collision.

Among several crash scenarios, lateral vehicle collisions have received increased attention in recent years. According to Euro NCAP, vehicles are tested by impacting them against a fixed, rigid pole with a diameter of 254 mm (10 inches). The vehicle travels at a speed up to and including 32 km/h (20 mph) and is positioned at an angle of 75° formed by the impact reference line with the vertical plane through the centre of gravity of the head of the dummy in the driver seating position. The test condition is depicted in {test condition}.

This study focuses on optimizing the design of a rocker sill assembly for a **Class B battery electric vehicle (BEV)** subjected to an **oblique pole side impact**.

The available design volume of the rocker sill assembly is constrained by the overall vehicle architecture and is derived by considering:

1. Battery size and shape
2. Passenger seating position
3. Other BIW modules, such as the central compartment floor position

## 3. Numerical Simulation

### 3.1 Rocker Sill Assembly and Design Variables

The design of the rocker sill assembly in a BEV is influenced by battery topology, size, its location and other architectural considerations while also adhering to specific design philosophies and vehicle concepts. As a result, the assembly has a constrained design volume, with design choices also inherently restricted by the assembly's role in supporting pillar assemblies and other BIW modules.

Based on benchmarking studies and engineering design knowledge, a **2 m long, 0.1 m wide and 0.2 m high hexagonal-section rocker sill assembly** is investigated in this study.

Two design variables are considered:

- **Angle** - the geometric parameter defining the shape of the rocker sill section
- **Thickness** - the component parameter defining the panel thickness

The baseline model, including the design variables, is depicted in {baseline rocker sill FE model}.
