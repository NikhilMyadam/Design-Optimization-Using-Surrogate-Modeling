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
