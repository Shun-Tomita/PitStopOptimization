import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scores import scorer, parse_arguments

# hyperparameters
upper_bound = 5


model = gb.Model()
num_district_lng = range(30)
num_district_lat = range(20)

x = model.addVars(num_district_lat, num_district_lng, vtype=GRB.INTEGER)
args = parse_arguments()
scorer = scorer(args)
U_score, S_score, L_score = scorer.scores()
weights = np.array([0.5, 0.3])

model.setObjective(sum(weights[0] * x[i,j] * U_score[i, j] 
                       + weights[1] * x[i,j] * S_score[i, j] 
                       for i in num_district_lat for j in num_district_lng))
model.modelSense = GRB.MAXIMIZE

# lower & upper bound
for i in num_district_lat:
    for j in num_district_lng:
        model.addConstr(x[i,j] >= L_score[i,j])
        model.addConstr(x[i,j] <= upper_bound)

# budget constraint
model.addConstr(sum(x[i,j]*200 for i in num_district_lat for j in num_district_lng)<= 8600)

# optimizing model
model.optimize()

# optimal solution
print('optimal solution :')
solution = np.zeros([20,30])
for i in num_district_lat:
    for j in num_district_lng:
        solution[i,j] = x[i,j].x
print(solution)


