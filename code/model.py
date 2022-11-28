import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


# %%
def parse_arguments():
    parser = argparse.ArgumentParser(description='dataset and hyperparameters')
    parser.add_argument('--grid_lat',dest='grid_lat',type=int,default=20)
    parser.add_argument('--grid_lng',dest='grid_lng',type=int,default=30)
    parser.add_argument('--east',dest='east',type=float,default=-122.357476)
    parser.add_argument('--west',dest='west',type=float,default=-122.514731)
    parser.add_argument('--north',dest='north',type=float,default=37.811151)
    parser.add_argument('--south',dest='south',type=float,default=37.708448)
    parser.add_argument('--cleaning',dest='cleaning',type=str,default="dataset/Cleaning_request_dataset.csv")
    parser.add_argument('--encampments',dest='encampments',type=str,default='C:/Users/tomis/OneDrive/cmu class/2022 fall/94867_DABP/project/PitStopOptimization/dataset/Encampments_dataset.csv')
    parser.add_argument('--toilets',dest='toilets',type=str,default='C:/Users/tomis/OneDrive/cmu class/2022 fall/94867_DABP/project/PitStopOptimization/dataset/public_toilet_dataset.csv')
    return parser.parse_args()

# %%
class scorer():
    def __init__(self, args):
        self.coordinate = {
            'east' : args.east,
            'west' : args.west,
            'north' : args.north,
            'south' : args.south
        }
        self.grid_lat = args.grid_lat
        self.grid_lng = args.grid_lng
        self.cleaning_path = args.cleaning
        self.encampments_path = args.encampments
        self.toilets_path = args.toilets
        
    def get_coordinate(self):
        '''
        outputs:
            matrix (dictionary):
                key: if of grid (tuple)
                value: also dictionary, coordinate of each side
        '''
        step_size_lng = (self.coordinate['east'] - self.coordinate['west'])/self.grid_lng
        step_size_lat = (self.coordinate['north'] - self.coordinate['south'])/self.grid_lat
        matrix = {}
        for i in range(self.grid_lat):
            for j in range(self.grid_lng):
                lng_west = self.coordinate['west'] + j * step_size_lng
                lng_east = self.coordinate['west'] + j * step_size_lng + step_size_lng
                lat_south = self.coordinate['south'] + i * step_size_lat
                lat_north = self.coordinate['south'] + i * step_size_lat + step_size_lat
                matrix[i, j] = {'west': lng_west, 'east':lng_east, 'south': lat_south, 'north':lat_north}
        return matrix

    def get_score(self, df, matrix):
        '''
        inputs: 
            df (pd.DataFrame): dataframe that will be aggregated
            matrix (dictionary): dictionary that maps id of a grid to coordinates  
        outputs:
            scores (np.array): scores for each grid 
        '''        
        scores = np.zeros([self.grid_lat, self.grid_lng])
        for i in range(self.grid_lat):
            for j in range(self.grid_lng):
                coordinate = matrix[i,j]
                scores[i,j] = len(df[((df['Latitude'] >= coordinate['south']) & (df['Latitude'] < coordinate['north'])) &
                                    ((df['Longitude'] >= coordinate['west']) & (df['Longitude'] < coordinate['east']))])
        return scores

    def normalize(self, scores):
        '''
        normalize score matrix
        '''
        mean = np.mean(scores)
        std = np.std(scores)
        return (scores-mean)/std

    def scores(self):
        '''
        the main function of this class.
        outputs:
            scores (list of tuples):
                each tuple will store (U_ij, S_ij, L_ij), where all of them are np.array 
        '''
        df_cleaning = pd.read_csv(self.cleaning_path)
        df_encampments = pd.read_csv(self.encampments_path)
        df_toilets = pd.read_csv(self.toilets_path)
        matrix = self.get_coordinate()
        U = self.get_score(df_cleaning, matrix)
        S = self.get_score(df_encampments, matrix)
        L = self.get_score(df_toilets, matrix)
        U_score = self.normalize(U)
        S_score = self.normalize(S)
        # L_score = self.normalize(L)
        return (U_score, S_score, L)
        

# hyperparameters
upper_bound = 5


num_district_lat = range(20)
num_district_lng = range(30)

model = gp.Model()
X = model.addVars(num_district_lat, num_district_lng, vtype=GRB.INTEGER)
args = parse_arguments()
scorer = scorer(args)
U_score, S_score, L_score = scorer.scores()
weights = np.array([0.3, 0.6])

model.setObjective(sum(weights[0] * X[i,j] * U_score[i, j] + weights[1] * X[i,j] * S_score[i, j] for i in num_district_lat for j in num_district_lng))
model.modelSense = GRB.MAXIMIZE

# lower & upper bound
for i in num_district_lat:
    for j in num_district_lng:
        model.addConstr(X[i,j] >= L_score[i,j])
        model.addConstr(X[i,j] <= upper_bound)

# budget constraint
model.addConstr(200 * sum(sum(X[i,j] for i in num_district_lat) for j in num_district_lng)<= 86000)

# optimizing model
model.optimize()

# optimal solution
print('optimal solution :')
solution = np.zeros([20,30])
for i in num_district_lat:
    for j in num_district_lng:
        solution[i,j] = X[i,j].x
result = ''
for i in num_district_lat:
    for j in num_district_lng:
        result += str(int(solution[i,j]))
    result += '\n'
print(result)


