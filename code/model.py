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
    parser.add_argument('--main_lat',dest='main_lat',type=int,default=10)
    parser.add_argument('--main_lng',dest='main_lng',type=int,default=10)
    parser.add_argument('--east',dest='east',type=float,default=-122.357476)
    parser.add_argument('--west',dest='west',type=float,default=-122.514731)
    parser.add_argument('--north',dest='north',type=float,default=37.811151)
    parser.add_argument('--south',dest='south',type=float,default=37.708448)
    parser.add_argument('--cleaning',dest='cleaning',type=str,default="dataset/Cleaning_request_dataset.csv")
    parser.add_argument('--encampments',dest='encampments',type=str,default='dataset/Encampments_dataset.csv')
    parser.add_argument('--toilets',dest='toilets',type=str,default='dataset/Existing_Pit_Stop_Locations.csv')
    parser.add_argument('--upper_bound',dest='upper_bound',type=int,default=5)
    parser.add_argument('--cont_upper_bound',dest='cont_upper_bound',type=int,default=13)
    parser.add_argument('--weight_U1',dest='weight_U1',type=int,default=100)
    parser.add_argument('--weight_U2',dest='weight_U2',type=int,default=40)
    parser.add_argument('--weight_U3',dest='weight_U3',type=int,default=10)
    parser.add_argument('--weight_S1',dest='weight_S1',type=int,default=0)
    parser.add_argument('--weight_S2',dest='weight_S2',type=int,default=0)
    parser.add_argument('--weight_S3',dest='weight_S3',type=int,default=0)
    parser.add_argument('--intercept_E1',dest='intercept_E1',type=float,default=30)
    parser.add_argument('--intercept_E2',dest='intercept_E2',type=float,default=80)
    parser.add_argument('--intercept_E3',dest='intercept_E3',type=float,default=110)
    parser.add_argument('--budget',dest='budget',type=float,default=8600)
    parser.add_argument('--contiguity_obj',dest='contiguity_obj',type=bool,default=True)
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
        self.main_lat = args.main_lat
        self.main_lng = args.main_lng
        self.cleaning_path = args.cleaning
        self.encampments_path = args.encampments
        self.toilets_path = args.toilets
        
    def get_coordinate_toilets(self):
        '''
        outputs:
            matrix (dictionary):
                key: id of grid (tuple)
                value: also dictionary, coordinate of each side
        '''
        step_size_lng = (self.coordinate['east'] - self.coordinate['west'])/self.grid_lng
        step_size_lat = (self.coordinate['north'] - self.coordinate['south'])/self.grid_lat
        matrix = {}
        for i in range(self.grid_lat):
            for j in range(self.grid_lng):
                lng_west = self.coordinate['west'] + j * step_size_lng
                lng_east = self.coordinate['west'] + j * step_size_lng + step_size_lng
                lat_north = self.coordinate['north'] - i * step_size_lat
                lat_south = self.coordinate['north'] - i * step_size_lat - step_size_lat
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
        df_cleaning = pd.read_csv(self.cleaning_path, encoding = 'utf-8')
        df_encampments = pd.read_csv(self.encampments_path, encoding = 'utf-8')
        df_toilets = pd.read_csv(self.toilets_path, encoding = 'utf-8')
        matrix = self.get_coordinate_toilets()
        U = self.get_score(df_cleaning, matrix)
        S = self.get_score(df_encampments, matrix)
        L = self.get_score(df_toilets, matrix)
        U_score = self.normalize(U)
        S_score = self.normalize(S)
        # L_score = self.normalize(L)
        return (U_score, S_score, L)

def contiguity(grid_lat, grid_lng):
    '''
    input: 
        grid_lat, grid_lng: int, which represents the shape of grids in map
    output:
        out: a numpy array of (args.grid_lat, args.grid_lng, args.grid_lat, args.grid_lng)
        where contiguity matrix for district [i,j] is stored in out[i,j] 
    '''
    out = np.zeros([grid_lat, grid_lng, grid_lat, grid_lng])
    for i in range(grid_lat):
        for j in range(grid_lng):
            if i == 0:
                if j == 0:
                    out[i,j, :i+2, :j+2] = 1
                elif j == grid_lng-1:
                    out[i,j, :i+2, j-1:] = 1
                else:
                    out[i,j, :i+2, j-1:j+2] = 1
            elif i == grid_lat-1:
                if j == 0:
                    out[i,j, i-1:, :j+2] = 1
                elif j == grid_lng-1:
                    out[i,j, i-1:, j-1:] = 1
                else:
                    out[i,j, i-1:, j-1:j+2] = 1
            else:
                if j == 0:
                    out[i,j, i-1:i+2, :j+2] = 1
                elif j == grid_lng-1:
                    out[i,j, i-1:i+2, j-1:] = 1
                else:
                    out[i,j, i-1:i+2, j-1:j+2] = 1
    return out
           

class modeler():
    def __init__(self, args):
        self.scorer = scorer(args)
        self.model = gp.Model()
        self.num_district_lat = range(self.scorer.grid_lat)
        self.num_district_lng = range(self.scorer.grid_lng)
        self.num_main_lat = range(self.scorer.main_lat)
        self.num_main_lng = range(self.scorer.main_lng)
        self.bigM = 1e7
        
    def model_setup(self, weight_U_list, weight_S_list, intercept_list, upper_bound, budget, cont_upper_bound, contiguity_obj = False):
        # get score matrix
        self.U_score, self.S_score, self.L_score = self.scorer.scores()
        self.weight_U_list = weight_U_list
        self.weight_S_list = weight_S_list
        self.intercept_list = intercept_list
                
        # set decision variables
        self.X = self.model.addVars(self.num_district_lat, self.num_district_lng, vtype=GRB.INTEGER)
        self.Y = self.model.addVars(self.num_main_lat, self.num_main_lng, vtype = GRB.BINARY) 
        self.K = self.model.addVars(self.num_district_lat, self.num_district_lng)
                
        # contiguity matrix
        conti = contiguity(self.scorer.grid_lat, self.scorer.grid_lng)
        
        # to implement spillover effects of toilets on next districts, it will change weight matrix
        if contiguity_obj:
            self.U_score = np.tensordot(conti, self.U_score)/9
            self.S_score = np.tensordot(conti, self.S_score)/9
        
        # set objective function
        self.model.setObjective(sum(self.K[i,j] for i in self.num_district_lat for j in self.num_district_lng))
        self.model.modelSense = GRB.MAXIMIZE

        # constraints for piece-wise objective function 
        for i in self.num_district_lat:
            for j in self.num_district_lng:
                for k in range(3):
                    self.model.addConstr(self.K[i,j] <= (weight_U_list[k] * self.X[i,j] * self.U_score[i, j] + weight_S_list[k] * self.X[i,j] * self.S_score[i, j] + intercept_list[k]))
        
        
        # lower & upper bound
        for i in self.num_district_lat:
            for j in self.num_district_lng:
                self.model.addConstr(self.X[i,j] >= self.L_score[i,j])
                self.model.addConstr(self.X[i,j] <= upper_bound)

        # constraints for auxiliary variables
        for p in self.num_main_lat:
            for q in self.num_main_lng:
                self.model.addConstr(self.bigM * self.Y[p,q] >= sum(self.X[i,j] for i in range(2*p, 2*p+2) for j in range(3*q, 3*q+3))) 

        # budget constraint
        self.model.addConstr(200 * (sum(self.X[i,j] for i in self.num_district_lat for j in self.num_district_lng)-self.L_score.sum()) # installation cost
                             + 60 * (sum(self.Y[p,q] for p in self.num_main_lat for q in self.num_main_lng)) # maintenance cost
                             <= budget)        
        
        # contiguity constraint
        for i in self.num_district_lat:
            for j in self.num_district_lng:
                self.model.addConstr(sum(self.X[a,b]*conti[i,j,a,b] for a in self.num_district_lat for b in self.num_district_lng) <= cont_upper_bound)
        

    def run(self):
        # optimizing model
        self.model.optimize()
        
        # optimal solution
        print('optimal solution :')
        result = ''
        total = 0
        for i in self.num_district_lat:
            for j in self.num_district_lng:
                result += str(int(self.X[i,j].x)-int(self.L_score[i,j]))
                total += int(self.X[i,j].x) - int(self.L_score[i,j])
            result += '\n'
        solution = np.zeros([self.scorer.grid_lat, self.scorer.grid_lng])
        for i in self.num_district_lat:
            for j in self.num_district_lng:
                solution[i,j] = self.X[i,j].x-self.L_score[i,j]
                
        result_path = 'results/'
        for i in [self.weight_U_list, self.weight_S_list, self.intercept_list]:
            for j in range(3):
                result_path += str(i[j])
                result_path += ','
            result_path += '_'
        np.save(result_path, solution)
        print(result)
        print('Total Number of toilets :' , total) 


    def plot_utility(self, upper_bound):
        """
        input: results from optimization model
        output: graph of utility function
        """
        x = np.linspace(0,upper_bound) # 0 to upper bound 
        for i in self.num_district_lat:
            for j in self.num_district_lng:
                for k in range(3):
                    y = (weight_U_list[k] * x + weight_S_list[k] * x + intercept_list[k])
                    plt.plot(x, y, '-b')

        plt.title('Graph of utility functions' )
        plt.xlabel('Number of toilets')
        plt.ylabel('Public Utility')
        plt.grid()
        plot_path = 'images/'
        for i in [self.weight_U_list, self.weight_S_list, self.intercept_list]:
            for j in range(3):
                plot_path += str(i[j])
                plot_path += ','
            plot_path += '_'
        plt.savefig(plot_path)
        plt.show()


if __name__ == '__main__':
    args = parse_arguments()
    model = modeler(args)
    weight_U_list = [args.weight_U1, args.weight_U2, args.weight_U3]
    weight_S_list = [args.weight_S1, args.weight_S2, args.weight_S3]
    intercept_list = [args.intercept_E1, args.intercept_E2, args.intercept_E3]
    
    model.model_setup(weight_U_list= weight_U_list, weight_S_list = weight_S_list, intercept_list = intercept_list,
                      upper_bound=args.upper_bound, budget = args.budget, 
                      cont_upper_bound = args.cont_upper_bound, contiguity_obj=args.contiguity_obj)
    model.run()
    model.plot_utility(upper_bound=args.upper_bound)