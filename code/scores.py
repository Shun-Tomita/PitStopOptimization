# %%
import pandas as pd
import numpy as np
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
    parser.add_argument('--cleaning',dest='cleaning',type=str,default='Cleaning_request_dataset.csv')
    parser.add_argument('--encampments',dest='encampments',type=str,default='Encampments_dataset.csv')
    parser.add_argument('--toilets',dest='toilets',type=str,default='public_toilet_dataset.csv')
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
        self.cleaning_path = 'dataset/'+args.cleaning
        self.encampments_path = 'dataset/'+args.encampments
        self.toilets_path = 'dataset/'+args.toilets
        
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
        L_score = self.normalize(L)
        return (U_score, S_score, L_score)
        




# %%
if __name__ == '__main__':
    args = parse_arguments()
    scorer = scorer(args)
    scores = scorer.scores()
    print(f'grid size: {args.grid_lat} x {args.grid_lng}')    
    print('U_score:')
    print(scores[0])
    print('S_score:')
    print(scores[1])
    print('L_score:')
    print(scores[2])

