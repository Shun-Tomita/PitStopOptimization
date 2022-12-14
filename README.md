# PitStopOptimization
To address growing concerns of public hygiene and to meet the demand of public toilets, the city of San Francisco started the Pit Stop public toilet program in 2014, which aims to provide clean public toilets for everyone, and provide a safe spot to dispose of used needles and bags to dispose dog’s waste. To meet the increasing need for public toilets, in addition to the already installed 31 Pit Stops, we want to find out the optimal number and the location to install additional Pit Stops in the San Francisco area using Integer Linear Programming (ILP). 

## Data
In identifying the candidate locations and the number of Pit Stops, we hypothesize that putting Pit Stops in areas with many 311 cleaning cases (uncleanliness) or large homeless populations (susceptibility) will increase public utility. With regard to our two hypotheses, we mainly used the SF311 Cases dataset which is provided by the San Francisco city government. Also, we used data of locations of existing Pit Stops. Sample raw dataset is [here](https://github.com/Shun-Tomita/PitStopOptimization/blob/main/dataset/311_Cases_1121_sampled1000.csv) and we preprocessed data [here](https://github.com/Shun-Tomita/PitStopOptimization/blob/main/code/Get_Datasets_from_311.ipynb) and preprocessed data is stored [here](https://github.com/Shun-Tomita/PitStopOptimization/tree/main/dataset).

## Model
To solve the Pit Stops allocation problem, we implemented the ILP problem by dividing the San Francisco region into 20 x 30 grids and tried to find the optimal number of Pit Stops in each cell. Each cell is as large as 8 blocks in practice and it would take 10 minutes to walk from one end to the other. According to the hypotheses mentioned above, our utility function is a function of uncleanliness and susceptibility and our model computes the optimal number of Pit Stops that maximizes the utility function. Since marginal utility diminishes and our utility function needs to be linear in terms of our decision variables, a piecewise linear function is used to implement concave utility function with three different slopes. Also, we implemented several constraints such as budgets constraints, upper bound constraints, and lower bound constraints. We tried different weights and intercepts to model different assumptions in our utility functions.

## Code
Make sure your environment has [gurobipy](https://pypi.org/project/gurobipy/) and [folium](https://python-visualization.github.io/folium/installing.html#installation). To run our model, you can run following scripts:

```
python3 .\code\model.py --weight_U1 100 --weight_U2 50 --weight_U3 20 --weight_S1 0 --weight_S2 0 --weight_S3 0 --intercept_E1 1000 --intercept_E2 1050 --intercept_E3 1100
```

where weights and intercepts are parameters you can tune. You will get the result printed in your terminal and stored in [results folder](https://github.com/Shun-Tomita/PitStopOptimization/tree/main/results). If you want to visualize your result, use [this notebook](https://github.com/Shun-Tomita/PitStopOptimization/blob/main/code/generate_grid.ipynb)
