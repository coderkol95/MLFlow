from sklearn.datasets import make_regression
import json
import numpy as np

with open('config.json','r') as f:
    config = json.load(f)
RANDOM_STATE=config['RANDOM STATE']

def _generate_train_data():
    X,y=make_regression(n_samples=100, n_features=5, n_informative=4, noise=0.5, random_state=RANDOM_STATE)
    np.savetxt(fname="data/X.csv",X=X, delimiter=',')
    np.savetxt(fname="data/y.csv",X=y, delimiter=',')

def generate_live_data(size:int=1):
    X,y=make_regression(n_samples=size, n_features=5, n_informative=4, shuffle=False, random_state=RANDOM_STATE)
    return X,y

if __name__=="__main__":
    _generate_train_data()