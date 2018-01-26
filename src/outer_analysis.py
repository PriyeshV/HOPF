import numpy as np

dataset = 'movielens'
metric = 'f1'

path = '../../aws/outer/'
metrics = {'bae': 8, 'f1': 4, 'accuracy': 6}

def find(self, name, path):
    # recursively search for any file in a folder
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def get_data(folder, dataset):
    pos = metrics[metric]
    results = np.zeros((5,5)) #k x folds x outer_epoch
    for depth in range(1,6):
        res = np.load(folder+dataset+str(depth)+'_batch_results.npy').item()
        print(res)
        print()
        for fold in range(5):
            results[depth-1][fold] = res[fold][0][pos]
            #print(res[fold-1][0][pos])

    return results



results = get_data(path+dataset+'/', dataset)
print(np.mean(results, axis=0) )