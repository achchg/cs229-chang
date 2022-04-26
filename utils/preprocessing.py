import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd



def add_random_noise(torch_tensor, std):
    '''add gaussian noise to original data sample as sample batch effect
    '''
    torch_tensor1 = torch.clone(torch_tensor)
    sampled_noise = torch.randn(torch_tensor1.shape)
    torch_tensor1 += std*sampled_noise
    return torch_tensor1

def train_test_split(data, test_size, state):
    '''random split or datasets with seed
    '''
    np.random.seed(state)
    n = len(data)
    n_train = int((1 - test_size) * n)
    
    mnist_train, mnist_rest = data[:n_train], data[n_train:]
    return mnist_train, mnist_rest

def noisy_torch(torch_train_list, torch_rest_list, std_list, state):
    '''add random noise to training torch
    '''
    torch.manual_seed(state)
    
    
    transform  = transforms.Normalize((0.5,), (0.5,))
    trainX = []
    trainy = []
    trainX_rest = []
    trainy_rest = []
    
    for ds, data in enumerate(torch_train_list):
        trainX_ds = []
        trainy_ds = []
        for sample in data:
            trainX_ds.append(add_random_noise(sample[0], std = std_list[ds]))
            trainy_ds.append(torch.tensor([sample[1]]))
            
        trainX_ds = torch.vstack(trainX_ds)
        trainy_ds = torch.vstack(trainy_ds)
        
#         print(trainX_ds.shape)
        trainX_ds = transform(trainX_ds)
        trainX.append(trainX_ds)
        trainy.append(trainy_ds)
    
    for ds, data in enumerate(torch_rest_list):
        trainX_rest_ds = []
        trainy_rest_ds = []
        for sample in data:
            trainX_rest_ds.append(add_random_noise(sample[0], std = std_list[ds]))
            trainy_rest_ds.append(torch.tensor([sample[1]]))
            
        trainX_rest_ds = torch.vstack(trainX_rest_ds)
        trainy_rest_ds = torch.vstack(trainy_rest_ds)
        
#         print(trainX_rest_ds.shape)
        trainX_rest_ds = transform(trainX_rest_ds)
        trainX_rest.append(trainX_rest_ds)
        trainy_rest.append(trainy_rest_ds)
    
    
        
    trainX = torch.vstack(trainX)
#     print(trainX.shape)
    trainy = torch.vstack(trainy)
    trainX_rest = torch.vstack(trainX_rest)
    trainy_rest = torch.vstack(trainy_rest)
    
    
    
    
    return (trainX, 
            trainX_rest, 
            trainy.reshape(-1,), 
            trainy_rest.reshape(-1,))


def generate_label(label_array):
    ''' generate label for FashionMNIST
    '''
    label = pd.DataFrame(label_array)
    label.columns = ['item']
    label['count'] = 1
    label = label.groupby('item')\
                 .count()
    return label

def define_category(label_array, category_wt, item):
    ''' define category for later selection bias purpose
    '''
    category_wt_dictionary = pd.DataFrame.from_dict(category_wt).T.reset_index()
    category_wt_dictionary.columns = ['category', 'selection_wt']
    category_wt = category_wt_dictionary.set_index('category')

    item_dictionary = pd.DataFrame.from_dict(item).T
    item_dictionary.columns = ['type', 'category']

    label = generate_label(label_array)
    true_distribution = label.join(item_dictionary, how = 'left', on = 'item').reset_index()\
                             .set_index('category')\
                             .join(category_wt, how = 'left', on = 'category').reset_index()\
                             .assign(item_pct = lambda x: x['count']/x['count'].sum())
    return(true_distribution)


def create_item_list(y):
    ''' create item list for later selection bias purpose
    '''
    array = y
    n = len(array)
    n_item = len(set(array))

    item_list = []
    for item in range(n_item):
        item_element = []
        for i in range(n):
            if array[i] == item:
                item_element.append(i)
        item_list.append(item_element)
        
    return(item_list)

def create_biased_sample(X, y, item_list, true_distribution):
    
    y_selected, X_selected, y_unselected, X_unselected = list(), list(), list(), list()
    n_item = len(item_list)
    
    for item in range(n_item):
        
        p = true_distribution['selection_wt'][item]
        index_item = item_list[item]
        selected = np.random.binomial(1, p, len(y[index_item]))
        y_s = y[index_item][selected == 1]
        X_s = X[index_item][selected == 1]
        y_uns = y[index_item][selected == 0]
        X_uns = X[index_item][selected == 0]

        y_selected.append(y_s)
        X_selected.append(X_s)
        y_unselected.append(y_uns)
        X_unselected.append(X_uns)

    X_selected = np.vstack(X_selected)
    y_selected = np.concatenate(y_selected,axis=0)
    X_unselected = np.vstack(X_unselected)
    y_unselected = np.concatenate(y_unselected,axis=0)
    
    return(X_selected, y_selected, X_unselected, y_unselected)


def torch_to_numpy(torch_train, torch_rest):
    
    trainX = []
    trainy = []
    trainX_rest = []
    trainy_rest = []
    for sample in torch_train:
        trainX.append(sample[0].cpu().detach().numpy())
        trainy.append(sample[1])
    
    for sample in torch_rest:
        trainX_rest.append(sample[0].cpu().detach().numpy())
        trainy_rest.append(sample[1])
        
    trainX = np.vstack(trainX)
    trainy = np.vstack(trainy)
    trainX_rest = np.vstack(trainX_rest)
    trainy_rest = np.vstack(trainy_rest)
    
    return (trainX, 
            trainX_rest, 
            trainy.reshape(-1,), 
            trainy_rest.reshape(-1,))

def generate_data_mixture(state, mnist_train, ds_size_list, std_list, category_wt, item, batch_size):
    torch_train_list = []
    torch_rest_list = []
    size = 0
    start = [0]
    for ds in range(len(ds_size_list)):
        mnist_train_1, mnist_rest_1 = train_test_split(mnist_train, ds_size_list[ds], 123)
        size += len(mnist_train_1)
        start.append(size)
        torch_train_list.append(mnist_train_1)
        torch_rest_list.append(mnist_rest_1)
    
#     print(len(torch_train_list[0]) + len(torch_train_list[1]) + len(torch_train_list[2]))
    
    (trainX_mixture, trainX_rest_mixture, 
     trainy_mixture, trainy_rest_mixture) = noisy_torch(torch_train_list, torch_rest_list, std_list, state)
    
#     print(trainX_mixture.shape)
    
    mnist_train_mixture = loaders(trainX_mixture, trainy_mixture, batch_size)[1]
    mnist_train_rest_mixture = loaders(trainX_rest_mixture, trainy_rest_mixture, batch_size)[1]
    

    trainX, trainX_rest, trainy, trainy_rest = torch_to_numpy(mnist_train_mixture, mnist_train_rest_mixture)
#     print(start)
    obs_X_list = []
    obs_y_list = []
    nonobs_X_list = []
    nonobs_y_list = []
    
    for ds in range(len(ds_size_list)):
        trainX_1 = trainX[start[ds]:start[ds+1]]
        trainy_1 = trainy[start[ds]:start[ds+1]]
        true_distribution_1 = define_category(trainy_1, category_wt[ds], item)
        item_list_1 = create_item_list(trainy_1)
        
        (trainX_1_selected, trainy_1_selected,\
         trainX_1_unselected, trainy_1_unselected) = create_biased_sample(trainX_1, trainy_1, 
                                                                          item_list_1, true_distribution_1)
        
        
        obs_X_list.append(trainX_1_selected)
        obs_y_list.append(trainy_1_selected)
        nonobs_X_list.append(trainX_1_unselected)
        nonobs_y_list.append(trainy_1_unselected)
    
    return (obs_X_list, obs_y_list, nonobs_X_list, nonobs_y_list)


def loaders(torch_trainX, torch_trainy, batch_size):
    import torch
    mnist_train_mixture = []
    
    for n in range(len(torch_trainX)):
        mnist_train_mixture.append((torch_trainX[n].unsqueeze(dim=0), torch_trainy[n]))
    
    train_loader = torch.utils.data.DataLoader(mnist_train_mixture, 
                                               batch_size=batch_size)
    
    return train_loader, mnist_train_mixture
    

def create_largest(obs_X_list, obs_y_list, nonobs_X_list, nonobs_y_list, ds_num):
    trainX_selected = np.vstack(obs_X_list)
    trainy_selected = np.concatenate(obs_y_list, axis = 0)
    trainX_unselected = np.vstack(nonobs_X_list)
    trainy_unselected = np.concatenate(nonobs_y_list, axis = 0)

    torch_trainX_selected = torch.from_numpy(trainX_selected)
    torch_trainy_selected = torch.from_numpy(trainy_selected)
    torch_trainX_unselected = torch.from_numpy(trainX_unselected)
    torch_trainy_unselected = torch.from_numpy(trainy_unselected)
    
    indx_from = 0
    indx_to = len(obs_X_list[0])
        
    for i in range(1, ds_num):
        indx_from += len(obs_X_list[i-1])
        indx_to += len(obs_X_list[i])
        
    trainX_largest = torch_trainX_selected[indx_from:indx_to]
    trainy_largest = torch_trainy_selected[indx_from:indx_to]
    trainX_rest_largest = torch_trainX_unselected[indx_from:indx_to]
    trainy_rest_largest = torch_trainy_unselected[indx_from:indx_to]
    
    return trainX_largest, trainy_largest, trainX_rest_largest, trainy_rest_largest
    

    
def create_mixture(obs_X_list, obs_y_list, nonobs_X_list, nonobs_y_list):
    trainX_selected = np.vstack(obs_X_list)
    trainy_selected = np.concatenate(obs_y_list, axis = 0)
    trainX_unselected = np.vstack(nonobs_X_list)
    trainy_unselected = np.concatenate(nonobs_y_list, axis = 0)

    torch_trainX_selected = torch.from_numpy(trainX_selected)
    torch_trainy_selected = torch.from_numpy(trainy_selected)
    torch_trainX_unselected = torch.from_numpy(trainX_unselected)
    torch_trainy_unselected = torch.from_numpy(trainy_unselected)
    
    return torch_trainX_selected, torch_trainy_selected, torch_trainX_unselected, torch_trainy_unselected


def generate_data_mixture_base(state, mnist_train, ds_size_list, std_list, category_wt, item, batch_size):
    '''generate data mixture that output also the remaining list'''
    torch_train_list = []
    torch_rest_list = []
    size = 0
    start = [0]
    for ds in range(len(ds_size_list)):
        mnist_train_1, mnist_rest_1 = train_test_split(mnist_train, ds_size_list[ds], 123)
        size += len(mnist_train_1)
        start.append(size)
        torch_train_list.append(mnist_train_1)
        torch_rest_list.append(mnist_rest_1)
    
#     print(len(torch_train_list[0]) + len(torch_train_list[1]) + len(torch_train_list[2]))
    
    (trainX_mixture, trainX_rest_mixture, 
     trainy_mixture, trainy_rest_mixture) = noisy_torch(torch_train_list, torch_rest_list, std_list, state)
    
#     print(trainX_mixture.shape)
    
    mnist_train_mixture = loaders(trainX_mixture, trainy_mixture, batch_size)[1]
    mnist_train_rest_mixture = loaders(trainX_rest_mixture, trainy_rest_mixture, batch_size)[1]
    

    trainX, trainX_rest, trainy, trainy_rest = torch_to_numpy(mnist_train_mixture, mnist_train_rest_mixture)
#     print(start)
    obs_X_list = []
    obs_y_list = []
    nonobs_X_list = []
    nonobs_y_list = []
    rest_X_list = []
    rest_y_list = []
    
    for ds in range(len(ds_size_list)):
        trainX_1 = trainX[start[ds]:start[ds+1]]
        trainy_1 = trainy[start[ds]:start[ds+1]]
        true_distribution_1 = define_category(trainy_1, category_wt[ds], item)
        item_list_1 = create_item_list(trainy_1)
        
        (trainX_1_selected, trainy_1_selected,\
         trainX_1_unselected, trainy_1_unselected) = create_biased_sample(trainX_1, trainy_1, 
                                                                          item_list_1, true_distribution_1)
        
        
        obs_X_list.append(trainX_1_selected)
        obs_y_list.append(trainy_1_selected)
        nonobs_X_list.append(trainX_1_unselected)
        nonobs_y_list.append(trainy_1_unselected)
        rest_X_list.append(trainX_rest)
        rest_y_list.append(trainy_rest)
    
    return (obs_X_list, obs_y_list, nonobs_X_list, nonobs_y_list, rest_X_list, rest_y_list)




def generate_data_mixture_2D(state, mnist_train, ds_size_list, std_list, category_wt, item, batch_size):
    '''generate data mixture in 2-D for additive model'''
    torch_train_list = []
    torch_rest_list = []
    size = 0
    start = [0]
    size_rest = 0
    start_rest = [0]
    for ds in range(len(ds_size_list)):
        mnist_train_1, mnist_rest_1 = train_test_split(mnist_train, ds_size_list[ds], 123)
        size += len(mnist_train_1)
        start.append(size)
        size_rest += len(mnist_rest_1)
        start_rest.append(size_rest)
        torch_train_list.append(mnist_train_1)
        torch_rest_list.append(mnist_rest_1)

    
    (trainX_mixture, trainX_rest_mixture, 
     trainy_mixture, trainy_rest_mixture) = noisy_torch(torch_train_list, torch_rest_list, std_list, state)
    
    mnist_train_mixture = loaders(trainX_mixture, trainy_mixture, batch_size)[1]
    mnist_train_rest_mixture = loaders(trainX_rest_mixture, trainy_rest_mixture, batch_size)[1]
    

    trainX, trainX_rest, trainy, trainy_rest = torch_to_numpy(mnist_train_mixture, mnist_train_rest_mixture)
        
    obs_X_list = []
    obs_y_list = []
    nonobs_X_list = []
    nonobs_y_list = []
    obs_X2D_list = []
    nonobs_X2D_list = []
    rest_X_list = []
    rest_y_list = []
    rest_X2D_list = []

    for ds in range(len(ds_size_list)):
        trainX_1 = trainX[start[ds]:start[ds+1]]
        trainy_1 = trainy[start[ds]:start[ds+1]]
        true_distribution_1 = define_category(trainy_1, category_wt[ds], item)
        item_list_1 = create_item_list(trainy_1)

        (trainX_1_selected, trainy_1_selected,\
         trainX_1_unselected, trainy_1_unselected) = create_biased_sample(trainX_1, trainy_1, 
                                                                          item_list_1, true_distribution_1)
        
        trainX_rest


        obs_X2D_list.append(flatten_n_addGrpLabel(trainX_1_selected, ds+1))
        nonobs_X2D_list.append(flatten_n_addGrpLabel(trainX_1_unselected, ds+1))
        obs_X_list.append(trainX_1_selected)
        obs_y_list.append(trainy_1_selected)
        nonobs_X_list.append(trainX_1_unselected)
        nonobs_y_list.append(trainy_1_unselected)
        rest_X_list.append(trainX_rest)
        rest_y_list.append(trainy_rest)
        rest_X2D_list.append(flatten_n_addGrpLabel(trainX_rest, ds+1))
        
    return (obs_X_list, obs_y_list, 
            nonobs_X_list, nonobs_y_list, 
            obs_X2D_list, nonobs_X2D_list,
            rest_X_list, rest_y_list, rest_X2D_list)


def flatten_n_addGrpLabel(X_3D, grp_label):
    n_sample = X_3D.shape[0]
    X_2D = X_3D.reshape(n_sample,-1)
    grp_label = np.ones((n_sample, 1))*grp_label
    X_2D = np.hstack((X_2D, grp_label))
    
    return X_2D