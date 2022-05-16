import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd


class DataPreprocessing:
    """data preprocessing to generate data mixture
    """
    def __init__(self, mnist_train, ds_size_list, std_list, mean_list, category_wt, item):
        """_summary_

        Args:
            mnist_train (list): training data input from torchvision.datasets
            ds_size_list (list): list of size to show data mixture distribution
            std_list (list): list of std to show variation exists in data mixture
            mean_list (list): list of mean to center of datasets in data mixture
            category_wt (_type_): _description_
            item (_type_): _description_
        """
        self.data_train = mnist_train
        self.ds_size_list = ds_size_list
        self.std_list = std_list
        self.mean_list = mean_list
        self.category_wt = category_wt
        self.item = item
        self.trainX_mixture = None
        self.trainy_mixture = None
        self.train_mixture  = None
        self.trainX_selected = None
        self.trainy_selected = None
        self.trainX_unselected = None
        self.trainy_unselected = None

    #### Batch-effect simulation ####

    def __add_random_noise(self, torch_tensor, std, mean=0):
        """add gaussian noise to original data sample as sample batch effect

        Args:
            torch_tensor (tensor): input tensor of original data samples
            std (float): batch-effect standard deviation
            mean (float, optional): batch-effect mean value. Defaults to 0.

        Returns:
            tensor: output tensor of original data samples
        """
        torch_tensor_output = torch.clone(torch_tensor)
        sampled_noise = torch.randn(torch_tensor_output.shape)
        torch_tensor_output += std*sampled_noise + mean
        return torch_tensor_output
    
    def __noisy_torch(self, state, torch_train_list):
        """add random noise to training torch

        Args:
            state (int): random seed
            torch_train_list (torch list): train torch
        """
        torch.manual_seed(state)
        
        
        transform  = transforms.Normalize((0.5,), (0.5,))
        trainX = []
        trainy = []
        
        for ds, data in enumerate(torch_train_list):
            trainX_ds = []
            trainy_ds = []
            for sample in data:
                trainX_ds.append(self.__add_random_noise(sample[0], std = self.std_list[ds], mean=self.mean_list[ds]))
                trainy_ds.append(torch.tensor([sample[1]]))
                
            trainX_ds = torch.vstack(trainX_ds)
            trainy_ds = torch.vstack(trainy_ds)
            
            trainX_ds = transform(trainX_ds)
            trainX.append(trainX_ds)
            trainy.append(trainy_ds)
        
        trainX = torch.vstack(trainX)
        trainy = torch.vstack(trainy)

        return (trainX, trainy.reshape(-1,))
    
     #### Data split, load, transform ####

    def __train_test_split(self, state, train_set):
        """random split or datasets with seed

        Args:
            state (int): random seed
            train_set (float): size of train set

        Returns:
            tuple: train and test sets
        """
        np.random.seed(state)
        n = len(self.data_train)
        n_train = int((train_set) * n)
        
        mnist_train, mnist_rest = self.data_train[:n_train], self.data_train[n_train:]
        return mnist_train, mnist_rest

    def __loaders(self, batch_size):
        import torch
        mnist_train_mixture = []
        
        for n in range(len(self.trainX_mixture)):
            mnist_train_mixture.append((self.trainX_mixture[n].unsqueeze(dim=0), self.trainy_mixture[n]))
        
        train_loader = torch.utils.data.DataLoader(mnist_train_mixture, 
                                                batch_size=batch_size)
        
        return train_loader, mnist_train_mixture
    
    def __torch_to_numpy(self):
        """torch to numpy of train mixture

        Returns:
            tuple: numpy train X and y 
        """
        trainX = []
        trainy = []

        for sample in self.train_mixture:
            trainX.append(sample[0].cpu().detach().numpy())
            trainy.append(sample[1])
        
        trainX = np.vstack(trainX)
        trainy = np.vstack(trainy)
        
        return (trainX, trainy.reshape(-1,))

    
    #### Data label ####
    def __generate_label(self, label_array):
        """generate label (y) for training data


        Args:
            label_array (array): training data y per data in the mixture

        Returns:
            DataFrame: label counts
        """
        label = pd.DataFrame(label_array)
        label.columns = ['item']
        label['count'] = 1
        label = label.groupby('item')\
                    .count()
        return label

    def __define_category(self, label_array, ds):
        """define category for later selection bias purpose

        Args:
            label_array (array): training data y per data in the mixture
        """
        category_wt_dictionary = pd.DataFrame.from_dict(self.category_wt[ds]).T.reset_index()
        category_wt_dictionary.columns = ['category', 'selection_wt']
        category_wt = category_wt_dictionary.set_index('category')

        item_dictionary = pd.DataFrame.from_dict(self.item).T
        item_dictionary.columns = ['type', 'category']

        label = self.__generate_label(label_array)
        true_distribution = label.join(item_dictionary, how = 'left', on = 'item').reset_index()\
                                .set_index('category')\
                                .join(category_wt, how = 'left', on = 'category').reset_index()\
                                .assign(item_pct = lambda x: x['count']/x['count'].sum())
        return(true_distribution)
    
    def __create_item_list(self, label_array):
        """create item list for later selection bias purpose

        Args:
            label_array (array): training data y per data in the mixture
        """
        array = label_array
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

    def __create_biased_sample(self, X, y, item_list, true_distribution):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_): _description_
            item_list (_type_): _description_
            true_distribution (_type_): _description_
        """
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

    def generate_data_mixture_base(self, state, batch_size):
        """generate data mixture

        Args:
            state (int): random seed
            batch_size (int): batch size
        """
        torch_train_list = list()
        size = 0
        start = [0]

        # load datasets to form data mixture
        for ds in range(len(self.ds_size_list)):
            mnist_train_output, mnist_rest_output = self.__train_test_split(state, self.ds_size_list[ds])
            size += len(mnist_train_output)
            start.append(size)
            torch_train_list.append(mnist_train_output)
        
        # add batch effect
        (self.trainX_mixture, self.trainy_mixture) = self.__noisy_torch(state, torch_train_list,)
        
        # load train mixture 
        self.train_mixture = self.__loaders(batch_size)[1]
        trainX, trainy = self.__torch_to_numpy()
        
        obs_X_list = []
        obs_y_list = []
        nonobs_X_list = []
        nonobs_y_list = []
        
        for ds in range(len(self.ds_size_list)):
            trainX_1 = trainX[start[ds]:start[ds+1]]
            trainy_1 = trainy[start[ds]:start[ds+1]]
            true_distribution_1 = self.__define_category(trainy_1, ds)
            item_list_1 = self.__create_item_list(trainy_1)
            
            (trainX_1_selected, trainy_1_selected,\
            trainX_1_unselected, trainy_1_unselected) = self.__create_biased_sample(trainX_1, trainy_1, item_list_1, true_distribution_1)
            
            
            obs_X_list.append(trainX_1_selected)
            obs_y_list.append(trainy_1_selected)
            nonobs_X_list.append(trainX_1_unselected)
            nonobs_y_list.append(trainy_1_unselected)

        self.trainX_selected = obs_X_list
        self.trainy_selected = obs_y_list
        self.trainX_unselected = nonobs_X_list
        self.trainy_unselected = nonobs_y_list
        
        return (self.trainX_selected, self.trainy_selected )









    

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