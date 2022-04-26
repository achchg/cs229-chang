###########################
# Row-stacking inference #
##########################
from utils.preprocessing import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from cycler import cycler
import time

def rowStackingModel_inference(obs_X_list, obs_y_list, nonobs_X_list, nonobs_y_list, category_wt, item):
    """row-stacking true label + logistic regression inference

    Args:
        obs_X_list (list): Observed data (features)
        obs_y_list (list): Observed data (labels)
        nonobs_X_list (list): Non-observed data (features)
        nonobs_y_list (list): Non-observed data (labels)
        category_wt (dict(list)): Biased distribution of the data
        item (dict(list)): labels and categories of dataset

    Returns:
        tuple: 
        (true_target_obs_bias: True observed bias of target category
         true_target_nonobs_bias: True non-observed bias of target category 
         r_true_obs_biased: Relative True observed bias of target category
         r_true_nonobs_biased: Relative True non-observed bias of target category 
         obs_accuracy: Observed accuracy  
         nonobs_accuracy: Non-bserved accuracy 
         len(trainX_selected), 
         len(trainX_unselected))
    """
    start_time = time.time()
    
    print('====================== prepare data ======================')
    trainX_selected = np.vstack(obs_X_list)
    trainy_selected = np.concatenate(obs_y_list, axis = 0)
    trainX_unselected = np.vstack(nonobs_X_list)
    trainy_unselected = np.concatenate(nonobs_y_list, axis = 0)
    
    print(f'took {time.time() - start_time:.2f} sec')
    
    obs_data = define_category(trainy_selected, category_wt, item).drop('selection_wt', axis = 1)
    obs_data['true_pct'] = 0.1
    obs_data = obs_data.assign(true_obs_biased = lambda x: round((x.item_pct - x.true_pct)*100, 2),
                               r_true_obs_biased = lambda x: round((x.item_pct - x.true_pct)/x.true_pct*100, 2))\
                       .set_index('item')
    
    nonobs_data = define_category(trainy_unselected, category_wt, item)\
                            .rename(columns={'item_pct':'nonobs_item_pct'})\
                            [['item', 'nonobs_item_pct']].set_index('item')
    
    all_data = obs_data.join(nonobs_data, on = 'item', how = 'left')\
                       .assign(true_nonobs_biased = lambda x: round((x.nonobs_item_pct - x.true_pct)*100,2),
                               r_true_nonobs_biased = lambda x: round((x.nonobs_item_pct - x.true_pct)/x.true_pct*100, 2))
    
    true_target_obs_bias = all_data[all_data['type'] == 'T-shirt']['true_obs_biased'][0]
    true_target_nonobs_bias = all_data[all_data['type'] == 'T-shirt']['true_nonobs_biased'][0]
    r_true_obs_biased = all_data[all_data['type'] == 'T-shirt']['r_true_obs_biased'][0]
    r_true_nonobs_biased = all_data[all_data['type'] == 'T-shirt']['r_true_nonobs_biased'][0]
    
    print('====================== row-stacking inference ======================')
    trainX_selected2D = trainX_selected.reshape(trainX_selected.shape[0],-1)
    trainX_unselected2D = trainX_unselected.reshape(trainX_unselected.shape[0],-1)
    
    pipe = make_pipeline(MaxAbsScaler(), 
                         LogisticRegression(random_state=0, 
                                            multi_class='multinomial', 
                                            solver='lbfgs',
                                            tol=0.1))

    pipe.fit(trainX_selected2D, trainy_selected)
    
    trainy_selected_predict = pipe.predict(trainX_selected2D)
    trainy_unselected_predict = pipe.predict(trainX_unselected2D)
    
    obs_accuracy = round(pipe.score(trainX_selected2D, trainy_selected)*100, 2)
    nonobs_accuracy = round(pipe.score(trainX_unselected2D, trainy_unselected)*100, 2)
    
    print(f'took {time.time() - start_time:.2f} sec')

    
    return (true_target_obs_bias, 
            true_target_nonobs_bias, 
            r_true_obs_biased, 
            r_true_nonobs_biased, 
            obs_accuracy, 
            nonobs_accuracy,
            len(trainX_selected), len(trainX_unselected))


def plot1(ds1_size, ds3_size, std_list, mnist_train, item, batch_size):
    """plot row-stacking true label + logistic regression inference

    Args:
        ds1_size (float): Distribution of dataset 1 in data mixture
        ds3_size (float): Distribution of dataset 3 in data mixture
        std_list (list): Standard deviations of dataset 1, 2, and 3
        mnist_train (list): Training data loaded from Mnist
        item (dict(list)): Labels and categories of dataset
        batch_size (int): Batch size
    """
    ''''''
    true_target_obs_bias_list = []
    true_target_nonobs_bias_list = [] 
    r_true_target_obs_bias_list = []
    r_true_target_nonobs_bias_list = []
    true_target_test_bias_list = []
    r_true_target_test_bias_list = []
    n_obs_list = []
    n_nonobs_list = []
    
    # biased toward target
    category_wt1 = {
        'Top': [0.5],
        'Bottom': [0.5],
        'Shoe': [0.9],
        'Bag': [0.9]
    }

    # unbiased
    category_wt2 = {
        'Top': [0.9],
        'Bottom': [0.9],
        'Shoe': [0.9],
        'Bag': [0.9]
    }

    # biased toward non-target
    category_wt3 = {
        'Top': [0.9],
        'Bottom': [0.9],
        'Shoe': [0.3],
        'Bag': [0.3]
    }

    category_wt = [category_wt1, category_wt2, category_wt3]

    for size in range(1, 10):
        p = size/10
        test_size = [1-ds1_size, 1-p, 1-ds3_size]
        print(test_size)
        
        (obs_X_list, obs_y_list, 
         nonobs_X_list, nonobs_y_list, 
         rest_X_list, rest_y_list) = generate_data_mixture_base(0, mnist_train, 
                                                               test_size, 
                                                               std_list, 
                                                               category_wt, 
                                                               item, batch_size)
        
        (true_target_obs_bias, 
         true_target_nonobs_bias, 
         true_target_test_bias, 
         r_true_obs_biased, 
         r_true_nonobs_biased, 
         r_true_test_biased,
         n_obs, n_nonobs) = rowStackingModel_inference(obs_X_list, 
                                                       obs_y_list,
                                                       nonobs_X_list, 
                                                       nonobs_y_list,
                                                       category_wt[0],
                                                       item)

        true_target_obs_bias_list.append(true_target_obs_bias)
        true_target_nonobs_bias_list.append(true_target_nonobs_bias)
        r_true_target_obs_bias_list.append(r_true_obs_biased)
        r_true_target_nonobs_bias_list.append(r_true_nonobs_biased)
        true_target_test_bias_list.append(true_target_test_bias)
        r_true_target_test_bias_list.append(r_true_test_biased)
        n_obs_list.append(n_obs)
        n_nonobs_list.append(n_nonobs)

    plot1 = pd.DataFrame(np.vstack((np.array([size*10 for size in range(1, 10)]), 
                            np.array(true_target_obs_bias_list), 
                            np.array(true_target_nonobs_bias_list),
                            np.array(r_true_target_obs_bias_list), 
                            np.array(r_true_target_nonobs_bias_list),
                            np.array(true_target_test_bias_list),
                            np.array(r_true_target_test_bias_list),
                            np.array(n_obs_list),
                            np.array(n_nonobs_list)
                           )).T
                )
    plot1.columns = ['+ % of Dataset 2', 
                     'True obs bias', 
                     'True non-obs bias', 
                     'True obs relative bias', 
                     'True non-obs relative bias', 
                     'True test bias', 
                     'True test relative bias', 
                     'Num obs samples', 
                     'Num nonobs samples']
    
    plot1.to_csv(f'model/row_stacking/results/Data_bias_baseline_acc_{round((1-ds1_size)*100)}_{round((1-ds3_size)*100)}.csv')
    
    
def just_plot1(plot1):
    """data saved, row-stacking true label + logistic regression plotted

    Args:
        plot1 (pd.DataFrame): inference data prepared for plotting
    """
    ''''''
    fig,ax = plt.subplots(figsize=(10,3.5))
    custom_cycler = (cycler(color=['#0c4bbc', '#ff8800', '#61a712']) +
                     cycler(lw=[2, 2, 2]) +
                     cycler(linestyle=['-', '-', '-']))

    custom_cycler1 = (cycler(color=['#0c4bbc', '#ff8800', '#61a712']) +
                      cycler(lw=[2, 2, 2]) +
                     cycler(linestyle=['--', '--', '--']))
    # make a plot
    ax.set_prop_cycle(custom_cycler)
    ax.plot(plot1.drop(['Num obs samples', 'Num nonobs samples',
                        'True obs relative bias', 
                         'True non-obs relative bias', 
                         'True test relative bias'], 
                       axis = 1)\
                  .set_index('+ % of Dataset 2'))

    ax.legend(['True obs bias', 
               'True non-obs bias',  
               'True test bias'],
              bbox_to_anchor=(1.2, 0), loc='lower left', ncol=1)

    ax.set_ylabel("Absolute data biasness",fontsize=18)
    ax.set_xlabel('+ % of Dataset 2',fontsize=18)
    ax.xaxis.set_tick_params(labelsize='large')
    ax.yaxis.set_tick_params(labelsize='large')

    ax2=ax.twinx()
    ax2.set_prop_cycle(custom_cycler1)
    # make a plot with different y-axis using second axis object
    ax2.plot(plot1.drop(['Num obs samples', 'Num nonobs samples', 'True obs bias', 
                         'True obs bias', 
                         'True non-obs bias',  
                         'True test bias'], axis = 1)\
                   .set_index('+ % of Dataset 2'), marker="o")

    ax2.legend(['True obs relative bias', 
                'True non-obs relative bias', 
                'True test relative bias'],
              bbox_to_anchor=(1.2, 1), loc='upper left', ncol=1)

    ax2.set_prop_cycle(custom_cycler)
    ax2.set_ylabel("Relative data biasness",fontsize=18)
    ax2.xaxis.set_tick_params(labelsize='large')
    ax2.yaxis.set_tick_params(labelsize='large')

    ax3=ax.twinx()
    ax3.axis('off')
    x = [10*i for i in range(0, 11)]
    y = [10*i for i in range(0, 11)]

    ax3.grid()
    ax3.margins(0) # remove default margins (matplotlib verision 2+)

    for i in range(1, len(x)):
        total = int((ds1_size+ds3_size)*100+x[i])
        mixture_size = [int((ds1_size)*10000/total), int(x[i]*100/total), int((ds3_size)*10000/total)]
        ax3.axhspan(0, mixture_size[0], x[i-1]/100, x[i]/100, facecolor='white', alpha=0.1)
        ax3.axhspan(mixture_size[0], mixture_size[0]+mixture_size[1], x[i-1]/100, x[i]/100, facecolor='#61a712', alpha=0.1)
        ax3.axhspan(mixture_size[0]+mixture_size[1], mixture_size[0]+mixture_size[1]+mixture_size[2], x[i-1]/100, x[i]/100, 
                    facecolor='white', alpha=0.1)
    ax3.legend(['DS-1 + DS-3 size in mixture', 
                'DS-2 size in mixture'],
               bbox_to_anchor=(0.8, 0.5), loc='right', ncol=1)

    plt.title(f'{round((ds1_size)*100)}% of Dataset 1 + {round((ds3_size)*100)}% of Dataset 3 in mixture', fontsize=16)
    plt.savefig(f'model/row_stacking/results/Data_bias_baseline_acc_{round((ds1_size)*100)}_{round((ds3_size)*100)}')
    
    
####################################
# Weighted-row-stacking inference #
###################################
    
def additiveModel_transformation(trainX_2D_selected, obs_y_list):
    n_sample = trainX_2D_selected.shape[0]
    d = trainX_2D_selected.shape[1] - 1
    
    trainX_selected2D_grp = trainX_2D_selected[:,-1].reshape(n_sample,1)
    trainX_selected2D_grp_orig = trainX_2D_selected[:,:d]
    scale = MaxAbsScaler().fit(trainX_selected2D_grp_orig)
    trainX_selected2D_grp_orig = scale.transform(trainX_selected2D_grp_orig)
    
    one_hot = OneHotEncoder(drop='first').fit(trainX_selected2D_grp)
    trainX_one_hot = one_hot.transform(trainX_selected2D_grp).toarray()
    
    trainX_2D_selected = np.hstack((trainX_selected2D_grp_orig, trainX_one_hot))
    trainy_selected = np.concatenate(obs_y_list, axis = 0)
    
    return (trainX_2D_selected, trainy_selected)



def mixedModel_transformation(obs_X2D_list, obs_y_list):
    trainX_selected = np.vstack(obs_X2D_list)
    n_sample = trainX_selected.shape[0]
    d = trainX_selected.shape[1] - 1
    
    trainX_grp = trainX_selected[:,-1].reshape(n_sample,1)
    trainX_selected_orig = trainX_selected[:,:d]
    scale = MaxAbsScaler().fit(trainX_selected_orig)
    trainX_selected_orig = scale.transform(trainX_selected_orig)
    
    trainX_selected = np.hstack((trainX_selected_orig, trainX_grp))
    
    pd_trainX_selected2D = pd.DataFrame(trainX_selected).add_prefix('x_')
    pd_trainX_selected2D = pd_trainX_selected2D.rename(columns={f'x_{d}':'group'})
    
    trainy_selected = np.concatenate(obs_y_list, axis = 0)
    pd_trainy_selected = pd.DataFrame(trainy_selected)
    pd_trainy_selected.columns = ['y']
    
    pd_train = pd.concat((pd_trainX_selected2D, pd_trainy_selected), axis = 1)
    pd_train['y'] = np.where(pd_train['y'] == 1, 1, 0)
    
    return (pd_train)



def additiveModel_inference(obs_X_list, obs_y_list, nonobs_X_list, nonobs_y_list, obs_X2D_list, nonobs_X2D_list,
                            rest_X_list, rest_y_list, rest_X2D_list):
    import time
    start_time = time.time()
    
    print('====================== prepare data ======================')
    trainX_selected = np.vstack(obs_X_list)
    trainX_2D_selected = np.vstack(obs_X2D_list)
    trainy_selected = np.concatenate(obs_y_list, axis = 0)
    
    trainX_unselected = np.vstack(nonobs_X_list)
    trainX_2D_unselected = np.vstack(nonobs_X2D_list)
    trainy_unselected = np.concatenate(nonobs_y_list, axis = 0)
    
    trainX_rest = np.vstack(rest_X_list)
    trainX_2D_rest = np.vstack(rest_X2D_list)
    trainy_rest = np.concatenate(rest_y_list, axis = 0)
    
    print(f'took {time.time() - start_time:.2f} sec')
    
    
    print('====================== row-stacking + base inference ======================')
    trainX_selected2D = trainX_selected.reshape(trainX_selected.shape[0],-1)
    trainX_unselected2D = trainX_unselected.reshape(trainX_unselected.shape[0],-1)
    trainX_rest2D = trainX_rest.reshape(trainX_rest.shape[0],-1)
    
    pipe = make_pipeline(MaxAbsScaler(), 
                         LogisticRegression(random_state=0, 
                                            multi_class='multinomial', 
                                            solver='lbfgs',
                                            tol=0.1))

    pipe.fit(trainX_selected2D, trainy_selected)
    
    trainy_selected_predict = pipe.predict(trainX_selected2D)
    trainy_unselected_predict = pipe.predict(trainX_unselected2D)
    trainy_rest_predict = pipe.predict(trainX_rest2D)
    
    obs_bias_base = (trainy_selected_predict == 1).mean()
    nonobs_bias_base = (trainy_unselected_predict == 1).mean()
    test_bias_base = (trainy_rest_predict == 1).mean()
    
    obs_accuracy_base = round(pipe.score(trainX_selected2D, trainy_selected)*100, 2)
    nonobs_accuracy_base = round(pipe.score(trainX_unselected2D, trainy_unselected)*100, 2)
    test_accuracy_base = round(pipe.score(trainX_rest2D, trainy_rest)*100, 2)
    
    print(f'took {time.time() - start_time:.2f} sec')
    
    print('====================== row-stacking + additiveModel inference ======================')
    trainX_2D_selected, trainy_selected = additiveModel_transformation(trainX_2D_selected, obs_y_list)
    trainX_2D_unselected, trainy_unselected = additiveModel_transformation(trainX_2D_unselected, nonobs_y_list)
    trainX_2D_rest, trainy_rest = additiveModel_transformation(trainX_2D_rest, rest_y_list)
    
    
    pipe_add = make_pipeline(LogisticRegression(random_state=0, 
                                            multi_class='multinomial', 
                                            solver='lbfgs',
                                            tol=0.1))

    pipe_add.fit(trainX_2D_selected, trainy_selected)
    
    trainy_selected_predict_add = pipe_add.predict(trainX_2D_selected)
    trainy_unselected_predict_add = pipe_add.predict(trainX_2D_unselected)
    trainy_rest_predict_add = pipe_add.predict(trainX_2D_rest)
    
    obs_bias_add = (trainy_selected_predict_add == 1).mean()
    nonobs_bias_add = (trainy_unselected_predict_add == 1).mean()
    test_bias_add = (trainy_rest_predict_add == 1).mean()
    
    obs_accuracy_add = round(pipe_add.score(trainX_2D_selected, trainy_selected)*100, 2)
    nonobs_accuracy_add = round(pipe_add.score(trainX_2D_unselected, trainy_unselected)*100, 2)
    test_accuracy_add = round(pipe_add.score(trainX_2D_rest, trainy_rest)*100, 2)
    
    print(f'took {time.time() - start_time:.2f} sec')
    
    return (obs_accuracy_base, nonobs_accuracy_base, test_accuracy_base,
            obs_accuracy_add, nonobs_accuracy_add, test_accuracy_add,
            obs_bias_base, nonobs_bias_base, test_bias_base,
            obs_bias_add, nonobs_bias_add, test_bias_add,
            len(trainX_selected), len(trainX_unselected))
    
#     print('====================== row-stacking + mixedModel inference ======================')
    
#     pd_train = mixedModel_transformation(obs_X2D_list, obs_y_list)
#     pd_unselected = mixedModel_transformation(nonobs_X2D_list, nonobs_y_list)
#     pd_rest = mixedModel_transformation(rest_X2D_list, rest_y_list)

#     random = {"a": '0 + C(group)'}

#     model = mixedglm.from_formula(
#         f"y ~ {'+'.join(pd_train.columns[:-2])}", 
#         random, pd_train
#     )

#     result = model.fit_vb()
#     obs_y_pred = (result.predict(pd_train.drop(['y'], axis = 1)) > 0.5)*1
#     nonobs_y_pred = (result.predict(pd_unselected.drop(['y'], axis = 1)) > 0.5)*1
#     test_y_pred = (result.predict(pd_rest.drop(['y'], axis = 1)) > 0.5)*1
    
#     from sklearn.metrics import accuracy_score
#     obs_accuracy_mix = accuracy_score(pd_train['y'], obs_y_pred)
#     nonobs_accuracy_mix = accuracy_score(pd_unselected['y'], nonobs_y_pred)
#     test_accuracy_mix = accuracy_score(pd_rest['y'], test_y_pred)
    
#     return (obs_accuracy_base, nonobs_accuracy_base, test_accuracy_base,
#             obs_accuracy_add, nonobs_accuracy_add, test_accuracy_add,
#             obs_accuracy_mix, nonobs_accuracy_mix, test_accuracy_mix,
#             len(trainX_selected), len(trainX_unselected))


def plot2(ds1_size, ds3_size, std_list, mnist_train, item, batch_size):
    '''plot weighted-row-stacking true label + additive logistic regression inference'''
    obs_accuracy_base_list = []
    nonobs_accuracy_base_list = []
    obs_accuracy_add_list = []
    nonobs_accuracy_add_list = []
    n_obs_list = []
    n_nonobs_list = []
    
    test_accuracy_base_list = []
    test_accuracy_add_list = []
    
    obs_bias_base_list = []
    nonobs_bias_base_list = []
    test_bias_base_list = []
    obs_bias_add_list = []
    nonobs_bias_add_list = []
    test_bias_add_list = []

    # biased toward target
    category_wt1 = {
        'Top': [0.5],
        'Bottom': [0.5],
        'Shoe': [0.9],
        'Bag': [0.9]
    }

    # unbiased
    category_wt2 = {
        'Top': [0.9],
        'Bottom': [0.9],
        'Shoe': [0.9],
        'Bag': [0.9]
    }

    # biased toward non-target
    category_wt3 = {
        'Top': [0.9],
        'Bottom': [0.9],
        'Shoe': [0.3],
        'Bag': [0.3]
    }

    category_wt = [category_wt1, category_wt2, category_wt3]

    for size in range(1, 10):
        p = size/10
        test_size = [1-ds1_size, 1-p, 1-ds3_size]
        print(test_size)
        
        print('=======train=======')
        
        (obs_X_list, obs_y_list, 
         nonobs_X_list, nonobs_y_list, 
         obs_X2D_list, nonobs_X2D_list,
         rest_X_list, rest_y_list, rest_X2D_list) = generate_data_mixture_2D(0, mnist_train, 
                                                                             test_size, 
                                                                             std_list, 
                                                                             category_wt, 
                                                                             item, batch_size)
        print('=======train score=======')
        (obs_accuracy_base, nonobs_accuracy_base, test_accuracy_base,
         obs_accuracy_add, nonobs_accuracy_add, test_accuracy_add,
         obs_bias_base, nonobs_bias_base, test_bias_base,
         obs_bias_add, nonobs_bias_add, test_bias_add,
         n_obs, n_nonobs) = additiveModel_inference(obs_X_list, obs_y_list, 
                                                   nonobs_X_list, nonobs_y_list, 
                                                   obs_X2D_list, nonobs_X2D_list,
                                                   rest_X_list, rest_y_list, rest_X2D_list)


        obs_accuracy_base_list.append(obs_accuracy_base)
        nonobs_accuracy_base_list.append(nonobs_accuracy_base)
        test_accuracy_base_list.append(test_accuracy_base)
        obs_accuracy_add_list.append(obs_accuracy_add)
        nonobs_accuracy_add_list.append(nonobs_accuracy_add)
        test_accuracy_add_list.append(test_accuracy_add)
        
        obs_bias_base_list.append(obs_bias_base)
        nonobs_bias_base_list.append(nonobs_bias_base)
        test_bias_base_list.append(test_bias_base)
        obs_bias_add_list.append(obs_bias_add)
        nonobs_bias_add_list.append(nonobs_bias_add)
        test_bias_add_list.append(test_bias_add)
        
        n_obs_list.append(n_obs)
        n_nonobs_list.append(n_nonobs)
        
    plot2 = pd.DataFrame(np.vstack((np.array([size*10 for size in range(1, 10)]), 
                            np.array(obs_accuracy_base_list), 
                            np.array(nonobs_accuracy_base_list),
                            np.array(test_accuracy_base_list),
                            np.array(obs_accuracy_add_list), 
                            np.array(nonobs_accuracy_add_list),
                            np.array(test_accuracy_add_list),
                            np.array(obs_bias_base_list), 
                            np.array(nonobs_bias_base_list),
                            np.array(test_bias_base_list),
                            np.array(obs_bias_add_list), 
                            np.array(nonobs_bias_add_list),
                            np.array(test_bias_add_list),
                            np.array(n_obs_list),
                            np.array(n_nonobs_list)
                           )).T
                )
    plot2.columns = ['+ % of Dataset 2', 
                     'Row-stacking obs accuracy', 
                     'Row-stacking nonobs accuracy', 
                     'Row-stacking accuracy in TEST set', 
                     'Weighted-Row-stacking obs accuracy', 
                     'Weighted-Row-stacking nonobs accuracy', 
                     'Weighted-Row-stacking accuracy in TEST set', 
                     'Row-stacking obs bias', 
                     'Row-stacking nonobs bias', 
                     'Row-stacking bias in TEST set', 
                     'Weighted-Row-stacking obs bias', 
                     'Weighted-Row-stacking nonobs bias', 
                     'Weighted-Row-stacking bias in TEST set', 
                     'Num obs samples', 
                     'Num nonobs samples']
    
    plot2.to_csv(f'model/row_stacking/results/Model_bias_baseline_acc_{round((1-ds1_size)*100)}_{round((1-ds3_size)*100)}.csv')

    
def just_plot2(plot2):
    '''data saved, inference plotting'''
    from cycler import cycler
    fig,ax = plt.subplots(figsize=(10,3.5))
    custom_cycler = (cycler(color=['#0c4bbc', '#ff8800', '#61a712', 
                                   '#0c4bbc', '#ff8800', '#61a712', 
                                   '#0c4bbc', '#ff8800', '#61a712', 
                                   '#0c4bbc', '#ff8800', '#61a712']) +
                     cycler(lw=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]) +
                     cycler(linestyle=['-', '-', '-', 
                                       '--', '--', '--', 
                                       '-.', '-.', '-.', 
                                       'dotted', 'dotted', 'dotted']))

    custom_cycler1 = (cycler(color=['#0c4bbc', '#ff8800', '#61a712', 
                                   '#0c4bbc', '#ff8800', '#61a712']) +
                      cycler(lw=[2, 2, 2,2, 2, 2]) +
                     cycler(linestyle=['-.', '-.', '-.', 
                                       'dotted', 'dotted', 'dotted']))
    # make a plot
    ax.set_prop_cycle(custom_cycler)
    ax.plot(plot2.drop(['Num obs samples', 'Num nonobs samples', 
                        'Row-stacking obs bias', 
                        'Row-stacking nonobs bias', 
                        'Row-stacking bias in TEST set', 
                        'Weighted-Row-stacking obs bias', 
                        'Weighted-Row-stacking nonobs bias', 
                        'Weighted-Row-stacking bias in TEST set'], axis = 1)\
                  .set_index('+ % of Dataset 2'))

    ax.legend(['Row-stacking obs accuracy', 
               'Row-stacking nonobs accuracy', 
               'Row-stacking accuracy in TEST set', 
               'Weighted-Row-stacking obs accuracy', 
               'Weighted-Row-stacking nonobs accuracy', 
               'Weighted-Row-stacking accuracy in TEST set'],
              bbox_to_anchor=(1.1, 0), loc='lower left', ncol=1)

    ax.set_ylabel("Model accuracy",fontsize=18)
    ax.set_xlabel('+ % of Dataset 2',fontsize=18)
    ax.xaxis.set_tick_params(labelsize='large')
    ax.yaxis.set_tick_params(labelsize='large')
    
    ax2=ax.twinx()
    ax2.set_prop_cycle(custom_cycler1)
    # make a plot with different y-axis using second axis object
    ax2.plot(plot2.drop(['Num obs samples', 'Num nonobs samples', 
                         'Row-stacking obs accuracy', 
                         'Row-stacking nonobs accuracy', 
                         'Row-stacking accuracy in TEST set', 
                         'Weighted-Row-stacking obs accuracy', 
                         'Weighted-Row-stacking nonobs accuracy', 
                         'Weighted-Row-stacking accuracy in TEST set'], axis = 1)\
                   .set_index('+ % of Dataset 2'), marker="o")

    ax2.legend(['Row-stacking obs bias', 
                'Row-stacking nonobs bias', 
                'Row-stacking bias in TEST set', 
                'Weighted-Row-stacking obs bias', 
                'Weighted-Row-stacking nonobs bias', 
                'Weighted-Row-stacking bias in TEST set'],
              bbox_to_anchor=(1.2, 1), loc='upper left', ncol=1)

    ax2.set_prop_cycle(custom_cycler)
    ax2.set_ylabel("Absolute data biasness",fontsize=18)
    ax2.xaxis.set_tick_params(labelsize='large')
    ax2.yaxis.set_tick_params(labelsize='large')


    ax3=ax.twinx()
    ax3.axis('off')
    x = [10*i for i in range(0, 11)]
    y = [10*i for i in range(0, 11)]

    ax3.grid()
    ax3.margins(0) # remove default margins (matplotlib verision 2+)

    for i in range(1, len(x)):
        total = int((ds1_size+ds3_size)*100+x[i])
        mixture_size = [int((ds1_size)*10000/total), int(x[i]*100/total), int((ds3_size)*10000/total)]
        ax3.axhspan(0, mixture_size[0], x[i-1]/100, x[i]/100, facecolor='white', alpha=0.1)
        ax3.axhspan(mixture_size[0], mixture_size[0]+mixture_size[1], x[i-1]/100, x[i]/100, facecolor='#61a712', alpha=0.1)
        ax3.axhspan(mixture_size[0]+mixture_size[1], mixture_size[0]+mixture_size[1]+mixture_size[2], x[i-1]/100, x[i]/100, 
                    facecolor='white', alpha=0.1)
    ax3.legend(['DS-1 + DS-3 size in mixture', 
                'DS-2 size in mixture'],
          bbox_to_anchor=(1.1, 1), loc='upper left', ncol=1)

    plt.title(f'{round((ds1_size)*100)}% of Dataset 1 + {round((ds3_size)*100)}% of Dataset 3 in mixture', fontsize=16)
    plt.savefig(f'Model_bias_baseline_acc_{round((ds1_size)*100)}_{round((ds3_size)*100)}')