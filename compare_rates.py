import nodespace
import nhpp_mod
import torch
import os
import smallnet
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def get_times(data_set):
    times = []
    time_col = 2
    for row in data_set:
        cur_t = row[time_col]
        times.append(cur_t)
    return np.array(times)

def compare_intensity_rates(model, nodespace, node_u, node_v, path, training_data, test_data):
    model.load_state_dict(torch.load(path))
    model.eval()

    train_t = get_times(training_data)
    test_t = get_times(test_data)

    plot_t = [train_t, test_t]

    gttrain = []
    restrain = []
    for ti in plot_t[0]:
        gttrain.append(nodespace.lambda_sq_fun(ti, node_u, node_v))
        restrain.append(model.lambda_fun(ti, node_u, node_v))

    gttest = []
    restest = []
    for ti in plot_t[1]:
        gttest.append(nodespace.lambda_sq_fun(ti, node_u, node_v))
        restest.append(model.lambda_fun(ti, node_u, node_v))

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(plot_t[0], restrain, color="red", label="est")
    ax[0].plot(plot_t[0], gttrain, color="blue", label="gt")
    ax[0].legend()
    ax[0].set_title("Train")
    ax[1].plot(plot_t[1], restest, color="red", label="est")
    ax[1].plot(plot_t[1],gttest, color="blue", label="gt")
    ax[1].legend()
    ax[1].set_title("Test")
    plt.show()


if __name__=="__main__":
    fpath = r"state_dicts/training_experiment"


    # Create dynamical system with constant velocity
    ns_gt = nodespace.NodeSpace()
    ns_gt.beta = 7.5

    z_gt = np.array([[-0.6, 0.], [0.6, 0.1], [0.,0.6], [0.,-0.6]])
    v_gt = np.array([[0.09,0.01], [-0.01,-0.01], [0.01,-0.09], [-0.01, 0.09]])
    a_gt = np.array([[0.,0.], [0.,0.], [0.,0.], [0., 0.]])
    n_points=len(z_gt)
    ns_gt.init_conditions(z_gt, v_gt, a_gt)

    # create model instance
    model = smallnet.SmallNet(n_points=n_points, init_beta=5.) # beta gets overwritten from state dict

    # Simulate event time data set for the two nodes
    ZERO_SEED = 0
    np.random.seed(ZERO_SEED)
    torch.manual_seed(ZERO_SEED)
    t = np.linspace(0, 10)
    rmat = nhpp_mod.root_matrix(ns_gt) 
    mmat = nhpp_mod.monotonicity_mat(ns_gt, rmat)
    nhppmat = nhpp_mod.nhpp_mat(ns=ns_gt, time=t, root_matrix=rmat, monotonicity_matrix=mmat)
    
    # create data set and sort by time
    ind = np.triu_indices(n_points, k=1)
    data_set = []
    for u,v in zip(ind[0], ind[1]):
        event_times = nhpp_mod.get_entry(nhppmat, u=u, v=v)
        for e in event_times:
            data_set.append([u, v, e])

    data_set = np.array(data_set)
    time_col = 2
    data_set = data_set[data_set[:,time_col].argsort()]

    # verify time ordering
    prev_t = 0.
    for row in data_set:
        cur_t = row[time_col]
        assert cur_t > prev_t
        prev_t = cur_t


    data_set = torch.from_numpy(data_set)
    num_train_samples = int(len(data_set)*0.8)
    training_data = data_set[0:num_train_samples]
    test_data = data_set[num_train_samples:]


    seed = 1
    fname = f"batch=ntrain_LR=0.025_75epoch_init_{seed}" + ".pth"

    path =  os.path.join(fpath, fname)
    compare_intensity_rates(model, ns_gt, 0, 3, path, training_data, test_data)