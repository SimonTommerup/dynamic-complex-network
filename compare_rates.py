import nodespace
import nhpp_mod
import torch
import os
from sklearn.datasets import make_blobs
import smallnet_sqdist as smallnet
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

    state_dict = model.state_dict()
    print("Beta", state_dict["beta"])
    # train_t = get_times(training_data)
    # test_t = get_times(test_data)

    train_t = np.linspace(0, training_data[-1][2])
    test_t = np.linspace(test_data[0][2], test_data[-1][2])
    full_time = np.linspace(0, test_data[-1][2])

    plot_t = [train_t, test_t]

    gttrain = []
    restrain = []
    print("Plot train")
    for ti in plot_t[0]:
        gttrain.append(nodespace.lambda_sq_fun(ti, node_u, node_v))
        restrain.append(model.lambda_sq_fun(ti, node_u, node_v))
    
    full_gt = []
    for ti in full_time:
        full_gt.append(nodespace.lambda_sq_fun(ti, node_u, node_v))
    
    gttest = []
    restest = []
    print("Plot test")
    for ti in plot_t[1]:
        gttest.append(nodespace.lambda_sq_fun(ti, node_u, node_v))
        restest.append(model.lambda_sq_fun(ti, node_u, node_v))

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(plot_t[0], restrain, color="red", label="est")
    ax[0].plot(plot_t[0], gttrain, color="blue", label="gt")
    ax[0].set_ylabel("Intensity")
    ax[0].set_xlabel("Time")

    #ax[0].set_yscale("log")
    ax[0].legend()
    ax[0].set_title("Train")
    ax[1].plot(plot_t[1], restest, color="red", label="est")
    ax[1].plot(plot_t[1],gttest, color="blue", label="gt")
    #ax[1].set_yscale("log")
    ax[1].set_ylabel("Intensity")
    ax[1].set_xlabel("Time")
    ax[1].legend()
    ax[1].set_title("Test")
    plt.tight_layout()
    plt.show()


def compare_intensity_rates_acc(model, nodespace, node_u, node_v, path, training_data, test_data):
    model.load_state_dict(torch.load(path))
    model.eval()

    state_dict = model.state_dict()
    print("Beta", state_dict["beta"])
    # train_t = get_times(training_data)
    # test_t = get_times(test_data)

    train_t = np.linspace(0, training_data[-1][2])
    test_t = np.linspace(test_data[0][2], test_data[-1][2])
    full_time = np.linspace(0, test_data[-1][2])

    plot_t = [train_t, test_t]

    gttrain = []
    restrain = []
    print("Plot train")
    for ti in plot_t[0]:
        gttrain.append(nodespace.lambda_fun(ti, node_u, node_v))
        restrain.append(model.lambda_fun(ti, node_u, node_v))
    
    full_gt = []
    for ti in full_time:
        full_gt.append(nodespace.lambda_fun(ti, node_u, node_v))
    
    gttest = []
    restest = []
    print("Plot test")
    for ti in plot_t[1]:
        gttest.append(nodespace.lambda_fun(ti, node_u, node_v))
        restest.append(model.lambda_fun(ti, node_u, node_v))

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(plot_t[0], restrain, color="red", label="est")
    ax[0].plot(plot_t[0], gttrain, color="blue", label="gt")
    ax[0].set_ylabel("Intensity")
    ax[0].set_xlabel("Time")

    #ax[0].set_yscale("log")
    ax[0].legend()
    ax[0].set_title("Train")
    ax[1].plot(plot_t[1], restest, color="red", label="est")
    ax[1].plot(plot_t[1],gttest, color="blue", label="gt")
    #ax[1].set_yscale("log")
    ax[1].set_ylabel("Intensity")
    ax[1].set_xlabel("Time")
    ax[1].legend()
    ax[1].set_title("Test")
    plt.tight_layout()
    plt.show()

def compare_intensity_rates_no_path(model, nodespace, node_u, node_v, training_data, test_data):
    #model.load_state_dict(torch.load(path))
    model.eval()

    #train_t = get_times(training_data)
    #test_t = get_times(test_data)

    train_t = np.linspace(0, training_data[-1][2])
    test_t = np.linspace(test_data[0][2], test_data[-1][2])

    plot_t = [train_t, test_t]

    gttrain = []
    restrain = []

    print("Plot train")
    for ti in plot_t[0]:
        gttrain.append(nodespace.lambda_sq_fun(ti, node_u, node_v))
        restrain.append(model.lambda_fun(ti, node_u, node_v))
    
    gttest = []
    restest = []
    print("Plot test")
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
    # ns_gt = nodespace.NodeSpace()
    # ns_gt.beta = 7.5

    # z_gt = np.array([[-0.6, 0.], [0.6, 0.1], [0.,0.6], [0.,-0.6]])
    # v_gt = np.array([[0.09,0.01], [-0.01,-0.01], [0.01,-0.09], [-0.01, 0.09]])
    # a_gt = np.array([[0.,0.], [0.,0.], [0.,0.], [0., 0.]])
    # n_points=len(z_gt)
    # ns_gt.init_conditions(z_gt, v_gt, a_gt)


    ZERO_SEED = 0
    np.random.seed(ZERO_SEED)
    torch.manual_seed(ZERO_SEED)
    np.seterr(all='raise')

    ns_gt = nodespace.NodeSpace()
    ns_gt.beta = 5
    n_points = 50
    
    z_gt, y = make_blobs(n_samples=n_points, centers=3, center_box=(-5,5))
    v_gt, a_gt = ns_gt.rand_init_dynamics(n_points)
    ns_gt.init_conditions(z_gt, v_gt, a_gt)

    # create model instance
    model = smallnet.SmallNet(n_points=n_points, init_beta=5., mc_samples=5) # beta gets overwritten from state dict

    # Simulate event time data set for the two nodes
    # ZERO_SEED = 0
    # np.random.seed(ZERO_SEED)
    # torch.manual_seed(ZERO_SEED)
    # t = np.linspace(0, 10)
    # rmat = nhpp_mod.root_matrix(ns_gt) 
    # mmat = nhpp_mod.monotonicity_mat(ns_gt, rmat)
    # nhppmat = nhpp_mod.nhpp_mat(ns=ns_gt, time=t, root_matrix=rmat, monotonicity_matrix=mmat)
    
    # # create data set and sort by time
    # ind = np.triu_indices(n_points, k=1)
    # data_set = []
    # for u,v in zip(ind[0], ind[1]):
    #     event_times = nhpp_mod.get_entry(nhppmat, u=u, v=v)
    #     for e in event_times:
    #         data_set.append([u, v, e])

    # data_set = np.array(data_set)
    # time_col = 2
    # data_set = data_set[data_set[:,time_col].argsort()]

    # load data
    data_set = np.load("200point-data_set.npy")
    plt.hist(data_set[:,2])
    split_ratio = 0.975
    num_train_samples = int(len(data_set)*split_ratio)
    plt.axvline(x=data_set[num_train_samples][2])
    plt.show()

    

    # verify time ordering
    time_col = 2
    prev_t = 0.
    print("Time ordering")
    for row in data_set:
        cur_t = row[time_col]
        assert cur_t > prev_t
        prev_t = cur_t
    print("Done.")
    split_ratio = 0.9
    data_set = torch.from_numpy(data_set)
    num_train_samples = int(len(data_set)*split_ratio)
    training_data = data_set[0:num_train_samples]
    test_data = data_set[num_train_samples:]



    batch_size=128
    NUM_EPOCHS = 20
    seed = 7
    fname_1 = f"batch=141_LR=0.001_75epoch_init_{seed}" + ".pth"
    fname_2 = f"batch=ntrain_LR=0.025_75epoch_init_{seed}" + ".pth"
    fname_3 = f"batch={batch_size}_epochs={NUM_EPOCHS}_50_nodes_montecarlo_integral_init_{seed}" + ".pth"
    fname_4 = f"batch={batch_size}_epochs={NUM_EPOCHS}_sample_nodes_montecarlo_integral_init_{seed}" + ".pth"
    path =  os.path.join(fpath, fname_3)

    np.random.seed(ZERO_SEED+6)
    rand_u = np.random.randint(0,n_points)
    rand_v = np.random.randint(0,n_points)

    print(f"u={rand_u}, v={rand_v}")

    compare_intensity_rates(model, ns_gt, rand_u, rand_v, path, training_data, test_data)
    plt.close()

