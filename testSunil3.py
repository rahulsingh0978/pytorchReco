import os

import numpy as np
import pytest
import torch
import scipy.stats as st

from spotlight.cross_validation import random_train_test_split
from spotlight.datasets import movielens
from spotlight.evaluation import mrr_score
from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.factorization.representations import BilinearNet
from spotlight.layers import BloomEmbedding


from spotlight.torch_utils import cpu, gpu

if __name__ == '__main__':
    interactions = movielens.get_movielens_dataset('100K')
    RANDOM_STATE = np.random.RandomState(42)
    if not os.path.exists("/Users/rahul.singh/Code/pytorchRec/save"):
        os.makedirs("/Users/rahul.singh/Code/pytorchRec/save")
    train, test = random_train_test_split(interactions,
                                          random_state=RANDOM_STATE)
    implicitFactorizationModel1 = ImplicitFactorizationModel(loss='pointwise',
                                       n_iter=10,
                                       batch_size=1024,
                                       learning_rate=1e-2,
                                       l2=1e-6,
                                       use_cuda=False)
    retrain = False
    if (retrain):
        print("traning started")

        implicitFactorizationModel1.fit(test)
        print("traning done")

        checkpoint_file = os.path.join("/Users/rahul.singh/Code/pytorchRec/save", "checkpoint.pth")
        torch.save({
            #"fir":implicitFactorizationModel._net,
            "model": implicitFactorizationModel1._net.state_dict(),
            "optimizer": implicitFactorizationModel1._net.state_dict(),
        }, checkpoint_file)
    
    bestmodel_file = os.path.join("/Users/rahul.singh/Code/pytorchRec/save", "checkpoint.pth")
    load_res = torch.load(bestmodel_file, map_location="cpu")
    
    implicitFactorizationModel1._initialize(train)
    implicitFactorizationModel1._net.load_state_dict(load_res["model"])
    implicitFactorizationModel1._net.load_state_dict(load_res["optimizer"])
    implicitFactorizationModel1._net.eval()
    print("testing started")
    # model._net.
    mrr = mrr_score(implicitFactorizationModel1, test, train=train).mean()

    print("error:", mrr)
    result = implicitFactorizationModel1.predict(np.array([1]))
    print(st.rankdata(result))
    
    implicitFactorizationModel1.fit(test)
    implicitFactorizationModel1._net.eval()
    print("testing started")
    # model._net.
    mrr = mrr_score(implicitFactorizationModel1, train, train=test).mean()

    print("error:", mrr)
    result = implicitFactorizationModel1.predict(np.array([907]))
    print(st.rankdata(result))
