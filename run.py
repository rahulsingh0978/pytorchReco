import argparse
import pickle

import torch, os
import numpy as np

from torchmf import (BaseModule, BPRModule, BasePipeline,
                     bpr_loss, PairwiseInteractions)
import utils
import config as myconfig


def explicit():
    print("inside explicit")
    train, test = utils.get_movielens_train_test_split()
    pipeline = BasePipeline(train, test=test, model=BaseModule,
                            n_factors=10, batch_size=1024, dropout_p=0.02,
                            lr=0.02, weight_decay=0.1,
                            optimizer=torch.optim.Adam, n_epochs=4,
                            verbose=True, random_seed=2017)
    return pipeline


def implicit():
    train, test = utils.get_movielens_train_test_split(implicit=True)
    print(train.shape, test.shape)
    pipeline = BasePipeline(train, test=test, verbose=True,
                            batch_size=1024, num_workers=4,
                            n_factors=20, weight_decay=0,
                            dropout_p=0., lr=.2, sparse=True,
                            optimizer=torch.optim.SGD, n_epochs=4,
                            random_seed=2017, loss_function=bpr_loss,
                            model=BPRModule,
                            interaction_class=PairwiseInteractions,
                            eval_metrics=('auc', 'patk'))
    print("before fit")
    return pipeline


def hogwild():
    train, test = utils.get_movielens_train_test_split(implicit=True)

    pipeline = BasePipeline(train, test=test, verbose=True,
                            batch_size=1024, num_workers=4,
                            n_factors=20, weight_decay=0,
                            dropout_p=0., lr=.2, sparse=True,
                            optimizer=torch.optim.SGD, n_epochs=4,
                            random_seed=2017, loss_function=bpr_loss,
                            model=BPRModule, hogwild=True,
                            interaction_class=PairwiseInteractions,
                            eval_metrics=('auc', 'patk'))
    return pipeline


def trainModel(config):
    pipeline = explicit()
    # pipeline = implicit()
    # pipeline = hogwild()
    pipeline.fit()

    checkpoint_file = os.path.join(config.save_dir, "checkpoint.pth")

    user, item = utils.getNumberOfUniqueUsersAndItems()
    torch.save({
        "model": pipeline.model.state_dict(),
        "optimizer": pipeline.optimizer.state_dict(),
    }, checkpoint_file)


def testModel(config):
    bestmodel_file = os.path.join(config.save_dir, "checkpoint.pth")
    load_res = torch.load(bestmodel_file, map_location="cpu")
    pipeline = explicit()
    # pipeline = implicit()
    # pipeline = hogwild()
    pipeline.model.load_state_dict(load_res["model"])
    pipeline.optimizer.load_state_dict(load_res["optimizer"])
    pipeline.model.eval()

    testUsers=[1,2]
    testMovie=[1]

    result = pipeline.model.predict(torch.Tensor(testusers).long(), torch.Tensor(testMovie).long())
    print("predtict", result.data.numpy())


if __name__ == '__main__':

    config, unparsed = myconfig.get_config()
    # ----------------------------------------
    # Parse configuration
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print("unparsed for DQN :", unparsed)
        # input("Press Enter to continue...")

    if config.mode == "train":
        trainModel(config)
    elif config.mode == "test":
        testModel(config)
    else:
        raise ValueError("Unknown run mode \"{}\"".format(config.mode))
