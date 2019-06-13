import argparse
import pickle

import torch

from torchmf import (BaseModule, BPRModule, BasePipeline,
                     bpr_loss, PairwiseInteractions)
import utils


def explicit():
    print ("inside explicit")
    train, test = utils.get_movielens_train_test_split()
    pipeline = BasePipeline(train, test=test, model=BaseModule,
                            n_factors=10, batch_size=1024, dropout_p=0.02,
                            lr=0.02, weight_decay=0.1,
                            optimizer=torch.optim.Adam, n_epochs=4,
                            verbose=True, random_seed=2017)
    pipeline.fit()
    return pipeline


def implicit():
    train, test = utils.get_movielens_train_test_split(implicit=True)
    print (train.shape , test.shape)
    pipeline = BasePipeline(train, test=test, verbose=True,
                           batch_size=1024, num_workers=4,
                           n_factors=20, weight_decay=0,
                           dropout_p=0., lr=.2, sparse=True,
                           optimizer=torch.optim.SGD, n_epochs=4,
                           random_seed=2017, loss_function=bpr_loss,
                           model=BPRModule,
                           interaction_class=PairwiseInteractions,
                           eval_metrics=('auc', 'patk'))
    print ("before fit")
    pipeline.fit()
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
    pipeline.fit()


if __name__ == '__main__':  
    train, test = utils.get_movielens_train_test_split(implicit=True)
    print ((test))
    #print (test).shape[0]
    parser = argparse.ArgumentParser(description='torchmf')
    parser.add_argument('--example',
                        help='explicit, implicit, or hogwild')
    args = parser.parse_args()
    #args ='implicit'
    #implicit()
    if args.example == 'explicit':
        model = explicit()
    elif args.example == 'implicit':
        model = implicit()
    elif args.example == 'hogwild':
        model = hogwild()
    else:
        print('example must be explicit, implicit, or hogwild')
        
        
    user ,item =utils.getUserItem()
    torch.save({
                "model": model.model.state_dict(),
                "optimizer": model.optimizer.state_dict(),
            },'/Users/rahul.singh/Code/modelSave1.pth')
    
    #model.model.eval();
    #model.model.predict([2],[2])
    
    #model.model.state_dict
    #print(test.describe())
    #print(model.model.predict())
    