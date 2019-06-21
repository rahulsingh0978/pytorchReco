#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 14:12:06 2019

@author: rahul.singh
"""

from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import mrr_score
from spotlight.factorization.implicit import ImplicitFactorizationModel

dataset = get_movielens_dataset(variant='100K')

train, test = random_train_test_split(dataset)


model = ImplicitFactorizationModel(n_iter=3,
                                   loss='bpr')
model.fit(train)
print("trained")
mrr = mrr_score(model, test)

print(mrr)