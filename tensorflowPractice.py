#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 08:57:24 2019

@author: rahul.singh
"""

import tensorflow as tf
a = tf.Variable(1 ,name='a')
b = tf.Variable(2 ,name='b')
f = a+ b
init = tf.global_variables_initializer()
with tf.Session() as s:
    init.run()
    print(f.eval())
