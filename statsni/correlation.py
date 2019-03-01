#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 13:58:29 2018

@author: kristianeschenburg
"""

import numpy as np

def fisher(samples):
    
    """
    Fisher transform samples of correlation values.
    """
    
    return (1./2) * np.log((1.+samples)/(1.-samples))

def fisher_inv(samples):
    
    """
    Inverse Fisher transform Z-transformed correlation values.
    """
    
    return np.tanh(samples)