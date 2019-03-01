#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 17:56:19 2018

@author: kristianeschenburg
"""

import numpy as np

def mutual_information(counts):
    
    """
    Compute the mutual information of a matrix.
    
    Parameters:
    - - - - -
        counts: numpy array of size N x M, where cell (n,m) contains the
                counts between cell n and m.
    Returns:
    - - - -
        mi: mutual information
    """
    
    assert counts[counts<0].sum() == 0
    
    # compute joint probability p(n,m)
    pXY = counts / np.sum(1.*counts)
    
    # compute marginal of n, p(n)
    pX = np.sum(pXY,axis=1)
    # compute marginal of m, p(m)
    pY = np.sum(pXY,axis=0)
    
    # normalize joint by marginal
    p = pXY/np.outer(pX,pY)
    
    # compute cell-by-cell mi, and sum
    # note: log(0) = 0 w.r.t information
    mi = pXY * np.log(p)
    mi = np.nansum(mi)
    
    return mi

def kullback_leibler(P,Q):
    
    """
    Compute the Kullback-Leibler divergence of two distributions.
    
    KL-Divergence is the amount of information lost by endcoding one distribution,
    using another distribution.  The formula is:
                    KL(P|Q) = \Sum_{i} p_{i} *log(p_{i}/q_{i})
        
    which is the expectation of log(p_{i} / q_{i}) with respect to P.
    
    Assumes Base-2 logarithm.
    
    Parameters:
    - - - - -
        P: ndarray
            "true" distribution of events
        Q: ndarray
            "model" distribution of events
    """
    
    assert P.shape == Q.shape, 'Distribution shapes do not match.'
    assert np.nansum(P) <= 1, 'P is not normalized.'
    assert np.nansum(Q) <= 1, 'Q is not normalized.'
    assert np.nansum(P<=0) == 0, 'Distribution cannot have a negative probability.'
    assert np.nansum(Q<=0) == 0, 'Distribution cannot have a negative probability.'
    
    Q[Q == 0] = np.nan
    P[P == 0] = np.nan

    kl = P*np.log(P/Q)
    
    return np.nansum(kl)
    
    
    