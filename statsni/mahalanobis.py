#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 12:27:32 2018

@author: kristianeschenburg
"""

from scipy.stats import ncx2,chi2
import numpy as np

def mahalanobis(samples,mu,sigma):
    
    """
    Compute the Mahalanobis distance for a set of samples from an MVN
    with parameters mean = mu and covariance = sigma.
    
    Parameters:
    - - - - -
        samples : test samples
        mu : mean of target distribution
        sigma : covariance of target distribution
    """
    
    try:
        P = np.linalg.inv(sigma)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError('Covariance matrix is singular.')
        
    if mu.ndim == 1:
        mu =  mu[np.newaxis,:]
    
    deviation = samples - mu
    quadratic = np.dot(deviation,np.dot(P,deviation.T))
    
    distances = np.sqrt(np.diag(quadratic))
    
    return distances

def noncentrality(mu,H):
    
    """
    Compute the non-centrality parameter of a non-central chi-squared
    distribution.
    
    Parameters:
    - - - - - 
        mu : mean vector
        H : transformation matrix
        
    where the quadratic term has the form (x-mu)'H(x-mu) such that the
    non-centrality parameter \phi = (mu)'H(mu)/2.
    """
    
    mu = mu.squeeze()
    if mu.ndim == 1:
        mu = mu[np.newaxis,:]
        
    phi = (1./2) * np.dot(mu.T,np.dot(H,mu))
    
    return phi

def chi_x2(samples,df):
    
    """
    Compute the central chi-squared statistics for set of chi-squared 
    distributed samples.
    
    Parameters:
    - - - - -
        samples : chi-square random variables
        df : degrees of freedom
    """
    
    return chi2.pdf(samples,df)

def nchi_x2(samples,df,nc):
    
    """
    Compute the non-central chi-squared statistics for a set non-central 
    chi-squared distributed samples.
    
    Parameters:
    - - - - -
        samples : chi-square random variables
        df : degrees of freedom
        nc : non-centrality parameter
    """
    
    return ncx2.pdf(samples,df,nc)