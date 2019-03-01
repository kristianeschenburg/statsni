#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 13:19:59 2018

@author: kristianeschenburg
"""

from sklearn.covariance import shrunk_covariance
from scipy import stats
import numpy as np

def one_sample(samples,pop_mu,covariance):
    
    """
    Compute one-sample Hotteling's T-Squared statistics (T2).  By default,
    users the user-supplied covariance matrix.  However, if the matrix is
    found to be singular, method applies shrinkage to the condition number of
    the original covariance matrix.
    
    Parameters:
    - - - - -
        samples : array of data samples
        pop_mu : population mean
        covariance : estimate of sample covariance matrix
    
    Returns:
    - - - -
        t : dictionary containing t2-statistic, degrees of freedom, and
            chi-squared p-value
        f : dictionary containing f-statistic, numerator degrees of freedom,
            denominator degrees of freedom, and F-distribution p-value
    """
    
    # test whether the supplied covariance matrix are singular
    covariance = condition(covariance)
    precision = np.linalg.inv(covariance)

    [n,p] = np.shape(samples)
    
    samp_mu = np.mean(samples,axis=0)[:,np.newaxis]    
    deviation = samp_mu - pop_mu
    
    t2 = n*np.dot(deviation.T,np.dot(precision,deviation))
    chi_p = stats.chi2.pdf(t2,df=p)
    
    dfn = (n-p)
    dfd = p*(n-1.)
    f = (dfn/dfd)*t2
    f_p = stats.f.pdf(f,p,dfn)

    t = {'t2-statistic': t2,
         'df': p,
         'p-value': chi_p}
    
    f = {'f-statistic': f,
         'dfn': dfn,
         'dfd': dfd,
         'p-value': f_p}
        
    return [t,f]

def two_sample(samples1,samples2,covariance1,
                  covariance2,equal=True):
    
    """
    Compute two-sample Hotelling's T2-Squared statistic.  By default, assumes 
    both data samples have covariance matrices that are random samples from 
    the same distribution.  
    
    By default, uses the user-supplied covariance matrices.  However, if the 
    matrices is found to be singular, method applies shrinkage to the 
    condition number of the original covariance matrices.
    
    Parameters:
    - - - - -
        samples1, samples2 : array of data samples for two populations
        pop_mu1, pop_mu2 : population means for both samples
        covariance1, covariance2 : estimates of sample covariance matrices
        equal : boolean indicating whether to assume equal covariance matrices
    
    Returns:
    - - - -
        t : dictionary containing t2-statistic, degrees of freedom, and
            chi-squared p-value
        f : dictionary containing f-statistic, numerator degrees of freedom,
            denominator degrees of freedom, and F-distribution p-value
    """
    
    [n1,p1] = samples1.shape
    [n2,p2] = samples2.shape
    
    mu1 = np.mean(samples1,axis=0)
    mu2 = np.mean(samples2,axis=0)
    deviation=mu1-mu2
    
    # test whether the supplied covariance matrices are singular
    covariance1 = condition(covariance1)
    covariance2 = condition(covariance2)
    
    # equal variances assumes
    if equal:
        # pooled covariance matrix
        S = ((n1-1.)* covariance1 + (n2-1.)*covariance2)/(n1 + n2 - 2.)
        S = S * (1./n1 + 1./n2)
        # pooled precision matrix
        P = np.linalg.inv(S)
        
        # compute T-statistic and Chi-2 test
        t2 = np.dot(deviation.T,np.dot(P,deviation))
        chi_p = stats.chi2.pdf(t2,p1)
        
        # compute F-statistic and F-test
        dfn = (n1 + n2 - p1 - 1.)
        dfd = p1 * (n1 + n2 -2.)
        f = (dfn/dfd) * t2
        f_p = stats.f.pdf(f,p1,dfn)
        
        f_dfd = dfn
        
        
    # unequal variances assumed
    else:
        # pooled covariance matrix
        S = (1./n1)*covariance1 + (1./n2)*covariance2
        # pooled precision matrix
        P = np.linalg.inv(S)
        
        # compute T-statistic and Chi-2 test
        t2 = np.dot(deviation.T,np.dot(P,deviation))
        chi_p = stats.chi2.pdf(t2,p1)
        
        # compute F-statistic and F-test
        s1 = (1./n1)*np.dot(P,np.dot(covariance1,P))
        s2 = (1./n2)*np.dot(P,np.dot(covariance2,P))
        
        nu1 = (1./(n1-1)) * (np.dot(deviation.T,np.dot(s1,deviation))/t2)**2
        nu2 = (1./(n2-1)) * (np.dot(deviation.T,np.dot(s2,deviation))/t2)**2
        nu = 1/(nu1+nu2)
        
        dfn = (n1 + n2 - p1 - 1.)
        dfd = p1 * (n1 + n2 - 2.)
        f = (dfn/dfd)*t2
        f_p = stats.f.pdf(f,p1,nu)
        
        f_dfd = nu

    t = {'t2-statistic': t2,
         'df': p1,
         'p-value': chi_p}
    
    f = {'f-statistic': f,
         'dfn': p1,
         'dfd': f_dfd,
         'p-value': f_p}
        
    return [t,f]

def condition(covariance):
    
    """
    Test whether the supplied covariance matrix is singular.  If it is, 
    applies a shrinkage penalty to reduce the condition number of the matrix.
    
    Parameters:
    - - - - -
        covariance : precomputed covariance matrix
    """
    
    try:
        np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        covariance = shrunk_covariance(covariance)
    finally:
        return covariance
        