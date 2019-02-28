#!/usr/bin/env python
# coding: utf-8

import pandas as pd, matplotlib.pyplot as plt
import numpy as np, statsmodels.api as sm
import theano, theano.tensor as tt, pymc3 as mc
from theano.compile.ops import as_op
from facets import facets

class DLMMC(object):
    def __init__(self, data, dlm):
        self.data = data
        self.mod = dlm     
    
    def fit_mcmc(self, sigma_obs_prior=10, sigma_trend_prior=1e-4,  \
                       sigma_seas_prior=1e-2, sigma_ar_prior=1e-4, \
                       nsam=6000, warmup=1000, chains=4, cores=4):
        with mc.Model() as model:
            sigma_obs = mc.HalfNormal("sigma_obs", sd=sigma_obs_prior)
            sigma_trend = mc.HalfNormal("sigma_trend", sd=sigma_trend_prior)
            sigma_seas = mc.HalfNormal("sigma_seas", sd=sigma_seas_prior)
            sigma_ar = mc.HalfNormal("sigma_ar", sd=sigma_ar_prior)
            rho = mc.Uniform("rho", lower=0, upper=1)
            
            @theano.compile.ops.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar], \
                                      otypes=[tt.dvector])
            def loglikelihood(o,l,m,n,s):
                logp = self.mod.loglike([o,l,m,n,s], transformed=True)
                logp = np.array(logp).reshape((1,))
                return logp
            
            lgp = mc.Potential('lgp', loglikelihood(sigma_obs, sigma_trend, sigma_seas, sigma_ar, rho))
            self.method = mc.step_methods.metropolis.Metropolis()
            self.trace = mc.sample(nsam, warmup=warmup, chains=chains, cores=cores, step=self.method)
        # posteriori sampled parameters
        self.par_names = ['sigma_obs', 'sigma_trend', 'sigma_seas', 'sigma_ar', 'rho']
        self.post = pd.DataFrame(columns=self.par_names)
        for col in self.par_names:
            self.post[col] = self.trace.get_values(col)
        return self.post
    
    def simulate(self, nsam = None):
        if not nsam:
            nsam = self.post.shape[0]
        tslen = self.data.shape[0]
        nstat = self.mod.k_states
        self.simulated = np.zeros((nsam, nstat, tslen))
        for i in range(nsam):
            self.mod.update(self.post.loc[i, :])
            sim = self.mod.simulation_smoother()
            st = sim.simulate()
            self.simulated[i,...] = sim.simulated_state
        return self.simulated
        
    def plot_trace(self):
        mc.traceplot(self.trace) 
        
    def plot_trend(self, wsize = 12, ax=None, nsam = None, colors=['black', 'orange', 'black'], alpha=0.4):
        simulated = self.simulate(nsam = nsam)
        trend = pd.DataFrame(simulated[:,0,:], columns = self.data.index)
        trend = trend.T.rolling(wsize, center=True).apply(lambda x: x[-1]-x[0])
        avg_trend = trend.mean(axis=1)
        std_trend = trend.std(axis=1).abs()        
        if not ax:
            fig, ax = facets(1, 1, width=12, aspect=0.34); ax = ax[0]
        ax.axhline(0, ls = '-.', color=colors[0])
        avg_trend.plot(ax=ax, lw=2, color=colors[1])
        ax.fill_between(avg_trend.index, avg_trend - std_trend, avg_trend + std_trend, color=colors[2], alpha=alpha)
        ax.set_xlabel('Year', fontsize=16, fontweight='bold')
        ax.set_ylabel('Trend [ppbv/yr]', fontsize=16, fontweight='bold')
        return ax
        
dlm = pd.read_pickle('./data/dlm.tco.pkl')

kw = {'level': 'smooth trend', 'exog':dlm.reg,
      'freq_seasonal':[{"period":12, "harmonics":2}],
      'autoregressive':1, 'mle_regression':False}

ssm = sm.tsa.UnobservedComponents(dlm.d, **kw)
ssm.fit().plot_diagnostics(figsize=(15,7))
plt.show()

model = DLMMC(dlm.d, ssm)
post = model.fit_mcmc(nsam=5000)

model.plot_trend()
plt.show()


