import numpy as np
import dynesty
import emcee
import scipy.special
import pickle
import multiprocessing as mp
import glob
import argparse

# Rachel's Path things
import os
HOME = os.environ["HOME"]+'/'
VERA = HOME#+'Research/Vera/'
os.sys.path.append(VERA+'Research/')
from idlsave import idlsave# importing sergey's idlsave
#from binary_bayes.utils import TruncatedLogNormal


print(mp.cpu_count())


class si:
    cache_vel = None
    cache_per = None
    
def make_samp(Nstars,
              mean_vel = 1.0,
              std_vel = 1.0,
              frac1 = 0.5,
              mean_per = 0.5,
              std_per = 0.1,
              vel_err=0.5,
              min_per = 0.01, 
              max_per=10,
              seed=44):
    rng = np.random.default_rng(seed)
    is_1 = (rng.uniform(0, 1, size=Nstars) < frac1).astype(int)
    
    vel1 = rng.normal(size=Nstars) * std_vel + mean_vel
    vel2 = rng.uniform(-50, 50, size=Nstars)
    
    per1 = rng.normal(size=Nstars * 100) * std_per + mean_per
    per1 = per1[np.logical_and(per1>min_per, per1<max_per)][:Nstars] # truncated normal
    per2 = 10**rng.uniform(np.log10(min_per) , np.log10(max_per), size=Nstars) # loguniform background
    
    vels = []
    pers = []
    for i in range(Nstars):
        v = (vel1[i] * is_1[i]) + (vel2[i] * (is_1[i]==0))
        p = (per1[i] * is_1[i]) + (per2[i] * (is_1[i]==0))
        vels.append([v])
        pers.append([p])
        
    si.cache_vel = np.array(vels)
    si.cache_per = np.array(pers)

def hyper_like(p,min_per = 0.01,  max_per=10,):
    """
    Hierarchical likelihood
    """
    
    meanper, stdper, frac1, meanvel, stdvel = p
    
    if frac1>=1 or frac1<=0 or stdper<0 or stdvel<0 or np.abs(meanper)>10:
        return -1e100
    
    
    # extract samples from cache
    VEL = si.cache_vel
    PER = si.cache_per
    # -------------likelihood part---------------

    # calc per part of like
    NNper = scipy.stats.norm(meanper, stdper)
    pernorm = NNper.cdf(max_per) - NNper.cdf(min_per)# normalization per truncated lognorm
    model_per = NNper.logpdf(PER) - np.log(pernorm)
    prior_per = scipy.stats.loguniform(min_per, max_per).logpdf(PER)
    
    # calc vel part of like
    NNrv1 = scipy.stats.norm(meanvel, stdvel)
    Unifrv = scipy.stats.uniform(loc = -50, scale = 100)
    
    # binary-nonbinary populations
    perpr1 = NNrv1.logpdf(VEL) + model_per
    perpr2 = Unifrv.logpdf(VEL) + prior_per
    
    like1 = perpr1 + np.log(frac1)
    like2 = perpr2 + np.log(1 - frac1)
    ret = np.logaddexp(like1, like2).sum(axis=0)
    
    if not np.isfinite(ret):
        print('oops', p, ret)
        ret = (-1e100)
        
    return ret


def dosamp(name, path,min_per = 0.01, max_per=10,):

    nw = 40
    ndim = 5
    nsteps = 1000
    nthreads = mp.cpu_count()-2
    seed = 44442323
    rst = np.random.default_rng(seed)
    xs = []
    while len(xs) < nw:
        # meanper, stdper, binfrac, meanvel, sigvel
        p = np.array([
            rst.uniform(min_per, max_per),
            rst.uniform(0, 1),
            rst.uniform(0, 1),
            rst.uniform(-1, 1),
            rst.uniform(0, 1)
        ])
        if hyper_like(p) > -1e20:
            xs.append(p)

    # xs = np.random.normal(size=(nw, 5)) * .2 + .5
    with mp.Pool(nthreads) as poo:
        es = emcee.EnsembleSampler(nw,
                                   ndim,
                                   hyper_like,
                                   pool=poo)
        es.random_state = np.random.mtrand.RandomState(seed).get_state()
        R = es.run_mcmc(xs, nsteps)
        xs = R[0]
        es.reset()
        R = es.run_mcmc(xs, nsteps, progress=True)
        
    with open(path+'/chain_%s.pkl'%name, 'wb') as f:
        pickle.dump(es.chain[:, 0:, :].reshape(-1, ndim), f)
        
if __name__ == '__main__':
    
    # start argparsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default=VERA+"v/Test/Sinusoid/", help="what folder are the binary param samples stored in (absolute path)?")
    parser.add_argument("--name", type=str, help="name of dataset")
    parser.add_argument("--nstars", type=int, default=200, help="number of stars")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--par", nargs=5, type=float, default=None, help="hyper parameters")
    
    args = parser.parse_args()
    
    Nstars = args.nstars
    
    if args.par is None:
        mean_per = 1.0
        std_per = 1.0
        frac1 = 0.5
        mean_vel = -1
        std_vel = 1.0
    else:        
        mean_per = args.par[0]
        std_per = args.par[1]
        frac1 = args.par[2]
        mean_vel = args.par[3]
        std_vel = args.par[4]

    
    seed = args.seed
    new_name = args.name + '_seed%i'%seed
    print(new_name, args.par)
    
    print('Make curves (save in si)\n')
    make_samp(Nstars,
                 mean_per = mean_per,
                 std_per = std_per,
                 frac1 = frac1,
                 mean_vel = mean_vel,
                 std_vel = std_vel,
                 seed=seed)
    
    print('hyper parameter sampling \n')
    dosamp(name=new_name, path=args.folder)
