import numpy as np
import dynesty
import emcee
import scipy.special
import scipy.stats
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
DISP_PRIOR = 100
def nb_samp_evid(vels,
                evels,
                jitter=0.0,
                zpt_prior_width=DISP_PRIOR,
                rng = np.random.default_rng(), 
                size=1):
    """

    Parameters:
    -----------
    zpt_prior_width: real
        This is the width of the gaussian of the prior on the zpt of the RV
    return_samp: boolean
        If true then the sample of parameters is returned (including sin(i))
    jitter: real
        The extra RV uncertainty
    """
    if hasattr(zpt_prior_width, 'unit'):
        zpt_prior_width = zpt_prior_width.to_value(auni.au / auni.year)
    if hasattr(jitter, 'unit'):
        jitter = jitter.to_value(auni.au / auni.year)
    # t2 = time.time()

    # as a first step everything is divided by errors
    vels = vels / np.sqrt(evels**2 + jitter**2)
    II = 1. / np.sqrt(evels**2 + jitter**2)
    ITV = (II * vels).sum()

    norm = 1. / zpt_prior_width**2 + (1. / (evels**2 + jitter**2)).sum()
    
    # ----------------Evidence things----------------
    logfrontmult = -np.log(zpt_prior_width) - 0.5 * np.log(evels**2 +
                                                           jitter**2).sum()
    # this is the I^T I + 1/s^2 term
    VTZV = (vels**2).sum() - 1. / norm * ITV**2
    lnorm_nbin = -0.5 * np.log(norm) - 0.5 * VTZV + logfrontmult
    
    # correction comes from dropping  1/sqrt(2pi ev**2) in the like and 
    # from dropping 1/sqrt(2pi)^n in binary model nb evid calc
    correction = -(- 0.5 * np.log(evels**2 + jitter**2).sum() - 0.5*np.log(2*np.pi))
    evid = lnorm_nbin - len(vels) * 0.5 * np.log(2*np.pi)   + correction
    # ------------------------------------------------
    
    # return samples
    samples = ITV/norm + rng.normal(size=size)/np.sqrt(norm)
    return samples, evid


class KDEsave:
    cache_Z = None
    cache_Znb = None
    dat = None
    is_binary = None
    kde_bin = None
    kde_nb = None
    diffs =None
    
    
def getlogzs(velocities, binarity, path, rng = np.random.default_rng(), s = 2500):
    # load up the evidence values from a run that has the same hyper parameters
    if KDEsave.cache_Z is None or KDEsave.cache_Znb is None or KDEsave.dat is None or KDEsave.is_binary is None:
        prefix = path + '/bin'
        prefix_nb = path + '/nb'
        
        dat = []
        truep = []
        evid = []
        for _ in sorted(glob.glob(prefix+'*psav')):
            temp = idlsave.restore(_, 'samp, logz, dat, truep')
            evid.append(temp[1])
            dat.append(temp[2])
            truep.append(temp[3])
        KDEsave.cache_Z = np.array(evid)  # store binary evidence Z
        KDEsave.is_binary = np.array(truep)[:,-1]==1.0
        KDEsave.dat = dat
        
        evid = []
        for _ in sorted(glob.glob(prefix_nb+'*psav')):
            temp = idlsave.restore(_, 'samp, logz, dat, truep')
            evid.append(temp[1])
        KDEsave.cache_Znb = np.array(evid)
        KDEsave.diffs = KDEsave.cache_Z - KDEsave.cache_Znb
        
    if KDEsave.kde_bin is None or KDEsave.kde_nb is None:
        is_bin = KDEsave.is_binary
        vels = np.array([cur[1] for cur  in KDEsave.dat])
        
        arr = np.array([vels[is_bin][:,0], KDEsave.cache_Z[is_bin]- KDEsave.cache_Znb[is_bin]])
        KDEsave.kde_bin = scipy.stats.gaussian_kde(arr)
        arr = np.array([vels[~is_bin][:,0], KDEsave.cache_Z[~is_bin]- KDEsave.cache_Znb[~is_bin]])
        KDEsave.kde_nb = scipy.stats.gaussian_kde(arr)
    
    binevids = []
    nbevids = []
    for vel, b in zip(velocities, binarity):

        d = rng.uniform(min(diffs),max(diffs), size=s)
        v = np.ones_like(d) * vel
        if b:
            logpdfs = KDEsave.kde_bin.logpdf(np.array([v,d]))
        else:
            logpdfs = KDEsave.kde_nb.logpdf(np.array([v,d]))
        acceptance = np.log(rng.uniform(size=s)) < (logpdfs - np.max(logpdfs))
        dlogL = np.random.choice(d[acceptance], size=3)[0]
        
        Znb, _ = nb_samp_evid(vel, np.array([0.0]))
        
        binevids.append(Znb + dlogL)
        nbevids.append(Znb)
        
    return binevids, nbevids


class si:
    cache_vel = None
    cache_per = None
    cache_logz = None
    cache_logz_nb = None
    
def make_samp(Nstars,
              mean_vel = 1.0,
              std_vel = 1.0,
              frac1 = 0.5,
              mean_per = 0.5,
              std_per = 0.1,
              vel_err=0.5,
              min_per = 0.01, 
              max_per=10,
              seed=44, sampfolder=None):
    rng = np.random.default_rng(seed)
    is_1 = (rng.uniform(0, 1, size=Nstars) < frac1).astype(int)
    
    vel1 = rng.normal(size=Nstars) * std_vel + mean_vel
    vel2 = rng.uniform(-50, 50, size=Nstars)
    
    per1 = rng.normal(size=Nstars * 100) * std_per + mean_per
    per1 = per1[np.logical_and(per1>min_per, per1<max_per)][:Nstars] # truncated normal
    per2 = 10**rng.uniform(np.log10(min_per) , np.log10(max_per), size=Nstars)
    
    vels = []
    pers = []
    for i in range(Nstars):
        v = (vel1[i] * is_1[i]) + (vel2[i] * (is_1[i]==0))
        p = (per1[i] * is_1[i]) + (per2[i] * (is_1[i]==0))
        vels.append([v])
        pers.append([p])
        
    si.cache_vel = np.array(vels)
    si.cache_per = np.array(pers)
    
    # calc some evidences to use
    Zbin, Znb = getlogzs(vels, is_1==1.0, path=sampfolder)
    si.cache_logz = np.array(Zbin)
    si.cache_logz_nb = np.array(Znb)

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
    prior_per = scipy.stats.uniform(loc = min_per, scale = max_per-min_per).logpdf(PER)
    
    # calc vel part of like
    NNrv1 = scipy.stats.norm(meanvel, stdvel)
    Unifrv = scipy.stats.uniform(loc = -50, scale = 100)
    
    # binary-nonbinary populations
    perpr1 = NNrv1.logpdf(VEL) + model_per
    perpr2 = Unifrv.logpdf(VEL) + prior_per
    
    like1 = perpr1 + si.cache_logz + np.log(frac1)
    like2 = perpr2 + si.cache_logz_nb + np.log(1 - frac1)
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
                 seed=seed, sampfolder = folder + 'twopar-hierarch_test1_seed456')
    
    print('hyper parameter sampling \n')
    dosamp(name=new_name, path=args.folder)
