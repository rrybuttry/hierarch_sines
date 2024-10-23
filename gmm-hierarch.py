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
#from idlsave import idlsave# importing sergey's idlsave

print(mp.cpu_count())

DISP_PRIOR = 100  # width of gaussian prior on velocity
class si:
    cache_vel = None
    cache_evid = None
    
def gau_err_samp(vels,
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

def make_samp(Nstars,
              Nsamp = 100,
              mean1 = 1.0,
              std1 = 1.0,
              frac1 = 0.5,
              mean2 = -1.0,
              std2 = 1.0,
              vel_err = 0.5,
              seed=44):
    rng = np.random.default_rng(seed)
    
    vel1 = rng.normal(size=Nstars) * std1 + mean1
    vel2 = rng.normal(size=Nstars) * std2 + mean2
    is_1 = (rng.uniform(0, 1, size=Nstars) < frac1).astype(int)
    
    res = []
    evids = []
    for i in range(Nstars):
        # generate data with some errors
        v = (vel1[i] * is_1[i]) + (vel2[i] * (is_1[i]==0)) + rng.normal(size=1) * vel_err
        # generate posterior samples from the samples 
        samples, evid = gau_err_samp(np.ones(1) * v, np.ones(1) * vel_err, size=Nsamp, rng=rng)
        res.append(samples)
        evids.append(evid)
        
    si.cache_vel = np.array(res)
    si.cache_evid = np.array(evid)

def hyper_like(p):
    """
    Hierarchical likelihood
    """
    
    meanvel1, sigvel1, frac1, meanvel2, sigvel2 = p
    
    if frac1>=1 or frac1<=0 or sigvel1<0 or sigvel2<0:
        return -1e100
    
    
    # extract samples from cache
    VEL = si.cache_vel
    nsamp = VEL.shape[1]
    # -------------likelihood part---------------

    # calc vel part of like
    NNrv0 = scipy.stats.norm(0, DISP_PRIOR) # fiducial vel 
    NNrv1 = scipy.stats.norm(meanvel1, sigvel1)
    NNrv2 = scipy.stats.norm(meanvel2, sigvel2)
    #perpr1 = NNrv1.logpdf(VEL) 
    #perpr2 = NNrv2.logpdf(VEL) 
    
    lrat_1 = NNrv1.logpdf(VEL) - NNrv0.logpdf(VEL) # prior ratio 
    lrat_2 = NNrv2.logpdf(VEL) - NNrv0.logpdf(VEL) # prior ratio 
    
    # combined prior ratios
    perpr1 = scipy.special.logsumexp(lrat_1, axis=1) - np.log(nsamp)
    perpr2 = scipy.special.logsumexp(lrat_2, axis=1) - np.log(nsamp)
    
    # populations
    like1 = perpr1 + si.cache_evid + np.log(frac1)
    like2 = perpr2 + si.cache_evid + np.log(1 - frac1)
    ret = np.logaddexp(like1, like2).sum(axis=0)
    
    if not np.isfinite(ret):
        print('oops', p, ret)
        ret = (-1e100)
        
    return ret


def dosamp(name, path):

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
            rst.uniform(-1, 1),
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
        mean1 = 1.0
        std1 = 1.0
        frac1 = 0.5
        mean2 = -1
        std2 = 1.0
    else:        
        mean1 = args.par[0]
        std1 = args.par[1]
        frac1 = args.par[2]
        mean2 = args.par[3]
        std2 = args.par[4]

    
    seed = args.seed
    new_name = args.name + '_seed%i'%seed
    print(new_name, mean1, std1, frac1, mean2, std2)
    
    print('Make curve samples (save in si)\n')
    make_samp(Nstars,
                 mean1 = mean1,
                 std1 = std1,
                 frac1 = frac1,
                 mean2 = mean2,
                 std2 = std2,
                 seed=seed)
    
    print('hyper parameter sampling \n')
    dosamp(name=new_name, path=args.folder)
