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

VREF = 30

DISP_PRIOR = 100  # width of gaussian prior on velocity

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


class si:
    cache_per = None
    cache_vel_bin = None
    cache_vel_nb = None
    cache_logz = None
    cache_logz_nb = None
    prefixes = None
    cache_dat_bin = None
    cache_dat_nb = None
    cache_truep_bin = None
    cache_truep_nb = None


    
def make_samp(Nstars,
              Npt,
              dt=5,
              bin_frac=0.5,
              min_per=0.01,
              max_per=10,
              mean_per = 2.4,
              std_per = 2.28,
              mean_vel=0.0,
              std_vel=5,
              vel_err=0.5,
              seed=44):
    rng = np.random.default_rng(seed)
    
    v0 = rng.normal(size=Nstars) * std_vel + mean_vel
    per = rng.normal(size=Nstars * 100) * std_per + mean_per
    per = per[np.logical_and(per>min_per, per<max_per)][:Nstars] # truncated normal
    
    #per = 10**rng.uniform(np.log10(min_per), np.log10(max_per), size=Nstars)
    phase = np.ones_like(per) * 0.0 #rng.uniform(0, 2 * np.pi, size=Nstars)
    #cosi = rng.uniform(0, 1, size=Nstars)
    sini = np.ones_like(per) * 1.0 #np.sqrt(1 - cosi**2)
    amp0 = VREF / per**(1. / 3) * sini
    res = []
    is_bin = (rng.uniform(0, 1, size=Nstars) < bin_frac).astype(int)
    for i in range(Nstars):
        ts = rng.uniform(0, dt, size=Npt)
        v = (v0[i] + amp0[i] * is_bin[i] * np.sin(2 * np.pi / per[i] * ts - phase[i]) +
             rng.normal(size=Npt) * vel_err)
        ev = v * 0 + vel_err
        res.append([ts, v, ev])
    truep = np.array([v0, per, phase, sini, is_bin]).T
    return res, truep

class Prior:

    def __init__(self, vel_sig, min_per, max_per):
        self.MINV = -1000
        self.min_per = min_per
        self.max_per = max_per
        self.vel_sig = vel_sig

    def __call__(self, x):
        V = scipy.special.ndtri(x[0]) * self.vel_sig
        per = self.min_per * (self.max_per / self.min_per)**x[1]
        x1 = x * 0
        x1[0] = V
        x1[1] = per
        #x1[2] = x[2] * 2 * np.pi # phase
        #x1[3] = np.sqrt(1 - x[3]**2)  # sini
        return x1


def like(p, data):
    t, v, ev, bin_switch = data
    if bin_switch:
        v0, per = p
        sini = 1 # set sini=1
        phase = 0 # set phase = 0
    else:
        v0 = p[0]
        per, phase = 1, 0
        sini = 0
    amp0 = VREF / per**(1. / 3) * sini 
    model = amp0 * np.sin(2 * np.pi / per * t - phase) + v0
    logl = -0.5 * np.sum(((model - v) / ev)**2) # dropped 1/sqrt(2pi ev**2)
    return logl


def posterior(t, v, ev, binary, minp=None, maxp=None, seed=1):
    if binary:
        pri = Prior(DISP_PRIOR, minp, maxp)
        ndim = 2
        periodic = None #[2]

        data = ((t, v, ev, binary), )

        rng = np.random.default_rng(seed)
        nlive = 1000
        dns = dynesty.DynamicNestedSampler(like,
                                           pri,
                                           ndim,
                                           rstate=rng,
                                           nlive=nlive,
                                           bound='multi',
                                           sample='rslice',
                                           periodic=periodic,
                                           logl_args=data)
        # dns.run_nested(n_effective=10000, print_progress=False)
        print_progress = False
        dns.run_nested(n_effective=10000,print_progress=print_progress)  #, maxbatch=1)
        # for i in range(10):
        #    dns.add_batch(mode='full')
        res = dns.results.samples_equal()
        logz = dns.results.logz[-1]
    else:
        # use analytical form of the evidence
        res, logz = nb_samp_evid(v,ev, size=(10000,1))
        
        
    return res, logz


def hierarch_like(p, prefix, prefix_nb, nsamp=1000, seed=12, min_per=0.1, max_per=10):
    """
    Hierarchical likelihood
    """
    #meanper,  binfrac = p
    meanper, stdper, binfrac, meanvel, sigvel = p
    
    if binfrac>=1 or binfrac<=0 or np.abs(meanper)>10 or stdper<0 or stdper > 10:
        return -1e100
    
    # load both bin and nb samps and logz's
    nsamp0 = 10000
    if si.cache_logz is None or si.cache_logz_nb is None:
        
        # load binary samp
        persamp = []
        velsamp=[]
        evid = []
        
        # load data for checking
        dat = []
        truep = []
        for _ in sorted(glob.glob(prefix+'*psav')):
            temp = idlsave.restore(_, 'samp, logz, dat, truep')
            persamp.append(temp[0][:, 1][:nsamp0])
            velsamp.append(temp[0][:, 0][:nsamp0])
            evid.append(temp[1])
            dat.append(temp[2])
            truep.append(temp[3])
            
        ARR = np.array(persamp) # store binary period samples
        ARR_vel = np.array(velsamp)
        si.cache_logz = np.array(evid)  # store binary evidence Z
        si.cache_dat_bin = dat
        si.cache_truep_bin = truep
        
        # do the permutations
        rng = np.random.default_rng(seed)
        permut = rng.integers(nsamp0, size=(ARR.shape[0], nsamp))
        #ARR = ARR[np.arange(ARR.shape[0])[:, None] + permut * 0, permut]
        # save
        si.cache_per = ARR[np.arange(ARR.shape[0])[:, None] + permut * 0, permut]
        si.cache_vel_bin  = ARR_vel[np.arange(ARR.shape[0])[:, None] +permut * 0, permut]
        
        # load nbin samp
        velsamp = []
        evid = []
        
        dat = []
        truep = []
        for _ in sorted(glob.glob(prefix_nb+'*psav')):
            temp = idlsave.restore(_, 'samp, logz, dat, truep')
            velsamp.append(temp[0][:, 0][:nsamp0])
            evid.append(temp[1])
            dat.append(temp[2])
            truep.append(temp[3])
        
        ARR_vel = np.array(velsamp)
        si.cache_logz_nb = np.array(evid) # store non-binary evidence Z
        si.cache_dat_nb = dat
        si.cache_truep_nb = truep
        
        # do the permutation (doesn't really matter here) and save
        si.cache_vel_nb = ARR_vel[np.arange(ARR.shape[0])[:, None] +permut * 0, permut]
        
        si.prefixes=[prefix, prefix_nb]
    
    
    # extract samples from cache
    PER = si.cache_per
    VEL_bin = si.cache_vel_bin
    VEL_nbin = si.cache_vel_nb
    
    # -------------likelihood part---------------
    
    # calc per part of like
    pi0_per = scipy.stats.loguniform(min_per, max_per).logpdf(PER)
    NNper = scipy.stats.norm(meanper, stdper)
    pernorm = NNper.cdf(max_per) - NNper.cdf(min_per)# normalization per truncated norm
    model_per = NNper.logpdf(PER) - np.log(pernorm) 

    # calc vel part of like
    NNrv = scipy.stats.norm(meanvel, sigvel)
    NNrv0 = scipy.stats.norm(0, DISP_PRIOR) # fiducial vel 
    lrat_bin = NNrv.logpdf(VEL_bin) - NNrv0.logpdf(VEL_bin) # prior ratio binary
    lrat_nbin = NNrv.logpdf(VEL_nbin) - NNrv0.logpdf(VEL_nbin) # prior ratio nonbinary
    
    # combined prior ratios
    perpr1 = scipy.special.logsumexp(model_per - pi0_per + lrat_bin, axis=1) - np.log(nsamp)
    perpr2 = scipy.special.logsumexp(lrat_nbin, axis=1) - np.log(nsamp)
    
    # binary-nonbinary populations
    like1 = perpr1 + si.cache_logz + np.log(binfrac)
    like2 = perpr2 + si.cache_logz_nb + np.log(1 - binfrac)
    ret = np.logaddexp(like1, like2).sum(axis=0)
    
    if not np.isfinite(ret):
        print('oops', p, ret, pernorm)
        ret = (-1e100)
        
    return ret


def dosamp(name, path, min_per = 0.01, max_per=10,):
    args = (path + '/'+name+'/bin', path + '/'+name+'/nb')
    
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
        if hierarch_like(p, *args) > -1e20:
            xs.append(p)

    # xs = np.random.normal(size=(nw, 5)) * .2 + .5
    with mp.Pool(nthreads) as poo:
        es = emcee.EnsembleSampler(nw,
                                   ndim,
                                   hierarch_like,
                                   args=args,
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
    parser.add_argument("--npt", type=int, default=1, help="number of observations per curve")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--par", nargs=5, type=float, default=None, help="hyper parameters")
    parser.add_argument("--binsamp", action='store_true', help="do individual binary sampling")
    parser.add_argument("--hierarch", action='store_true', help="do hierarchical inference")
    
    args = parser.parse_args()
    #print(args.hierarch)
    Npt = args.npt
    Nstars = args.nstars
    vel_err = 0.5
    min_per = 0.1
    max_per = 10
    
    if args.par is None:
        mean_per = 1.0
        std_per = 1.0
        bin_frac = 0.5
        mean_vel = -1
        std_vel = 1.0
    else:        
        mean_per = args.par[0]
        std_per = args.par[1]
        bin_frac = args.par[2]
        mean_vel = args.par[3]
        std_vel = args.par[4]

    
    seed = args.seed
    path = args.folder+"/" +args.name + '_seed%i/'%seed # folder for binary samps
    new_name = args.name + '_seed%i'%seed
    print(new_name, args.par)    
    try:
        os.mkdir(path)
    except:
        pass
    
    if args.binsamp:
        print('Make curves \n')
        S,truep = make_samp(Nstars, 
                            Npt=Npt,
                            mean_per = mean_per,
                            std_per = std_per,
                            bin_frac = bin_frac,
                            mean_vel = mean_vel,
                            std_vel = std_vel,
                            min_per=min_per,
                            max_per=max_per,
                            seed=seed)
        
        print('sampling nonbinary posteriors\n')
        with mp.Pool(mp.cpu_count()-2) as poo:
            binary_model=False
            pref = path+'nb'
            
            R = []
            for i in range(Nstars):
                cur_dat = S[i]
                cur_truep = truep[i]
                kargs = (cur_dat[0], cur_dat[1], cur_dat[2], binary_model, min_per,
                        max_per, i)
                R.append((i, poo.apply_async(posterior, kargs), cur_dat, cur_truep))
            for cur_i, cur_r, cur_dat, cur_true in R:
                cur_samp, cur_logz = cur_r.get()
                idlsave.save(f'{pref}_{cur_i:05d}.psav', 'dat, samp, logz, truep',
                             cur_dat, cur_samp, cur_logz, cur_true)
        
        print('sampling binary posteriors\n')
        with mp.Pool(mp.cpu_count()-2) as poo:
            binary_model=True
            pref = path+'bin'
            R = []
            for i in range(Nstars):
                cur_dat = S[i]
                cur_truep = truep[i]
                kargs = (cur_dat[0], cur_dat[1], cur_dat[2], binary_model, min_per,
                        max_per, i)
                R.append((i, poo.apply_async(posterior, kargs), cur_dat, cur_truep))
            for cur_i, cur_r, cur_dat, cur_true in R:
                cur_samp, cur_logz = cur_r.get()
                idlsave.save(f'{pref}_{cur_i:05d}.psav', 'dat, samp, logz, truep',
                             cur_dat, cur_samp, cur_logz, cur_true)
                
    if args.hierarch:
        print('Hyper parameter sampling \n')
        dosamp(name=new_name, path=args.folder)
