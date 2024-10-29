import numpy as np
import dynesty
import scipy.special
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
print(mp.cpu_count())

def evid_nb(vels,
            evels,
            jitter=0.0,
            zpt_prior_width=DISP_PRIOR):
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
    logfrontmult = -np.log(zpt_prior_width) - 0.5 * np.log(evels**2 +
                                                           jitter**2).sum()
    ITV = (II * vels).sum()

    norm = 1. / zpt_prior_width**2 + (1. / (evels**2 + jitter**2)).sum()
    # this is the I^T I + 1/s^2 term
    VTZV = (vels**2).sum() - 1. / norm * ITV**2
    lnorm_nbin = -0.5 * np.log(norm) - 0.5 * VTZV + logfrontmult
    
    # correction comes from dropping  1/sqrt(2pi ev**2) in the like and 
    # from dropping 1/sqrt(2pi)^n in binary model nb evid calc
    correction = -(- 0.5 * np.log(evels**2 + jitter**2).sum() - 0.5*np.log(2*np.pi))
    
    return lnorm_nbin - len(vels) * 0.5 * np.log(2*np.pi)   + correction

def make_samp(Nstars,
              Npt,
              Nsamp,
              dt=5,
              bin_frac=0.5,
              min_per=0.001,
              max_per=10,
              mean_logper = 2.4,
              std_logper = 2.28,
              vel_mean=0.0,
              vel_disp=5,
              vel_err=0.5,
              seed=44):
    rng = np.random.default_rng(seed)
    
    per = 10**(rng.normal(mean_logper,std_logper, size=Nstars*100) )
    per = per[(per>min_per)& (per<max_per)][:Nstars]
    
    #per = 10**rng.uniform(np.log10(min_per), np.log10(max_per), size=Nstars)
    phase = rng.uniform(0, 2 * np.pi, size=Nstars)
    v0 = rng.normal(size=Nstars) * vel_disp + vel_mean
    cosi = rng.uniform(0, 1, size=Nstars)
    sini = np.sqrt(1 - cosi**2)
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
        x1[2] = x[2] * 2 * np.pi # phase
        x1[3] = np.sqrt(1 - x[3]**2)  # sini
        return x1


class PriorNB:

    def __init__(self, vel_sig):
        self.MINV = -1000
        self.vel_sig = vel_sig

    def __call__(self, x):
        V = scipy.special.ndtri(x[0]) * self.vel_sig
        x1 = x * 0
        x1[0] = V
        return x1


def like(p, data):
    t, v, ev, bin_switch = data
    if bin_switch:
        v0, per, phase, sini = p
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
        ndim = 4
        periodic = [2]
        evid_analytic=None
    else:
        pri = PriorNB(DISP_PRIOR, )
        ndim = 1
        periodic = None
        # use analytical form of the evidence
        evid_analytic = evid_nb(v,ev)
        
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
    
    if evid_analytic is None:
        logz = dns.results.logz[-1]
    else:
        logz = evid_analytic
        
    return res, logz


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
        
        # do the permutation and save
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
    pernorm = NNper.cdf(np.log10(max_per)) - NNper.cdf(np.log10(min_per))# normalization per truncated lognorm
    model_per = NNper.logpdf(np.log10(PER)) - np.log(pernorm) - np.log(PER * np.log(10))

    # calc vel part of like
    NNrv = scipy.stats.norm(meanvel, sigvel)
    NNrv0 = scipy.stats.norm(0, DISP_PRIOR) # fiducial vel 
    lrat_bin = NNrv.logpdf(VEL_bin) - NNrv0.logpdf(VEL_bin) # prior ratio binary
    lrat_nbin = NNrv.logpdf(VEL_nbin) - NNrv0.logpdf(VEL_nbin) # prior ratio nonbinary
    
    # combined prior ratios (seperable v0 and per?)
    perpr1 = scipy.special.logsumexp(model_per - pi0_per, axis=1) - np.log(nsamp) + scipy.special.logsumexp(lrat_bin, axis=1) - np.log(nsamp)
    perpr2 = scipy.special.logsumexp(lrat_nbin, axis=1) - np.log(nsamp)
    
    # binary-nonbinary populations
    like1 = perpr1 + si.cache_logz + np.log(binfrac)
    like2 = perpr2 + si.cache_logz_nb + np.log(1 - binfrac)
    ret = np.logaddexp(like1, like2).sum(axis=0)
    
    if not np.isfinite(ret):
        print('oops', p, ret, pernorm)
        ret = (-1e100)
        
    return ret

if __name__ == '__main__':
    
    # start argparsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--headfolder", type=str, default=VERA+"v/Test/Sinusoid/", help="what folder are the binary param samples stored in (absolute path)?")
    parser.add_argument("--name", type=str, help="name of dataset")
    parser.add_argument("--nstars", type=int, default=200, help="number of stars")
    parser.add_argument("--npt", type=int, default=4, help="number of observations per curve")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument("--par", nargs=5, type=float, default=None, help="number of observations per curve")
    parser.add_argument("--binary", action='store_true', help="do the binary samples")
    
    args = parser.parse_args()
    
    Npt = args.npt
    Nstars = args.nstars
    vel_err = 0.5
    Nsamp = 1000
    min_per = 0.1
    max_per = 10
    
    if args.par is None:
        mean_logper = 2.1
        std_logper = 2.0
        bin_frac = 0.5
        vel_mean = 0.0
        vel_disp = DISP_PRIOR
    else:
        mean_logper = args.par[0]
        std_logper = args.par[1]
        bin_frac = args.par[2]
        vel_mean = args.par[3]
        vel_disp = args.par[4]

    
    seed = args.seed
    #path = VERA+'v/Test/Sinusoid/sines%iper_0.8muT_seed%i/'%(Npt, seed)
    path = args.headfolder+"/" +args.name + '_seed%i/'%seed # folder for binary samps
    
    try:
        os.mkdir(path)
    except:
        pass
    print(path, mean_logper, std_logper, bin_frac, vel_mean, vel_disp)
    
    #binary_model = True
    #binary_model = False
    binary_model = args.binary
    
    if binary_model:
        pref = path+'bin'
    else:
        pref = path+'nb'
    
    print('Make curves \n')
    S, truep = make_samp(Nstars,
                         Npt,
                         Nsamp,
                         dt=5,
                         bin_frac=bin_frac,
                         min_per=min_per,
                         max_per=max_per,
                         mean_logper = mean_logper,
                         std_logper = std_logper,
                         vel_mean=vel_mean,
                         vel_disp=vel_disp,
                         vel_err=vel_err, 
                         seed=seed)
    print('sampling \n')
    with mp.Pool(mp.cpu_count()-3) as poo:
        R = []
        for i in range(Nstars):
            cur_dat = S[i]
            cur_truep = truep[i]
            args = (cur_dat[0], cur_dat[1], cur_dat[2], binary_model, min_per,
                    max_per, i)
            R.append((i, poo.apply_async(posterior, args), cur_dat, cur_truep))
        for cur_i, cur_r, cur_dat, cur_true in R:
            cur_samp, cur_logz = cur_r.get()
            idlsave.save(f'{pref}_{cur_i:05d}.psav', 'dat, samp, logz, truep',
                         cur_dat, cur_samp, cur_logz, cur_true)
