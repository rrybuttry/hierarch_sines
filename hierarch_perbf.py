import model_perbf_nbevid
import emcee
import numpy as np
import multiprocessing as mp
import corner
import matplotlib.pyplot as plt
import argparse
import pickle


def dosamp(name, path):
    
    args = (path + '/'+name+'/bin', path + '/'+name+'/nb')

    nw = 30
    ndim = 3
    nsteps = 300
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
        ])
        if model_perbf_nbevid.hierarch_like(p, *args) > -1e20:
            xs.append(p)

    # xs = np.random.normal(size=(nw, 5)) * .2 + .5
    with mp.Pool(nthreads) as poo:
        es = emcee.EnsembleSampler(nw,
                                   ndim,
                                   model_perbf_nbevid.hierarch_like,
                                   args=args,
                                   pool=poo)
        es.random_state = np.random.mtrand.RandomState(seed).get_state()
        R = es.run_mcmc(xs, nsteps)
        xs = R[0]
        es.reset()
        R = es.run_mcmc(xs, nsteps)
    #ptrue = [.5, .2, .6, 12, 10]
    #corner.corner(es.chain[:, 0:, :].reshape(-1, ndim),
    #              truths=ptrue,
    #              labels=['mlogp', 'slogp', 'bfrac', 'vel', 'disp'])
    #plt.savefig('chain.png')
    
    with open(path+'/chain_%s.pkl'%name, 'wb') as f:
        pickle.dump(es.chain[:, 0:, :].reshape(-1, ndim), f)


if __name__ == '__main__':
    # start argparsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="name of dataset")
    parser.add_argument("--folder", type=str, help="what folder are the binary param samples stored in (absolute path)? emcee chain is saved in folder right above it")    
    args = parser.parse_args()
    
    dosamp(name=args.name, path=args.folder)
