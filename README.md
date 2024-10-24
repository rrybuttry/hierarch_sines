# hierarch_sines
Example of how to run:
```
$ python model_lognormper_nbevid.py --name sines_N1000_1pt --headfolder ~/samples/ --npt 1 --nstars 1000 --seed 456 # generates curves from given params and seeds and samples non-binary posterior
$ python model_lognormper_nbevid.py --name sines_N1000_1pt --headfolder ~/samples/ --npt 1 --nstars 1000 --seed 456 --binary # generates curves from given params and seeds and samples binary posterior
$ python hierarch_lognormper.py --folder ~/samples/ --name sines_N1000_1pt_seed456 # MCMC to sample hierarchical likelihood using the binary and non-binary posterior samples
```

### Scripts
model_perbf_nbevid.py + hierarch_perbf.py - Infer period dist and binfrac

model_lognormper.py + hierarch_lognormper.py - Infer period dist, binfrac, and vel dist

model_lognormper_nbevid.py + hierarch_lognormper.py - Infer period dist, binfrac, and vel dist where the evidence integral for non-binaries is analytically calculated

model_perv0_nbevid.py + hierarch_lognormper.py - Same as "model_lognormper_nbevid.py" except that sini=1 and phase0=0; Combined the sampling into single step with sampling analytical form of nonbinary model

two-par-hierarch.py - basically the same as model_perv0_nbevid.py except that the period distribution is normal rather than lognormal

two-par_fakeevid.py - "Two population - two parameter model" where there are evidence values that are randomly determined based on the distrbutions found from "two-par-hierarch.py"

two-par.py - "Two population - two parameter model" 

gmm-hierarch.py - gaussian mixture model with two gaussians to represent two populations where the posterior distributions of the observed quantities are samples from a gaussian prior

gmm.py - gaussian mixture model with two gaussians to represent two populations