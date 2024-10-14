#!/bin/bash

loc=$HOME/Research/toy_sinusoids/samples/
cd $loc/../hierarch_sines/


name=sines
npt=1
nstars=1000
seed=123
truths='2.1 2.0 0.6 0.0 10.0'

python model_lognormper.py --name ${name}_N${nstars}_${npt}pt --nstars ${nstars} --npt ${npt} --seed $seed --binary --headfolder $loc --par $truths
python model_lognormper.py --name ${name}_N${nstars}_${npt}pt --nstars ${nstars} --npt ${npt} --seed $seed --headfolder $loc --par $truths
python hierarch_lognormper.py --folder $loc --name ${name}_N${nstars}_${npt}pt_seed${seed}

python model_lognormper_nbevid.py --name ${name}_nbevid_N${nstars}_${npt}pt --nstars ${nstars} --npt ${npt} --seed $seed --binary --headfolder $loc --par $truths
python model_lognormper_nbevid.py --name ${name}_nbevid_N${nstars}_${npt}pt --nstars ${nstars} --npt ${npt} --seed $seed --headfolder $loc --par $truths
python hierarch_lognormper.py --folder $loc --name ${name}_nbevid_N${nstars}_${npt}pt_seed${seed}
