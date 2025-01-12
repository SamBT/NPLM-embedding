import glob, h5py, math, time, os, json, argparse, datetime, sys
import numpy as np
from FLKutils import *
from SampleUtils import *
import uproot
import shutil
from tqdm import tqdm

def load_data(folder,rewgt=False):
    base=f"{folder}/predict_output/"
    fQCD = uproot.open(base+"pred_contrastive_embedding_ZJetsToNuNu.root")["Events"]
    fW = uproot.open(base+"pred_contrastive_embedding_WToQQ.root")["Events"]
    fZ = uproot.open(base+"pred_contrastive_embedding_ZToQQ.root")["Events"]
    fT = uproot.open(base+"pred_contrastive_embedding_TTBar.root")["Events"]
    fHb = uproot.open(base+"pred_contrastive_embedding_HToBB.root")["Events"]
    
    key = 'embeddings'
    eQCD = fQCD[key].array().to_numpy()
    eW = fW[key].array().to_numpy()
    eZ = fZ[key].array().to_numpy()
    eT = fT[key].array().to_numpy()
    eHb = fHb[key].array().to_numpy()
    
    if rewgt:
        print("Changing relative fractions of backgrounds to match H->bb search")
        nQCD = 5173887
        nT = 19213
        nZ = 48862
        nW = 142980
        
        nsum = nQCD + nT + nZ + nW
        fQCD = nQCD/nsum
        fT = nT/nsum
        fZ = nZ/nsum
        fW = nW/nsum

        fmax = np.max([fQCD,fT,fZ,fW])
        ntot = eQCD.shape[0]
        
        eQCD = eQCD[:int(ntot*fQCD/fmax)]
        eT = eT[:int(ntot*fT/fmax)]
        eW = eW[:int(ntot*fW/fmax)]
        eZ = eZ[:int(ntot*fZ/fmax)]
        
    data = np.concatenate([eQCD,eW,eZ,eT,eHb],axis=0)
    labels = np.concatenate([0*np.ones(eQCD.shape[0]),
                             1*np.ones(eW.shape[0]),
                             2*np.ones(eZ.shape[0]),
                             3*np.ones(eT.shape[0]),
                             4*np.ones(eHb.shape[0])],
                             axis=0).astype(int)
    
    rng = np.random.default_rng(seed=1382957)
    shuf = rng.permutation(data.shape[0])
    
    return data[shuf], labels[shuf]

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--signal', type=int, help="signal (number of signal events)", required=True)
parser.add_argument('-b', '--background', type=int, help="background (number of background events)", required=True)
parser.add_argument('-r', '--reference', type=int, help="reference (number of reference events, must be larger than background)", required=True)
parser.add_argument('-t', '--toys', type=int, help="toys", required=True)
parser.add_argument('-l', '--signalclass', type=int, help="class number identifying the signal", required=True)
parser.add_argument('--dim',type=int,required=True,help="dimension of contrastive space to analyze")
parser.add_argument('--temp',type=str,required=True,help="temperature of contrastive space to analyze")
parser.add_argument('--inputs',type=str,required=True,help="What training inputs to use (kinpid or full)")
parser.add_argument('--training',type=str,required=False,default="",help="Specific training to analyze (defaults to most recent)")
parser.add_argument('--fractions',action='store_true',default=False)
args = parser.parse_args()

N_ref = args.reference
N_bkg = args.background
N_sig = args.signal
rewgt_bkgs = args.fractions
w_ref = N_bkg*1./N_ref

# give a name to each model and provide a path to where the model's prediction for bkg and signal classes are stored
temp = args.temp
dim = args.dim
inputs = args.inputs
base_dir = f"/n/home11/sambt/contrastive_anomaly/training_JetClass/nate_scripts/trainings/{inputs}/ParT_contrastive_outDim{dim}_temp{temp}/"
trainings = [f for f in os.listdir(base_dir) if os.path.isdir(f"{base_dir}/{f}")]
if args.training != "":
    if os.path.isdir(f"{base_dir}/{args.training}"):
        target_dir = f"{base_dir}/{args.training}/"
    else:
        print(f"Requested training directory {base_dir}/{args.training} does not exist! Exiting...")
        sys.exit()
else:
    most_recent = max(trainings,key=lambda d: os.path.getctime(f"{base_dir}/{d}"))
    target_dir = f"{base_dir}/{most_recent}/"

# class number identifying the signal
sig_labels=[args.signalclass]
# class number identifying the background
bkg_labels=[0, 1, 2, 3]

# hyper parameters of the model
M=int(1.2*np.sqrt(N_ref))
flk_sigma_perc=90 #%
lam =1e-6
iterations=1_000_000
Ntoys = args.toys

# details about the save path
folder_out = f"{target_dir}/NPLM/"
if not os.path.isdir(folder_out):
    os.makedirs(folder_out)

sig_string = ''
sig_string+='SIG'
for s in sig_labels:
    sig_string+='-%i'%(s)

if args.fractions:
    NP = f"NPLM_realisticBkgRatios_{sig_string}_NR{int(N_ref/1000)}k_NB{int(N_bkg/1000)}k_NS{N_sig}_M{M}_lam{str(lam)}_iter{int(iterations/1e6)}M"
else:
    NP = f"NPLM_{sig_string}_NR{int(N_ref/1000)}k_NB{int(N_bkg/1000)}k_NS{N_sig}_M{M}_lam{str(lam)}_iter{int(iterations/1e6)}M"

if os.path.exists(f"{folder_out}/{NP}"):
    shutil.rmtree(f"{folder_out}/{NP}")
else:
    os.makedirs(f"{folder_out}/{NP}")

############ begin load data
# This part needs to be modified according to how the predictions of your model are stored.
print('Load data')
features, target = features, target = load_data(target_dir,rewgt=rewgt_bkgs)

mask_SIG = np.zeros_like(target)
mask_BKG = np.zeros_like(target)
for sig_label in sig_labels:
    mask_SIG += 1*(target==sig_label)
for bkg_label in bkg_labels:
    mask_BKG += 1*(target==bkg_label)

features_SIG = features[mask_SIG>0]
features_BKG = features[mask_BKG>0]
############ end load data

######## standardizes data
print('standardize')
features_mean, features_std = np.mean(features_BKG, axis=0), np.std(features_BKG, axis=0)
print('mean: ', features_mean)
print('std: ', features_std)
features_BKG = standardize(features_BKG, features_mean, features_std)
features_SIG = standardize(features_SIG, features_mean, features_std)

#### compute sigma hyper parameter from data
#### (This doesn't need modifications)
flk_sigma = candidate_sigma(features_BKG[:2000, :], perc=flk_sigma_perc)
print('flk_sigma', flk_sigma)

## run toys
print('Start running toys')
ts=np.array([])
seeds = np.arange(Ntoys)+datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
for i in tqdm(range(Ntoys)):
    seed = seeds[i]
    rng = np.random.default_rng(seed=seed)
    N_bkg_p = rng.poisson(lam=N_bkg, size=1)[0]
    N_sig_p = rng.poisson(lam=N_sig, size=1)[0]
    rng.shuffle(features_SIG)
    rng.shuffle(features_BKG)
    features_s = features_SIG[:N_sig_p, :]
    features_b = features_BKG[:N_bkg_p+N_ref, :]
    features  = np.concatenate((features_s,features_b), axis=0)

    label_R = np.zeros((N_ref,))
    label_D = np.ones((N_bkg_p+N_sig_p, ))
    labels  = np.concatenate((label_D,label_R), axis=0).reshape(-1,1)
    
    plot_reco=False
    verbose=False
    # make reconstruction plots every 20 toys (can be changed)
    #if not i%20:
    #    plot_reco=True
    #    verbose=True
    flk_config = get_logflk_config(M,flk_sigma,[lam],weight=w_ref,iter=[iterations],seed=None,cpu=False)
    test_label = f"d{dim}_t{temp}"
    t, pred = run_toy(test_label, features, labels, weight=w_ref, seed=seed,
                      flk_config=flk_config, output_path=f'{folder_out}/{NP}/', plot=plot_reco, savefig=plot_reco,
                      verbose=verbose)
    
    ts = np.append(ts, t)

# collect previous toys if existing
seeds_past = np.array([])
ts_past = np.array([])
if os.path.exists('%s/%s/tvalues_flksigma%s.h5'%(folder_out, NP, flk_sigma)):
    print('collecting previous tvalues')
    f = h5py.File('%s/%s/tvalues_flksigma%s.h5'%(folder_out, NP, flk_sigma), 'r')
    seeds_past = np.array(f.get('seed_toy'))
    ts_past = np.array(f.get(str(flk_sigma) ) )
    f.close()
ts = np.append(ts_past, ts)
seeds = np.append(seeds_past, seeds)

f = h5py.File('%s/%s/tvalues_flksigma%s.h5'%(folder_out, NP, flk_sigma), 'w')
f.create_dataset(str(flk_sigma), data=ts, compression='gzip')
f.create_dataset('seed_toy', data=seeds, compression='gzip')
f.close()
