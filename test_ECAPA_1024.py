import sys, time, os, argparse, socket, torch, glob, zipfile, datetime
from tuneThreshold import *
from SpeakerNet import *
from DatasetLoader import *
import torch.distributed as dist
import torch.multiprocessing as mp

parser = argparse.ArgumentParser(description = "SpeakerNet")
## Data loader
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training')
parser.add_argument('--eval_frames',    type=int,   default=200,      help='Input length to the network for testing 0 uses the whole files')
parser.add_argument('--num_eval',       type=int,   default=10,      help='Input length to the network for testing 0 uses the whole files')
parser.add_argument('--num_spk',        type=int,   default=100,    help='Number of speakers per batch, i.e., batch size = num_spk * num_utt')
parser.add_argument('--num_utt',        type=int,   default=2,      help='Number of utterances per speaker in batch')
parser.add_argument('--max_seg_per_spk',type=int,   default=600,    help='Maximum number of utterances per speaker per epoch')
parser.add_argument('--num_thread',     type=int,   default=10,      help='Number of loader threads')
parser.add_argument('--augment',        type=bool,  default=True,   help='Augment input')
parser.add_argument('--seed',           type=int,   default=10,     help='Seed for the random number generator')

## Training details
parser.add_argument('--test_interval',  type=int,   default=1,      help='Test and save every [test_interval] epochs')
parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum number of epochs')
parser.add_argument('--trainfunc',      type=str,   default="angleproto", help='Loss function')
parser.add_argument('--dur_masking',    dest='dur_masking', action='store_true', help='Masking for duration invariant training')
parser.add_argument('--dur_grouping',   dest='dur_grouping', action='store_true', help='Grouping for duration invariant training')

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam')
parser.add_argument('--scheduler',      type=str,   default="cosine_annealing_warmup_restarts", help='Learning rate scheduler')
parser.add_argument('--weight_decay',   type=float, default=2e-5,   help='Weight decay in the optimizer')
parser.add_argument('--lr',             type=float, default=0.001,  help='StepLR sched: Learning rate')
parser.add_argument("--lr_decay",       type=float, default=0.95,   help='StepLR sched: Learning rate decay every epoch')
parser.add_argument('--lr_t0',          type=int,   default=10,     help='Cosine sched: First cycle step size')
parser.add_argument('--lr_tmul',        type=float, default=1.0,    help='Cosine sched: Cycle steps magnification.')
parser.add_argument('--lr_max',         type=float, default=1e-3,   help='Cosine sched: First cycle max learning rate')
parser.add_argument('--lr_min',         type=float, default=0,      help='Cosine sched: First cycle min learning rate')
parser.add_argument('--lr_wstep',       type=int,   default=0,      help='Cosine sched: Linear warmup step size')
parser.add_argument('--lr_gamma',       type=float, default=0.9,    help='Cosine sched: Decrease rate of max learning rate by cycle')


## Loss functions
parser.add_argument('--margin',         type=float, default=0.2,    help='Loss margin, only for some loss functions')
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions')
parser.add_argument('--num_class',      type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses')
parser.add_argument('--num_pooling',    type=int,   default=5,      help='Number of speakers in the softmax layer, only for softmax-based losses')
parser.add_argument('--w_cls',          type=float, default=1.0,    help='Weight for softmax-based loss')
parser.add_argument('--w_mtr',          type=float, default=1.0,    help='Weight for metric-based loss')

## Evaluation parameters
parser.add_argument('--dcf_p_target',   type=float, default=0.05,   help='A priori probability of the specified target speaker')
parser.add_argument('--dcf_c_miss',     type=float, default=1,      help='Cost of a missed detection')
parser.add_argument('--dcf_c_fa',       type=float, default=1,      help='Cost of a spurious detection')

## Load and save
parser.add_argument('--initial_model',   type=str,   default="",     help='Initial model weights')
parser.add_argument('--save_path',      type=str,   default="./save/test1", help='Path for model and logs')

## Training and test data
parser.add_argument('--train_list',     type=str,   default="./list/train_vox2.txt", help='Train list')
parser.add_argument('--test_list',      type=str,   default="./list/veri_test2.txt", help='Evaluation list')
parser.add_argument('--train_path',     type=str,   default="/home/mhhan/DB/VoxCeleb/VoxCeleb2/train/wav/",     help='Absolute path to the train set')
parser.add_argument('--test_path',      type=str,   default="/home/shmun/DB/VoxCeleb/VoxCeleb-all/wav/", help='Absolute path to the test set')
parser.add_argument('--musan_path',     type=str,   default="/home/shmun/DB/MUSAN/musan_split", help='Absolute path to the test set')
parser.add_argument('--rir_path',       type=str,   default="/home/shmun/DB/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set')


## Model definition
parser.add_argument('--num_mels',       type=int,   default=80,     help='Number of mel filterbanks')
parser.add_argument('--log_input',      type=bool,  default=True,   help='Log input features')
parser.add_argument('--model',          type=str,   default="ECAPA_TDNN", help='Name of model definition')
parser.add_argument('--pooling_type',   type=str,   default="ASP",  help='Type of encoder')
parser.add_argument('--num_out',        type=int,   default=192,    help='Embedding size in the last FC layer')
parser.add_argument('--eca_c',          type=int,   default=1024,   help='ECAPA-TDNN channel')
parser.add_argument('--eca_s',          type=int,   default=8,      help='ECAPA-TDNN model-scale')

## Evaluation types
parser.add_argument('--eval',           dest='eval', action='store_true', help='Eval only')
#parser.add_argument('--eval_dur',       type=bool,  default=True,   help='4type evaluation')
parser.add_argument('--eval_dur',       type=str,   default='4type', help='[4type or 1type] evaluation')
parser.add_argument('--score_norm',     dest='score_norm', action='store_true', help='Score normalization')
parser.add_argument('--type_coh',       type=str,   default='utt',  help='Cohort type - select: spk or utt')
parser.add_argument('--top_coh_size',   type=int,   default=20000,  help='Maximum cohort size for adaptive s-norm')
parser.add_argument('--tta',            dest='tta', action='store_true', help='Test Time Augmentation')

## Distributed and mixed precision training
parser.add_argument('--port',           type=str,   default="8000", help='Port for distributed training, input as text')
parser.add_argument('--distributed',    dest='distributed', action='store_true', help='Enable distributed training')
#parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')

args = parser.parse_args("")
args.gpu = 0

s = SpeakerNet(**vars(args))
s = WrappedModel(s).cuda(args.gpu)
trainer = ModelTrainer(s, **vars(args))
#=======================================================================#
'''
trainer.loadParameters('./save/angleproto/model/model000000433.model')
args.test_list = './list/list_test_all2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Extended")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

args.test_list = './list/list_test_hard2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Hard")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

trainer.loadParameters('./save/angleproto/model/model000000498.model')
args.test_list = './list/list_test_all2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Extended")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

args.test_list = './list/list_test_hard2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))
trainer.loadParameters('./save/angleproto/model/model000000326.model')

args.test_list = './list/list_test_all2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Extended")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

args.test_list = './list/list_test_hard2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Hard")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))


trainer.loadParameters('./save/angleproto/model/model000000460.model')
args.test_list = './list/list_test_all2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Extended")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

args.test_list = './list/list_test_hard2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Hard")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

#========================================================================#
trainer.loadParameters('./save/newloss/model/model000000384.model')
args.test_list = './list/list_test_all2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Extended")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

args.test_list = './list/list_test_hard2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Hard")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

trainer.loadParameters('./save/newloss/model/model000000446.model')
args.test_list = './list/list_test_all2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Extended")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

args.test_list = './list/list_test_hard2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Hard")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

trainer.loadParameters('./save/newloss/model/model000000402.model')
args.test_list = './list/list_test_all2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Extended")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

args.test_list = './list/list_test_hard2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Hard")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))
'''
trainer.loadParameters('./save/newloss/model/model000000471.model')
args.test_list = './list/list_test_all2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Extended")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

args.test_list = './list/list_test_hard2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Hard")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

#=============================================================================

trainer.loadParameters('./save/normsoftmax/model/model000000243.model')
args.test_list = './list/list_test_all2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Extended")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

args.test_list = './list/list_test_hard2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Hard")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

trainer.loadParameters('./save/normsoftmax/model/model000000412.model')
args.test_list = './list/list_test_all2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Extended")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

args.test_list = './list/list_test_hard2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Hard")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

trainer.loadParameters('./save/normsoftmax/model/model000000429.model')
args.test_list = './list/list_test_all2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Extended")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

args.test_list = './list/list_test_hard2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Hard")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))


trainer.loadParameters('./save/normsoftmax/model/model000000457.model')
args.test_list = './list/list_test_all2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Extended")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

args.test_list = './list/list_test_hard2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Hard")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

#+=============================================================================================

trainer.loadParameters('./save/han_softmax/model/model000000078.model')
args.test_list = './list/list_test_all2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Extended")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

args.test_list = './list/list_test_hard2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Hard")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

trainer.loadParameters('./save/han_softmax/model/model000000108.model')
args.test_list = './list/list_test_all2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Extended")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

args.test_list = './list/list_test_hard2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Hard")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

trainer.loadParameters('./save/han_softmax/model/model000000246.model')
args.test_list = './list/list_test_all2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Extended")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

args.test_list = './list/list_test_hard2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Hard")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))


trainer.loadParameters('./save/han_softmax/model/model000000248.model')
args.test_list = './list/list_test_all2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Extended")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

args.test_list = './list/list_test_hard2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Hard")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

trainer.loadParameters('./save/han_softmax/model/model000000258.model')
args.test_list = './list/list_test_all2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Extended")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))

args.test_list = './list/list_test_hard2.txt'
sc1, lab = trainer.evaluateFromList_H(**vars(args))
eer1 = tuneThresholdfromScore(sc1, lab, [1, 0.1])[1]
fnrs, fprs, thresholds = ComputeErrorRates(sc1, lab)
dcf1 = ComputeMinDcf(fnrs, fprs, thresholds, args.dcf_p_target, args.dcf_c_miss, args.dcf_c_fa)[0]
print("Hard")
print('\n',time.strftime("%Y-%m-%d %H:%M:%S"), "EER {:2.4f}, MinDCF {:2.5f}".format(eer1, dcf1))
