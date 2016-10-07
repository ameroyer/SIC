#!/bin/bash
#set -o xtrace

DATA='20samples(50)'
SCRIPT='Scripts/run_hmm.py'
EVALSCRIPT='Scripts/evaluate_hmm.py'
FEATURES="MFCC_0_D_A_Z"
OUTPUT="result"
TS=0.8
NCEPS=12

python $SCRIPT -ts $TS -o $OUTPUT -d $DATA -f $FEATURES -c $NCEPS
#cat $OUTPUT
python $EVALSCRIPT -r $OUTPUT -d $DATA
