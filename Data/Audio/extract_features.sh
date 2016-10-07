#!/bin/bash
#set -o xtrace

INPUT='/nfs/pyrex/raid6/aroyer/Ester2_audio/Ester2/Samples'
OUTPUT='/udd/aroyer/Stage/Code/Data/Audio/ESTER2/Features'
SCRIPT='Scripts/extract_features.py'
NCEPS=12

# 1st set: MF Cepstral Coefficient
# Features Options
features=("MFCC")
qualif=("" "_0" "_E" "_0_E" "_Z" "_0_Z" "_0_E_Z" "_E_Z" "_D" "_E_D" "_0_E_Z" "_D_Z" "_E_D_Z" "_0_D_Z" "_D_A" "_E_D_A" "_D_A_Z" "_E_D_A_Z" "_0_D" "_0_D_A" "_0_D_A_Z" "_0_D_A_T" "_E_D_A_T" "_E_D_A_T_Z" "_0_D_A_T_Z")

# Extract all combinations
for f in "${features[@]}"   
do   
    for q in "${qualif[@]}" 
    do
	echo 'Extracting ' $f$q
	python $SCRIPT -i $INPUT -o $OUTPUT -f $f$q -c $NCEPS
    done
done
exit
# 2st set: Linear Prediction coefficients
features=("LPC" "LPREFC" "LPCEPSTRA" "PLP")
qualif=("" "_E" "_Z" "_E_Z" "_D" "_D_Z" "_E_D_Z" "_D_A" "_D_A_Z" "_E_D" "_E_D_A" "_E_D_A_Z" "_E_D_A_T" "_D_A_T" "_D_A_T_Z")

for f in "${features[@]}"   
do   
    for q in "${qualif[@]}" 
    do
	echo 'Extracting ' $f$q
	python $SCRIPT -i $INPUT -o $OUTPUT -f $f$q -c $NCEPS
    done
done
