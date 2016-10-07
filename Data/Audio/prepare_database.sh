#!/bin/bash
#set -o xtrace


STORE=/nfs/pyrex/raid6/aroyer/Ester2_audio/Ester2
SAMPLES=$STORE/Samples
DATA=/udd/aroyer/Stage/Code/Data/Audio/ESTER2
WORDSTOKEEP=/udd/aroyer/Stage/Code/Data/Audio/ESTER2/Stats/words_to_keep.txt
SCRIPTS=/udd/aroyer/Stage/Code/Data/Audio/Scripts
MINLENGTH=0.2
HOMONYMS=/udd/aroyer/Stage/Code/Data/Audio/ESTER2/Stats/words_homonyms.txt

#Extract samples
echo "Extracting Samples"
cd $STORE
rm -r $SAMPLES
mkdir $SAMPLES
mkdir $SAMPLES/Discarded
perl chop.pl $STORE/word-align.ester2-dev.mlf.utf8 > output_chop.log

#Remove too short samples
echo "Removing too short samples"
python $SCRIPTS/check_segment_length.py -i $SAMPLES -l $MINLENGTH -ho $HOMONYMS > $DATA/Stats/Ester2_segment_lengthes.txt

#Generate index to labels and ground_truthes
echo "Generating ground truth and index_to_label files"
python $SCRIPTS/prepare.py $STORE -o $DATA > $DATA/Stats/summary_ground_truth_unique.txt
python $SCRIPTS/prepare.py $STORE -ho $HOMONYMS -o $DATA > $DATA/Stats/summary_ground_truth_homonyms.txt

#Generate qrel
python ~/Stage/Code/src/utils/parse.py
