#!/bin/sh
#OAR -n AUDIO_BIN_2000
#OAR -l nodes=1,walltime=100:00:00 
#OAR -p mem_node > 45*1024
#OAR -O /Logs/OARjob.%jobid%.output
#OAR -E /Logs/OARjob.%jobid%.error


#### FOLDERS
MAINDIR=/temp_dd/$USER/
SCRATCHDIR=/temp_dd/$USER/Scratch
OUTPUTDIR=/temp_dd/$USER/Outputs/Outputs_$OAR_JOB_ID
TEMPDIR=/temp_dd/$USER/Temp/Temp_$OAR_JOB_ID


#### COMMAND LINE PARAMETERS
DATA=AUDIO
ITER=2000
NMIN=100
NMAX=200
SIM=BIN
VERBOSE=4
PROC=7
DIST=RND
TS=0.4
CLAS=HTK

RAM=$(awk -F":" '$1~/MemTotal/{print $2}' /proc/meminfo)
CORES=$(grep -c ^processor /proc/cpuinfo)


#### Info
if [ -n "$OAR_NODE_FILE" ] 
then 
    echo "[OAR] OAR_JOB_ID=$OAR_JOB_ID"
    echo "[OAR] Nodes:" 
    sort $OAR_NODE_FILE | uniq -c | awk '{printf(" %s*%d\n",$2,$1)}END{printf("\n")}' | sed -e 's/,$//' 
fi

echo "
##########################################################################
#  STARTING EXPERIMENTS
#
#  Memory: ${RAM}
#  CPUs: ${CORES}
#  
#  Output folder: ${MAINDIR}
#  Current parameters:
#   * ${ITER} iterations
#   * ${SIM} similarity
#   * ${TS} training size
#   * ${NMIN} to ${NMAX} synthetic labels
#   * ${DIST} annotation
#   * Verbosity: ${VERBOSE}
#
#  NOTE: Binary (Wapiti, MCL etc) must first be recompiled through an interactive job on the cluster
#
##########################################################################
"

RUNDIR=$SCRATCHDIR/$OAR_JOB_ID
mkdir -p $RUNDIR
cd $RUNDIR

EXECUTABLE=$HOME/Stage/Code/src/main.py

echo "Working directory :"
pwd

echo
echo "=============== RUN ==============="
echo

python $EXECUTABLE -N $ITER -t $PROC -d $DATA -p MCL -o $OUTPUTDIR -te $TEMPDIR/ -di $DIST -nmin $NMIN -nmax $NMAX -s $SIM -ts $TS -c $CLAS -v $VERBOSE --oar

echo
echo "============ DONE =================="

echo "============ CLEANING =================="
rm Logs/OARjob.$OAR_JOB_ID.output
#rm /temp_dd/igrida-fs1/aroyer/Logs/OARjob.$OAR_JOB_ID.error
rm -r $RUNDIR
rm -r $TEMPDIR
