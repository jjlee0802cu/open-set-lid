# UNI: jjl2245, me2680

# Defining Kaldi root directory
export KALDI_ROOT=`pwd`/../../../../
# Setting paths to useful tools
##export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$PWD:$PATH
# Defining audio data directory (modify it for your installation directory!)
#export DATA_ROOT="/home/{user}/kaldi/egs/digits/digits_audio"
# Enable SRILM
#. $KALDI_ROOT/tools/env.sh
# Variable needed for proper data sorting
#export LC_ALL=C

[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export LD_LIBRARY_PATH="$KALDI_ROOT/src/lib" # use this one fof mfcc
#export LD_LIBRARY_PATH="$KALDI_ROOT/src/lib" #use this one for mfcc_pitch 
export LD_LIBRARY_PATH="$KALDI_ROOT/tools/openfst/lib" #use this one for mfcc_pitch 
