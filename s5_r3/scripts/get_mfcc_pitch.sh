# UNI: jjl2245, me2680
#!/bin/bash
. ./path.sh || exit 1
. ./cmd.sh || exit 1
nj=1       # number of parallel jobs - 1 is perfect for such a small dataset
lm_order=1 # language model order (n-gram quantity) - 1 is enough for digits grammar

#!/bin/bash
echo "Script executed from: ${PWD}"

for VARIABLE in "BN" "EN" "ES" "FR" "JV" "KR" "MD" "RS" "TR"
do
    echo 
    echo
    echo
    echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo $VARIABLE
    lang=$VARIABLE
    mfccdir=../features/mfcc_pitch/$lang #mfccdir is the directory to which the data is saved

    # Safety mechanism (possible running this script with modified arguments)
    . utils/parse_options.sh || exit 1
    [[ $# -ge 1 ]] && { echo "Wrong arguments!"; exit 1; }
    # Removing previously created data
    rm -rf exp ../data/$lang/train/spk2utt ../data/$lang/train/cmvn.scp ../data/$lang/train/feats.scp ../data/$lang/train/split1 ../data/$lang/test/spk2utt ../data/$lang/test/cmvn.scp ../data/$lang/test/feats.scp ../data/$lang/test/split1 ../data/$lang/local/lang ../data/$lang/lang ../data/$lang/local/tmp ../data/$lang/local/dict/lexiconp.txt

    echo "===== PREPARING ACOUSTIC DATA ====="
    # Create the spk2utt files from utt2spk files
    utils/utt2spk_to_spk2utt.pl ../data/$lang/train/utt2spk > ../data/$lang/train/spk2utt
    utils/utt2spk_to_spk2utt.pl ../data/$lang/test/utt2spk > ../data/$lang/test/spk2utt

    echo "===== FEATURES EXTRACTION ====="
    # Run make_mfcc.sh on the train and test folders
    steps/make_mfcc_pitch.sh --nj $nj --cmd "$train_cmd" ../data/$lang/train ../features/logs/make_mfcc_pitch/train $mfccdir  # run mfcc.sh or mfcc_pitch.sh
    steps/make_mfcc_pitch.sh --nj $nj --cmd "$train_cmd" ../data/$lang/test ../features/logs/make_mfcc_pitch/train $mfccdir  # run mfcc.sh or mfcc_pitch.sh

    # echo
    # echo "===== CMVN STAT CREATION ====="
    # # Making cmvn.scp files
    # steps/compute_cmvn_stats.sh ../data/$lang/train exp/make_mfcc/train $mfccdir
    # steps/compute_cmvn_stats.sh ../data/$lang/test exp/make_mfcc/test $mfccdir
done


echo
echo "===== Script is finished ====="
echo
