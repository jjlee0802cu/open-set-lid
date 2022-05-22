# Modernizing Open-Set Speech Language Identification

#### Justin Lee, Mustafa Eyceoz

## Project Summary:

Speech Language Identification is the process of taking audio as an input and determining what language is being spoken, if any. There are two subsections to the language identification  problem (which will henceforth be referred to as LID): open-set and closed-set. In closed-set LID, set of languages to identify is defined, and for every audio input, the "most probable" language  within the set is outputted. In open-set LID, however, we also gain the option to "reject" that prediction and detect when the audio input matches none of our known languages well. While  most modern speech Language Identification methods are closed-set, we want to see if they can be modified and adapted for the open-set problem. Today, there are a number of modern-day state-of-the-art approaches to language identification, but almost all of them have opted to take the closed-set approach. In an era of data abundance, the limitations of the closed-set solution are typically circumvented by including hundreds of languages and training on thousands of hours of data for each of them. This workaround is obviously still not as ideal as the true open-set solution, though, as it lacks the ability to detect and reject or learn unknown languages, and in these cases it will unavoidably output an incorrect prediction. 

We tackle the open-set task by adapting two modern-day state-of-the-art approaches to closed-set language identification: the first using a CRNN with attention and the second using a TDNN. We enhanced our input feature embeddings using MFCCs, log spectral features, and pitch; and also adapted the aforementioned models to the open-set language identification problem with a threshold function. This threshold is used so that if all of the probabilities outputted by the softmax layer are under this threshold, the input is deemed out of the set and is rejected. 

## Tools:

- Python3 environment
- Python libraries: 
    - Kaldiio https://github.com/nttcslab-sp/kaldiio
    - PyTorch https://pytorch.org/
    - Numpy https://numpy.org/

## List of directories and executables that may be used to test the code.
- kaldi/egs/open_set_lid/s5_r3/scripts/scripts_model/open_set_demo.py 
    is the main decoder demo script that can be used to run everything.
- kaldi/egs/open_set_lid/s5_r3/scripts/ contains all other miscellaneous scripts
    for data formatting, data preparation, feature embeddings, model training.


## Usage
- The main script that can be used to run everything
    - kaldi/egs/open_set_lid/s5_r3/scripts/scripts_model/open_set_demo.py 
    - How to run: 
        python3 open_set_demo.py
    - Description: 
        This is the main decoder demo script. Selecting 300 random audio files from
        our test dataset (both in-set and out-of-set), this script will get the 
        feature embeddings and run them through our best trained model (TDNN) with 
        optimal threshold. For each test file, the script will print the true label, 
        as well as the label predicted by our open-set LID architecture. Upon 
        termination, the script will print in-set, out-of-set, and overall accuracies
        for the random test session.
- Important scripts for the process 
    - kaldi/egs/open_set_lid/s5_r3/scripts/scripts_data/ 
        for data preparation
    - kaldi/egs/open_set_lid/s5_r3/scripts/scripts_model/tdnn.py and kaldi/egs/open_set_lid/s5_r3/scripts/scripts_model/crnn_att.py
        for defining and training our rival models from scratch
    - kaldi/egs/open_set_lid/s5_r3/scripts/get_mfcc_pitch.sh
        for obtaining bases for feature embeddings


## References to where the data was obtained, what opensource code is being used, etc. 
- All data was obtained from OpenSLR: http://openslr.org/resources.php
    - Bengali dataset: http://openslr.org/53/
    - English dataset: http://openslr.org/45/
    - Spanish dataset: http://openslr.org/108/
    - French dataset: http://openslr.org/108/
    - Javanese dataset: http://openslr.org/35/
    - Korean dataset: http://openslr.org/58/
    - Mandarin dataset: http://openslr.org/47/
    - Russian dataset: http://openslr.org/96/
    - Turkish dataset:  http://openslr.org/108/
- Opensource code being used
    - Basic PyTorch TDNN: https://github.com/cvqluu/TDNN
    - Basic PyTorch CRNN + attention: https://github.com/chenjun2hao/Attention_ocr.pytorch


## Which files we have touched
- Data Preparation and Feature Extraction Scripts
    - kaldi/egs/open_set_lid/s5_r3/scripts/scripts_data/ark_reader.py
    - kaldi/egs/open_set_lid/s5_r3/scripts/scripts_data/BN_formater.py
    - kaldi/egs/open_set_lid/s5_r3/scripts/scripts_data/check_all_wav.py
    - kaldi/egs/open_set_lid/s5_r3/scripts/scripts_data/corpus_maker.py
    - kaldi/egs/open_set_lid/s5_r3/scripts/scripts_data/EN_formater.py
    - kaldi/egs/open_set_lid/s5_r3/scripts/scripts_data/flac_to_wav.py
    - kaldi/egs/open_set_lid/s5_r3/scripts/scripts_data/frequency.py
    - kaldi/egs/open_set_lid/s5_r3/scripts/scripts_data/JV_formater.py
    - kaldi/egs/open_set_lid/s5_r3/scripts/scripts_data/KR_formater.py
    - kaldi/egs/open_set_lid/s5_r3/scripts/scripts_data/MD_formater.py
    - kaldi/egs/open_set_lid/s5_r3/scripts/scripts_data/phones.py
    - kaldi/egs/open_set_lid/s5_r3/scripts/scripts_data/RS_formater.py
    - kaldi/egs/open_set_lid/s5_r3/scripts/scripts_data/timer.py
    - kaldi/egs/open_set_lid/s5_r3/scripts/scripts_data/tt_splitter.py
    - kaldi/egs/open_set_lid/s5_r3/scripts/scripts_data/utt2spk_maker.py
    - kaldi/egs/open_set_lid/s5_r3/scripts/scripts_data/wav_scp_creator.py
    - kaldi/egs/open_set_lid/s5_r3/scripts/get_mfcc_pitch.sh
    - kaldi/egs/open_set_lid/s5_r3/scripts/get_mfcc.sh
    - kaldi/egs/open_set_lid/s5_r3/scripts/cmd.sh
    - kaldi/egs/open_set_lid/s5_r3/scripts/path.sh
    - kaldi/egs/open_set_lid/s5_r3/scripts/conf/decode.config
    - kaldi/egs/open_set_lid/s5_r3/scripts/conf/mfcc.conf
    - kaldi/egs/open_set_lid/s5_r3/scripts/conf/pitch.conf
- Model Training and Testing Scripts
    - kaldi/egs/open_set_lid/s5_r3/scripts/scripts_model/crnn_att.py
    - kaldi/egs/open_set_lid/s5_r3/scripts/scripts_model/tdnn.py
    - kaldi/egs/open_set_lid/s5_r3/scripts/scripts_model/open_set_demo.py
