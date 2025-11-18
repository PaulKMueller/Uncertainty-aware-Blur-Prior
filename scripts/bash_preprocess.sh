#!/bin/bash

# EEG
for subject in {1..10}
do
  python preprocess/process_eeg_whiten.py --subject $subject
done

## MEG
# for subject in {1..4}
# do
#   python preprocess/process_meg.py --subject $subject
# done
