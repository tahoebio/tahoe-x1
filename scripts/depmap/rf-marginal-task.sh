# Copyright (C) Vevo Therapeutics 2024. All rights reserved.
python rf.py --base-path ${1} --model-type classifier --emb ${2} --split-file split-genes-lt5gt70.csv --split-col gene --n-jobs ${3} --fold 0
python rf.py --base-path ${1} --model-type classifier --emb ${2} --split-file split-genes-lt5gt70.csv --split-col gene --n-jobs ${3} --fold 1
python rf.py --base-path ${1} --model-type classifier --emb ${2} --split-file split-genes-lt5gt70.csv --split-col gene --n-jobs ${3} --fold 2
python rf.py --base-path ${1} --model-type classifier --emb ${2} --split-file split-genes-lt5gt70.csv --split-col gene --n-jobs ${3} --fold 3
python rf.py --base-path ${1} --model-type classifier --emb ${2} --split-file split-genes-lt5gt70.csv --split-col gene --n-jobs ${3} --fold 4