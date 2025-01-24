# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
python rf.py --base-path ${1} --model-type regressor --emb ${2}-70to80 --split-file split-cls.csv --split-col cell-line --n-jobs ${3} --fold 0
python rf.py --base-path ${1} --model-type regressor --emb ${2}-70to80 --split-file split-cls.csv --split-col cell-line --n-jobs ${3} --fold 1
python rf.py --base-path ${1} --model-type regressor --emb ${2}-70to80 --split-file split-cls.csv --split-col cell-line --n-jobs ${3} --fold 2
python rf.py --base-path ${1} --model-type regressor --emb ${2}-70to80 --split-file split-cls.csv --split-col cell-line --n-jobs ${3} --fold 3
python rf.py --base-path ${1} --model-type regressor --emb ${2}-70to80 --split-file split-cls.csv --split-col cell-line --n-jobs ${3} --fold 4

python rf.py --base-path ${1} --model-type regressor --emb ${2}-80to90 --split-file split-cls.csv --split-col cell-line --n-jobs ${3} --fold 0
python rf.py --base-path ${1} --model-type regressor --emb ${2}-80to90 --split-file split-cls.csv --split-col cell-line --n-jobs ${3} --fold 1
python rf.py --base-path ${1} --model-type regressor --emb ${2}-80to90 --split-file split-cls.csv --split-col cell-line --n-jobs ${3} --fold 2
python rf.py --base-path ${1} --model-type regressor --emb ${2}-80to90 --split-file split-cls.csv --split-col cell-line --n-jobs ${3} --fold 3
python rf.py --base-path ${1} --model-type regressor --emb ${2}-80to90 --split-file split-cls.csv --split-col cell-line --n-jobs ${3} --fold 4

python rf.py --base-path ${1} --model-type regressor --emb ${2}-90to100 --split-file split-cls.csv --split-col cell-line --n-jobs ${3} --fold 0
python rf.py --base-path ${1} --model-type regressor --emb ${2}-90to100 --split-file split-cls.csv --split-col cell-line --n-jobs ${3} --fold 1
python rf.py --base-path ${1} --model-type regressor --emb ${2}-90to100 --split-file split-cls.csv --split-col cell-line --n-jobs ${3} --fold 2
python rf.py --base-path ${1} --model-type regressor --emb ${2}-90to100 --split-file split-cls.csv --split-col cell-line --n-jobs ${3} --fold 3
python rf.py --base-path ${1} --model-type regressor --emb ${2}-90to100 --split-file split-cls.csv --split-col cell-line --n-jobs ${3} --fold 4