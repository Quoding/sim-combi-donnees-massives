#!/bin/bash
for i in {0..49}
do
#    python create_dataset.py --config configs/config_100.json --seed $i
#    python create_dataset.py --config configs/config_50.json --seed $i
#    python create_dataset.py --config configs/config_1000.json --seed $i
    # python create_dataset.py --config configs/config_500.json --seed $i
    python create_dataset.py --config configs/config_500_skew0.json --seed $i

done
