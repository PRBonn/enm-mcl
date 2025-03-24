#!/bin/bash

configs=("loc_config_test1.yaml" "loc_config_test2.yaml" "loc_config_test3.yaml" "loc_config_test4.yaml" "loc_config_test5.yaml")
suffixes=(1 2 3 4 5)
seeds=(52000 5200 22 12 2)

for config in "${configs[@]}"; do
  for i in "${!suffixes[@]}"; do
    suffix=${suffixes[$i]}
    seed=${seeds[$i]}
    python run_localization.py --config_file ./configs/global_localization/$config --suffix $suffix --seed $seed
  done
done

results_dir=./results/ipblab/loc_test
python eval_localization.py --results_dir $results_dir
