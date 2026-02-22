#!/bin/bash

#  Commands to generate global sense representations and then do WSD inferences.
python generate_sense_representations.py --semcor_glob "./data/train/*" --output_file train_table_global.txt global
# python wsd_inference.py --sense_table_path train_table_global.txt --semcor_glob "./data/test/*" --output_file inferences_global.csv global
# python wsd_inference.py --sense_table_path train_table_global.txt --semcor_glob "./data/test/*" --output_file inferences_global_mfs-fallback.csv --mfs_fallback global