#!/bin/bash

#  Commands to generate global sense representations and then do WSD inferences.
python3 generate_sense_representations.py --semcor_glob "./data/train/*" --output_file train_table_global.txt global
python3 wsd_inference.py --sense_table_path train_table_global.txt --semcor_glob "./data/test/*" --output_file inferences_global.csv global
python wsd_inference.py --sense_table_path train_table_global.txt --semcor_glob "./data/test/*" --output_file inferences_global_mfs-fallback.csv --mfs_fallback global

# Commands to generate contextual sense representations and then do WSD inferences.
python3 generate_sense_representations.py --semcor_glob "./data/train/*" --output_file train_table_contextual.txt contextual
python3 wsd_inference.py --sense_table_path train_table_contextual.txt --semcor_glob "./data/test/*" --output_file inferences_contextual.csv contextual
python3 wsd_inference.py --sense_table_path train_table_contextual.txt --semcor_glob "./data/test/*" --output_file inferences_contextual_mfs-fallback.csv --mfs_fallback contextual

# Commands to generate analysis
python3 analysis.py --inference_file inferences_global.csv
python3 analysis.py --inference_file inferences_global_mfs-fallback.csv
python3 analysis.py --inference_file inferences_contextual.csv
python3 analysis.py --inference_file inferences_contextual_mfs-fallback.csv


