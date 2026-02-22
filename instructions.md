Update code in files:
- static_vectors
- contextual_vectors
- wsd_inference
- analysis


Installs:
- pip install gensim
- pip install torch
- pip install transformers

Running the Code:
1.  Commands to generate global sense representations and then do WSD inferences.
- /mnt/dropbox/25-26/571F/envs/bin/python generate_sense_representations.py --semcor_glob "./data/train/*" --output_file train_table_global.txt global
- /mnt/dropbox/25-26/571F/envs/bin/python wsd_inference.py --sense_table_path train_table_global.txt --semcor_glob "./data/test/*" --output_file inferences_global.csv global
- With MFS fall-back: 
/mnt/dropbox/25-26/571F/envs/bin/python wsd_inference.py --sense_table_path train_table_global.txt --semcor_glob "./data/test/*" --output_file inferences_global_mfs-fallback.csv --mfs_fallback global

