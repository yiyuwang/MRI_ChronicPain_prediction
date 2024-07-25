#!/usr/bin/env python
import os
import pandas as pd
import sys
import glob

if len(sys.argv) == 3:
    subject = str(sys.argv[1])
    BIDS_dir = str(sys.argv[2])
else:
    raise Exception("too many arguments are being passed to python")

tsv_filepath= f'{BIDS_dir}/participants.tsv'

df = pd.DataFrame(columns=['participant_id'])

folder_list = glob.glob(f'{BIDS_dir}/sub*/')
sub_list = []
for sub_folder in folder_list:
    sub_id = os.path.basename(sub_folder)
    sub_list.append(sub_id)
    
sub_list.sort()
df_extended = pd.DataFrame(sub_list, columns=df.columns)

# concatenate to original
out = pd.concat([df, df_extended])

# Save as a .tsv file
out.to_csv(tsv_filepath, sep='\t', index=False)