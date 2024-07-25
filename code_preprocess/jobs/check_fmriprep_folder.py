#!/usr/bin/env python

'''
get the list of subject who had incomplete fmriprep output
source_dir: (str)
    fmriprep directory
    str(sys.argv[1])
output:
print out a list of subject folder names for the bash script

'''

# Yiyu Wang 2023/12/
import os
import pandas as pd
import sys
import glob
# main

if len(sys.argv) == 4:
    which_subjects = str(sys.argv[1])
    source_dir = str(sys.argv[2])
    target_dir = str(sys.argv[3])
else:
    raise Exception("too many arguments are being passed to python")


# functions
def print_for_bash(sub_list):
    # function to print subject ids that can be copied pasted to bash script
   
    # Join the subject IDs with spaces and quotation marks and print
    print_string = ' '.join(['"' + sub + '"' for sub in sub_list])
    return print_string

def list_difference(source_list, target_list):
    # Using list comprehension to find elements in source_list not in target_list
    difference = [item for item in source_list if item not in target_list]
    return difference

def extract_sub_num(source_id):
    import re
    # Use regular expression to extract the number part
    match = re.search(r'(bio|Bio|BIO)(\d+)', source_id)
    if match:
        # Extract the number part and convert it to integer
        sub_num = int(match.group(2))
    else:
        #print(f"{source_id} does not follow id format")
        sub_num = None

    return sub_num


source_list = os.listdir(source_dir)

sub_list = []  
for sub_folder in source_list:
    sub_id = os.path.basename(sub_folder)
    sub_list.append(sub_id)



print(print_for_bash(sub_list))