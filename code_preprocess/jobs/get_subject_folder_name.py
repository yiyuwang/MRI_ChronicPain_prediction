#!/usr/bin/env python

'''
get the list of subject level folder names
which_subjects (str, 'all', 'new', 'bioXXX'):
    what list  of subjects to extract
    str(sys.argv[1])
source_dir: (str)
    a directory where the input is 
    str(sys.argv[2])
target_dir (str): 
    a directory where the output will be
    str(sys.argv[3])
output:
print out a list of subject folder names for the bash script

'''

# Yiyu Wang 2023/12/
import os
import sys
import glob
import numpy as np
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
    
    difference = [item for item in source_list if format_sub_id(item) not in target_list]
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

def format_sub_id(source_id):
    # Make it a string
    source_id = str(source_id)
    sub_num = extract_sub_num(source_id)
    if sub_num:
        formatted_id = "bio" + "{:04}".format(sub_num)
    else:
        formatted_id = ""    
    return formatted_id


source_list = glob.glob(f'{source_dir}/*/')
if which_subjects == 'all': 
    sub_list = []  
    for sub_folder in source_list:
        sub_id = extract_sub_num(os.path.basename(os.path.dirname(sub_folder)))
        if format_sub_id(sub_id):
            sub_list.append(os.path.basename(os.path.dirname(sub_folder)))

elif which_subjects == 'new':
    source_id_list = []
    for sub_folder in source_list:
        source_id_list.append(os.path.basename(os.path.dirname(sub_folder)))

    # get the sub_id for target directory  
    target_id_list = []
    target_list = glob.glob(f'{target_dir}/*/')
    for sub_folder in target_list:
        target_id_list.append(format_sub_id(os.path.basename(os.path.dirname(sub_folder))))

    # get the difference between existing and the target folder list:
    diff_list = list_difference(source_id_list, target_id_list)
    
    # find the correponding file name in the source_list (because we try to process new folders in source dir)
    sub_list = []
    for sub_id in diff_list:
        sub_file_path = np.sort(glob.glob(f"{source_dir}/*{sub_id}*")).tolist()
        if len(sub_file_path) == 0:
            continue
            # print(f"cannot file for {sub_id} in the {source_dir}, check")
        elif len(sub_file_path) == 1:
            sub_list.append(os.path.basename(sub_file_path[0]))
        else:
            sub_list.append(os.path.basename(sub_file_path[0]))
            # print(f"more than 1 file for {sub_id} in the {source_dir}. Will process the first one")
        

elif "bio" in which_subjects:
    # extract the four digit in the id:
    sub_num = extract_sub_num(which_subjects)
    # find the id in the source list
    sub_folder_list = glob.glob(f'{source_dir}/*{sub_num}*/')
    if len(sub_folder_list) == 1:
        sub_list =[os.path.basename(os.path.dirname(sub_folder_list[0]))]
    elif len(sub_folder_list) > 1:
        raise Exception("more than one folder associated with this subject number!")
    elif len(sub_folder_list) == 0:
        raise Exception("no folder associated with this subject number!")
    else:
        raise Exception("unknow issue with the subject number! check")

else:
    raise Exception("no this option to extract subject ids")


print(print_for_bash(sub_list))