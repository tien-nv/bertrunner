#!/usr/bin/env python3

import os
from pyrouge import Rouge155

def remove_broken_files():
    error_id = []
    for f in os.listdir('outputs/ref'):
        try:
            open('outputs/ref/' + f).read()
        except:
            error_id.append(f)
    for f in os.listdir('outputs/hyp'):
        try:
            open('outputs/hyp/' + f).read()
        except:
            error_id.append(f)
    error_set = set(error_id)
    for f in error_set:
        os.remove('outputs/ref/' + f)
        os.remove('outputs/hyp/' + f)

def rouge():
    r = Rouge155()
    r.home_dir = '.'
    r.system_dir = 'outputs/hyp'
    r.model_dir =  'outputs/ref'

    r.system_filename_pattern = '(\d+).txt'
    r.model_filename_pattern = '#ID#.txt'
    
    command_75 = '-e /home/vbee/tiennv/basicsum/pyrouge/tools/ROUGE-1.5.5/data -a -c 95 -m -n 2 -b 75'
    output = r.convert_and_evaluate(rouge_args=command_75)
    #output = r.convert_and_evaluate()
    # print(output)
    print("working")
    # output_dict_75 = r.output_to_dict(output_75)
    # output_dict_275 = r.output_to_dict(output_275)
    output_dict = r.output_to_dict(output)
    # print(output_dict_75)
    # print(output_dict_275)
    print(output_dict)

if __name__ == '__main__':
    remove_broken_files()
    rouge()
