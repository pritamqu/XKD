# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

import os

def my_paths(server=None, db=None):
    
    
    output_dir = '/path/to/OUTPUTS'
    data_dir = fetch_db(db)
    
    return output_dir, data_dir
        

def fetch_db(db):
    
    root="/mnt/PS6T/datasets/"
    
    paths = {
        'kinetics400': root+'Video/kinetics/kinetics400',
        'kinetics_sound': root+'Video/kinetics/kinetics400', 
        'ucf101': root+'Video/ucf101',
        'hmdb51': root+'Video/hmdb51',
        'esc50': root+'Audio/ESC-50',
        'fsd50': root+'Audio/FSD50K',
        }
    
    try:
        return paths[db]   
    except:
        raise NotImplementedError(f'{db} is not available.')
        
