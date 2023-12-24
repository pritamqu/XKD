import os

def my_paths(server, db):
    
    
    if server == 'local':
        output_dir = '/mnt/PS6T/OUTPUTS'
        data_dir = fetch_linux_db(db)
        
    else:
        raise ValueError()
        
    return output_dir, data_dir


def fetch_linux_db(db):
    
    root="/mnt/PS6T/datasets/"
    
    paths = {
        'audioset': root+'AudioSet',
        'kinetics700': root+'Video/kinetics/kinetics700',
        'kinetics400': root+'Video/kinetics/kinetics400',
        'kinetics_sound': root+'Video/kinetics/kinetics400', 
        'ucf101': root+'Video/ucf101',
        'hmdb51': root+'Video/hmdb51',
        'esc50': root+'Audio/ESC-50',
        'dcase': root+'Audio/DCASE',
        'ssv2': root+'Video/ssv2',
        'charades': root+'Video/charades_video',
        'epic_kitchen': root+'Video/epic-kitchen',
        'fsd50': root+'Audio/FSD50K',
        'openmic': root+'Audio/openmic-2018'
        }
    
    try:
        return paths[db]   
    except:
        raise NotImplementedError(f'{db} is not available in local.')
  
        
