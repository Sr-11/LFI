# Rename
import os 
directory = '/math/home/eruisun/github/LFI/methods/Res_Net/checkpoints'
for dirpath, dirnames, filenames in os.walk(directory):
    for filename in filenames:
        if filename == 'type_2_error.npy':
            full_file_name = os.path.join(dirpath, filename)
            new_file_name = full_file_name.replace('type_2_error.npy', 'type_2_error_subset.npy')
            print(new_file_name)
            os.rename(full_file_name, new_file_name)
