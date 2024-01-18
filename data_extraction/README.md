To run Volpy extraction code on the zebrafish data: 

1. please replace the following

utils.py : replaces util.py in $home/CaImAn/caiman/source_extraction/volpy/util.py
visualize.py : replaces util.py in $home/CaImAn/caiman/source_extraction/volpy/mrcnn/visualize.py

These are modifications to the original Volpy script for saving the ROI footprints. 

2. Main script to run: runScript_Volpy_Batch1.py

To run this code, please download the data files (each corresponding to a single z-layer) and save them into two separate directories (corresponding to two experiment trials). 

e.g. 
work_dir_trial_1 =  '/data/zebrafish/230820/Fish2_1/stitched/aligned_layers'

work_dir_trial_2 =  '/data/zebrafish/230820/Fish2_2/stitched/aligned_layers'

You can then replace these directory names in runScript_Volpy_Batch1.py to run the script.
