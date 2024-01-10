# -*- coding: utf-8 -*-
"""

@author: Jack Zhang
"""

from fish_single_layer_manual import volpy_process_session
volpy_process_session

work_dir_trial_1 =  '/data/zebrafish/230820/Fish2_1/stitched/aligned_layers'
work_dir_trial_2 =  '/data/zebrafish/230820/Fish2_2/stitched/aligned_layers'


data_name_all = [
                            'stitch_layer_0',
                              'stitch_layer_1',
                            'stitch_layer_2',
                            'stitch_layer_3',
                            'stitch_layer_7',
                            'stitch_layer_9',
                            'stitch_layer_10',
                           'stitch_layer_11',
                           'stitch_layer_12',
                           'stitch_layer_13',
                          'stitch_layer_14',
                          'stitch_layer_15',
                          'stitch_layer_16',
                          'stitch_layer_17',
                        'stitch_layer_18',
                        'stitch_layer_27',                  
                        'stitch_layer_19',
                        'stitch_layer_20',
                        'stitch_layer_24',
                        'stitch_layer_25',
                        'stitch_layer_26',    
                        'stitch_layer_28',
                          'stitch_layer_29',
                          'stitch_layer_21',
                          'stitch_layer_22',
                          'stitch_layer_23',
                          'stitch_layer_4',
                        'stitch_layer_5',
                          'stitch_layer_8',
                        'stitch_layer_6',
                    ]

for data_name in data_name_all:
    volpy_process_session(work_dir_trial_1, work_dir_trial_2, data_name)


