#!/usr/bin/env python
"""
pipeline for processing voltage imaging data for zebrafish lightsheet data
author: @Jack Zhang
"""
import scipy.io as sio
import cv2
import os
import glob
import h5py
import logging
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

import numpy as np
import os

try:
    cv2.setNumThreads(0)
except:
    pass

try:
    if __IPYTHON__:
        # this is used for debugging purposes only. allows to reload classes
        # when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.paths import caiman_datadir
from caiman.source_extraction.volpy import utils
from caiman.source_extraction.volpy.volparams import volparams
from caiman.source_extraction.volpy.volpy import VOLPY
from caiman.summary_images import local_correlations_movie_offline
from caiman.summary_images import mean_image
from caiman.utils.utils import download_demo, download_model
from caiman.source_extraction.volpy.mrcnn import visualize

# %%
# Set up the logger (optional); change this if you like.
# You can log to a file using the filename parameter, or make the output more
# or less verbose by setting level to logging.DEBUG, logging.INFO,
# logging.WARNING, or logging.ERROR
logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]" \
                    "[%(process)d] %(message)s",
                    level=logging.INFO)

# %%
def volpy_process_session(trial_1_dir, trial_2_dir, layer_name):
    pass  # For compatibility between running under Spyder and the CLI

    # %%  Load demo movie and ROIs
    # fnames = download_demo('demo_voltage_imaging.hdf5', 'volpy')  # file path to movie file (will download if not present)
    
    #fnames = os.path.join(trial_1_dir + layer_name + '.tif' );
    fnames = []
    fnames.extend([os.path.join(trial_1_dir + '/raw/' + layer_name + '.hdf5' )]);
    fnames.extend([os.path.join(trial_2_dir + '/raw/' + layer_name + '.hdf5' )]);

    #path_ROIs = download_demo('demo_voltage_imaging_ROIs.hdf5', 'volpy')  # file path to ROIs file (will download if not present)
    file_dir = os.path.split(fnames[0])[0]
    tempDir = '/home/jack/Caiman_Workfolder/Movies/'
    
#%% dataset dependent parameters
    # dataset dependent parameters
    fr = 200                                        # sample rate of the movie

    # motion correction parameterspath_ROIs
    pw_rigid = False                                # flag for pw-rigid motion correction
    gSig_filt = (4, 4)                              # size of filter, in general gSig (see below),
                                                    # change this one if algorithm does not work
    max_shifts = (6, 6)                             # maximum allowed rigid shift
    strides = (64, 64)                              # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (32, 32)                             # overlap between pathes (size of patch strides+overlaps)
    max_deviation_rigid = 5                         # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = 'copy'

    opts_dict = {
        'fnames': fnames,
        'fr': fr,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan
    }

    opts = volparams(params_dict=opts_dict)

# %% play the movie (optional)
    # playing the movie using opencv. It requires loading the movie in memory.
    # To close the movie press q
    display_images = False

    if display_images:
        m_orig = cm.load(fnames)
        ds_ratio = 0.2
        moviehandle = m_orig.resize(1, 1, ds_ratio)
        moviehandle.play(q_max=99.5, fr=40, magnification=4)

# %% start a cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

# %%% MOTION CORRECTION
    # first we create a motion correction object with the specified parameters
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    # Run correction
    do_motion_correction = True 
    if do_motion_correction:
        mc.motion_correct(save_movie=True)
        plt.subplot(1, 2, 1); plt.imshow(mc.total_template_rig)  # % plot template
        plt.subplot(1, 2, 2); plt.plot(mc.shifts_rig)  # % plot rigid shifts
        plt.legend(['x shifts', 'y shifts'])
        plt.xlabel('frames')
        plt.ylabel('pixels')

        savefig(trial_1_dir +'/outline/' + layer_name + '_motion-correct.pdf', bbox_inches='tight')
        
    else: 
        mc_list = [file for file in os.listdir(file_dir) if 
                   (os.path.splitext(os.path.split(fnames)[-1])[0] in file and '.mmap' in file)]
        mc.mmap_file = [os.path.join(file_dir, mc_list[0])]
        print(f'reuse previously saved motion corrected file:{mc.mmap_file}')

   # bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.int)

    

# %% compare with original movie
    if display_images:
        m_orig = cm.load(fnames)
        m_rig = cm.load(mc.mmap_file)
        ds_ratio = 0.2
        moviehandle = cm.concatenate([m_orig.resize(1, 1, ds_ratio),
                                      m_rig.resize(1, 1, ds_ratio)], axis=2)
        moviehandle.play(fr=40, q_max=99.5, magnification=4)  # press q to exit

# %% MEMORY MAPPING
    do_memory_mapping = True
    if do_memory_mapping:
        border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
        # you can include the boundaries of the FOV if you used the 'copy' option
        # during motion correction, although be careful about the components near
        # the boundaries
        
        # memory map the file in order 'C'
        fname_new = cm.save_memmap_join(mc.mmap_file, base_name='memmap_' + os.path.splitext(os.path.split(fnames[0])[-1])[0],
                                        add_to_mov=border_to_0, dview=dview)  # exclude border
    else: 
        mmap_list = [file for file in os.listdir(file_dir) if 
                     ('memmap_' + os.path.splitext(os.path.split(fnames)[-1])[0]) in file]
        fname_new = os.path.join(file_dir, mmap_list[0])
        print(f'reuse previously saved memory mapping file:{fname_new}')
    
# %% SEGMENTATION
    # create summary images
    img = mean_image(mc.mmap_file[0], window = 1000, dview=dview)
    img = (img-np.mean(img))/np.std(img)
    
    gaussian_blur = False        # Use gaussian blur when there is too much noise in the video
    Cn = local_correlations_movie_offline(mc.mmap_file[0], fr=fr, window=fr*4, 
                                          stride=fr*4, winSize_baseline=fr, 
                                          remove_baseline=True, gaussian_blur=gaussian_blur,
                                          dview=dview).max(axis=0)
    img_corr = (Cn-np.mean(Cn))/np.std(Cn)
    summary_images = np.stack([img, img, img_corr], axis=0).astype(np.float32)
    # save summary images which are used in the VolPy GUI
    #cm.movie(summary_images).save(fnames[:-5] + '_summary_images.tif')
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(summary_images[0]); axs[1].imshow(summary_images[2])
    axs[0].set_title('mean image'); axs[1].set_title('corr image')

    #%% methods for segmentation
    methods_list = ['manual_annotation',       # manual annotations need prepared annotated datasets in the same format as demo_voltage_imaging_ROIs.hdf5 
                    'maskrcnn',                # Mask R-CNN is a convolutional neural network trained for detecting neurons in summary images
                    'gui_annotation']          # use VolPy GUI to correct outputs of Mask R-CNN or annotate new datasets 
                    
    method = methods_list[2]
    if method == 'manual_annotation':                
        with h5py.File(path_ROIs, 'r') as fl:
            ROIs = fl['mov'][()]  

    elif method == 'maskrcnn':                 
        weights_path = download_model('mask_rcnn')    
        ROIs = utils.mrcnn_inference(img=summary_images.transpose([1, 2, 0]), size_range=[3, 18],
                                     weights_path=weights_path, display_result=True, savedir=trial_1_dir, savename=layer_name) # size parameter decides size range of masks to be selected
        cm.movie(ROIs).save(fnames[0][:-5] + '_mrcnn_ROIs.hdf5')

    elif method == 'gui_annotation':
        # run volpy_gui.py file in the caiman/source_extraction/volpy folder
        # or run the following in the ipython:  %run volpy_gui.py
        #gui_ROIs =  caiman_datadir() + '/example_movies/volpy/gui_roi.hdf5'
        #gui_ROIs = '/media/jack/data/zebrafish/230820/Fish2_1/TIFF_files/layer_0_320_80_mrcnn_ROIs.hdf5'

        gui_ROIs = os.path.join(trial_1_dir + '/roi/' + layer_name + '_roi.hdf5' )
        with h5py.File(gui_ROIs, 'r') as fl:
            ROIs = fl['mov'][()]

        box = np.zeros([len(ROIs[:,0,0]),4],dtype=np.int32)
        #roi_id = list(range(len(ROIs[:,0,0])))
        roi_id = np.full((len(ROIs[:,0,0])),1,dtype=np.int32)
        

        for nn in range(len(ROIs[:,0,0])):
            xnzero = np.nonzero(np.sum(ROIs[nn,:,:], axis=0))   
            x1 = xnzero[0][0]
            x2 = xnzero[0][-1]
    
            ynzero = np.nonzero(np.sum(ROIs[nn,:,:], axis=1))   
            y1 = ynzero[0][0]
            y2 = ynzero[0][-1]
    
            box[nn] = y1, x1, y2, x2
    

        _, ax = plt.subplots(1,1, figsize=(16,16))
        sampleimg=summary_images.transpose([1, 2, 0])
        ROI_t = np.transpose(ROIs,(1, 2, 0))
        visualize.display_instances_no_contour(sampleimg, box, ROI_t, roi_id, 
                        ['BG', 'neurons'], roi_id, ax=ax, captions=None,
                        title="Predictions")
        #visualize.display_instances(img=summary_images.transpose([1, 2, 0]), box, ROIs, roi_id, ['BG', 'neurons'], roi_id, ax=ax, title="Predictions")
        savefig(trial_1_dir +'/outline/' + layer_name + '_roi_outline.pdf', bbox_inches='tight')

        visualize.display_instances(sampleimg, box, ROI_t, roi_id, 
                        ['BG', 'neurons'], roi_id, ax=ax, captions=None,
                        title="Predictions")
        #visualize.display_instances(img=summary_images.transpose([1, 2, 0]), box, ROIs, roi_id, ['BG', 'neurons'], roi_id, ax=ax, title="Predictions")
        savefig(trial_1_dir +'/outline/' + layer_name + '_roi_outline_contour.pdf', bbox_inches='tight')


    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(summary_images[0]); axs[1].imshow(ROIs.sum(0))
    axs[0].set_title('mean image'); axs[1].set_title('masks')
        
# %% restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False, maxtasksperchild=1)

# %% parameters for trace denoising and spike extraction
    ROIs = ROIs                                   # region of interests
    index = list(range(len(ROIs)))                # index of neurons
    weights = None                                # if None, use ROIs for initialization; to reuse weights check reuse weights block 

    template_size = 0.02                          # half size of the window length for spike templates, default is 20 ms 
    context_size = 35                             # number of pixels surrounding the ROI to censor from the background PCA
    visualize_ROI = False                         # whether to visualize the region of interest inside the context region
    flip_signal = False                           # Important!! Flip signal or not, True for Voltron indicator, False for others
    hp_freq_pb = 1 / 3                            # parameter for high-pass filter to remove photobleaching
    clip = 101                                    # maximum number of spikes to form spike template
    threshold_method = 'adaptive_threshold'                   #'adaptive_threshold'       # adaptive_threshold or simple 
    min_spikes= 4                               # minimal spikes to be found
    pnorm = 0.5                                   # a variable deciding the amount of spikes chosen for adaptive threshold method
    threshold = 5                                 # threshold for finding spikes only used in simple threshold method, Increase the threshold to find less spikes
    do_plot = False                               # plot detail of spikes, template for the last iteration
    ridge_bg= 0.01                                # ridge regression regularizer strength for background removement, larger value specifies stronger regularization 
    sub_freq = 20                                 # frequency for subthreshold extraction
    weight_update = 'ridge'                       # ridge or NMF for weight update
    n_iter = 2                                    # number of iterations alternating between estimating spike times and spatial filters
    
    opts_dict={'fnames': fname_new,
               'ROIs': ROIs,
               'index': index,
               'weights': weights,
               'template_size': template_size, 
               'context_size': context_size,
               'visualize_ROI': visualize_ROI, 
               'flip_signal': flip_signal,
               'hp_freq_pb': hp_freq_pb,
               'clip': clip,
               'threshold_method': threshold_method,
               'min_spikes':min_spikes,
               'pnorm': pnorm, 
               'threshold': threshold,
               'do_plot':do_plot,
               'ridge_bg':ridge_bg,
               'sub_freq': sub_freq,
               'weight_update': weight_update,
               'n_iter': n_iter}

    opts.change_params(params_dict=opts_dict);          

#%% TRACE DENOISING AND SPIKE DETECTION
    vpy = VOLPY(n_processes=n_processes, dview=dview, params=opts)
    vpy.fit(n_processes=n_processes, dview=dview)
   
#%% visualization
    display_images = False
    if display_images:
        print(np.where(vpy.estimates['locality'])[0])    # neurons that pass locality test
        idx = np.where(vpy.estimates['locality'] > 0)[0]
        utils.view_components(vpy.estimates, img_corr, idx)
    
#%% reconstructed movie
# note the negative spatial weights is cutoff    
    # if display_images:
    #     mv_all = utils.reconstructed_movie(vpy.estimates.copy(), fnames=mc.mmap_file,
    #                                        idx=idx, scope=(0,1000), flip_signal=flip_signal)
    #     mv_all.play(fr=40, magnification=3)
#%% save the traces
#vpy.estimates['t']
#for k, v in vpy.estimates.items():
    sio.savemat(os.path.join(trial_1_dir + '/matvars/T/T_' + layer_name + '.mat'), mdict={'T': vpy.estimates['t']})
    sio.savemat(os.path.join(trial_1_dir + '/matvars/spikes/spikes_' + layer_name + '.mat'), mdict={'spikes': vpy.estimates['spikes']})
    sio.savemat(os.path.join(trial_1_dir + '/matvars/threshold/thresh_' + layer_name + '.mat'), mdict={'thresh': vpy.estimates['thresh']})
    sio.savemat(os.path.join(trial_1_dir + '/matvars/snr/snr_' + layer_name + '.mat'), mdict={'snr': vpy.estimates['snr']})
    sio.savemat(os.path.join(trial_1_dir + '/matvars/t_sub/t_sub_' + layer_name + '.mat'), mdict={'t_sub': vpy.estimates['t_sub']})
    
    cm.movie(vpy.estimates['weights']).save(os.path.join(trial_1_dir + '/matvars/weights/weights_' + layer_name + '.hdf5'))
        

# Extract raw traces
    # Y = cm.load(fname_new)  # Load memory mapped movie
    # raw_traces = extract_raw_traces(Y, ROIs)    
    # # Save raw traces to a file
    # save_name2 = f'ResultVolpy_{os.path.split(fnames)[1][:-4]}_raw_traces.npy'  
    # np.save(save_name2, raw_traces)
    # print(f"Raw traces saved to {save_name2}")
    
    
# Function to extract raw traces
# def extract_raw_traces(Y, masks):
#     """
#     Extract raw traces from the given movie Y and masks.

#     Parameters:
#     ----------
#     Y : movie
#         The movie in which traces are to be extracted. Shape (T, d1, d2) where T is number of frames.
#     masks : array
#         ROIs masks. Shape (num_neurons, d1, d2).

#     Returns:
#     -------
#     raw_traces : array
#         Extracted raw traces. Shape (num_neurons, T).
#     """
#     T = Y.shape[0]
#     num_neurons = masks.shape[0]
#     raw_traces = np.empty((num_neurons, T))
#     for idx_neuron in range(num_neurons):
#         mask = masks[idx_neuron]
#         for t in range(T):
#             raw_traces[idx_neuron, t] = np.mean(Y[t][mask])
#     return raw_traces    
    
#%% save the result in .npy format 
    save_result = False
    if save_result:
        #vpy.estimates['ROIs'] = ROIs
        vpy.estimates['params'] = opts
        #save_name = f'volpy_{os.path.split(fnames)[1][:-5]}_{threshold_method}'
        np.save(os.path.join(trial_1_dir + '/matvars/results_' + layer_name), vpy.estimates)

        #np.save(os.path.join(file_dir, save_name), vpy.estimates)
        
#%% reuse weights 
## set weights = reuse_weights in opts_dict dictionary
#    estimates = np.load(os.path.join(file_dir, save_name+'.npy'), allow_pickle=True).item()
#    reuse_weights = []
#    for idx in range(ROIs.shape[0]):
#        coord = estimates['context_coord'][idx]
#        w = estimates['weights'][idx][coord[0][0]:coord[1][0]+1, coord[0][1]:coord[1][1]+1] 
#        #plt.figure(); plt.imshow(w);plt.colorbar(); plt.show()
#        reuse_weights.append(w)
    
# %% STOP CLUSTER and clean up log files
    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
        
# %% Remove memmap files
    memmap_list = os.listdir(file_dir)

    for item in memmap_list:
        if item.endswith(".mmap"):
            os.remove(os.path.join(file_dir, item))

    file_dir_2 = os.path.split(fnames[1])[0]
    memmap_list = os.listdir(file_dir_2)

    for item in memmap_list:
        if item.endswith(".mmap"):
            os.remove(os.path.join(file_dir_2, item))
            
# for now remove hdf5 too        
    # hdf5_list = os.listdir(file_dir)

    # for item in memmap_list:
    #     if item.endswith(".hdf5"):
    #         os.remove(os.path.join(file_dir, item))
        
# %%
# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()