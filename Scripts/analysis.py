import os
import numpy as np
import pandas as pd
from func_utils import load_volume

def analyse_predictions(test_set,pred_path,mask_path,vess_path,name='Other'):
    
    pred_path = os.path.join(pred_path, 'Output/Predictions/')

    result_roi_gt = np.zeros([len(test_set)],dtype='float32')
    result_roi_pred = np.zeros([len(test_set)],dtype='float32')
    result_vess_gt = np.zeros([len(test_set)],dtype='float32')
    result_vess_pred = np.zeros([len(test_set)],dtype='float32')
    
    df = pd.DataFrame({'MouseID': test_set})
    df = df.MouseID.str.split('_',expand=True)
    #  Remove motion correction phrase
    df[4] = df[4].str.replace('mcCorr1','')
    # Remove any addition phrase (e.g. leg)
    df[4] = df[4].str.replace(r'[^\d.]+','')
    
    store = (df[0] + '_' + df[1] + '_' + df[2])
    
    for i in range(len(test_set)):
        
        print('Analysing ... %s' %test_set[i])
        
        # Dataset training directory
        train_path = os.path.join(mask_path,test_set[i])
        
        # Analyse tumour volume
        subfolder = os.path.join(train_path,'roi_Mask/')
        roi_gt = load_volume(subfolder, datatype='uint8')
        
        file = test_set[i] + '_prediction_upscaled'
        filepath = os.path.join(mask_path, test_set[i], file)
        roi_pred = load_volume(filepath, stack=True, ext='.tiff', datatype='uint8')
        
        result_roi_gt[i] = np.count_nonzero(roi_gt)
        result_roi_pred[i] = np.count_nonzero(roi_pred)
        
        # Analyse blood volume
        files = os.listdir(vess_path)
        for j in range(len(store)):
            if not files[j].find(store[i]) == -1:
                rootfile = files[j]
        
        file = test_set[i] + '_vessel_sato_mask'
        filepath = os.path.join(vess_path, rootfile)
        filepath = os.path.splitext(filepath)[0]
        vess_gt = load_volume(filepath, stack=True, ext='.tiff', datatype='uint8')
        

        # Find location of vessels
        idx = np.nonzero(vess_gt)
        roi_gt = np.transpose(roi_gt,(2,0,1))
        for (m,n,l) in zip(*idx):
            if roi_gt[m-1, n-1, l-1] > 0:
                result_vess_gt[i] = result_vess_gt[i] + 1.0
            if roi_pred[m-1, n-1, l-1] > 0:
                result_vess_pred[i] = result_vess_pred[i] + 1.0
                

    result_roi_gt = 100 * result_roi_gt / roi_gt.size
    result_roi_pred = 100 * result_roi_pred / roi_pred.size
    
    result_vess_gt = 100 * result_vess_gt / vess_gt.size
    result_vess_pred = 100 * result_vess_pred / vess_gt.size
    
    filename = 'Tumour_Analysis.csv'
    df = pd.DataFrame({'Full ID': test_set,
                       'Time_Stamp': df.iloc[:,1],
                       'Date': df.iloc[:,2],
                       'Animal ID': df.iloc[:,3]+df.iloc[:,4],
                       'ROI Volume Fraction ({name})'.format(name=name): result_roi_gt,
                       'ROI Volume Fraction (CNN)': result_roi_pred,
                       'Blood Volume Fraction ({name})'.format(name=name): result_vess_gt,
                       'Blood Volume Fraction (CNN)': result_vess_pred})
    df.to_csv(filename)
 