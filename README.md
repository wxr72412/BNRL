# The codes and datsets of TAI manuscript.

## 1. Setting requirment
filepath: /requirements.txt

## 2. Data preprocessing
Download 4 BN datasets in BIF format from the bnrepository and execute data preprocessing.

(1) (filepath: \data\bn\a1_plot_bn.py)  
Visualize the DAG structure of BN.  
(2) (filepath: \data\bn\a2_rename_state_settting_none)  
Transform variable values from text to discrete values.     
(3) (filepath: \data\bn\ a3_verify_renameBN)  
Verify the correctness of BIF format file in the step (2).   
(4) (filepath: \data\bn\ a4_change_edges.py)  
Modifying the local structure of BN for the convenient conduction of experiments.  

## 3. Hyperparameter setting
(1) (filepath: \BNML\init_para\child_L2_init_para)  
Hyperparameter setting of our BNRL  
(2) (filepath: \para_init.py)  
Hyperparameter setting of Comparison methods  

## 4. Our method
(1) (filepath: \BNML\toy-1-train.py)  
Algorithm 1: VQVAE training  
(2) (filepath: \BNML\toy-2-train.py)   
Algorithm 2: Parameter learning of BNRL  
(3) (filepath: \BNML\toy-3-train.py)  
Reconstruct reduced latent variables of BNRL into original latent variables by the decoder of VQAVE.
This enable the parameters of BNRL to be compared with comparison methods.  
(4) (filepath: \BNML\toy-4-train.py)   
Calculate 3 error metrics (i.e, MSE, MAE, and KL divergence) based on the original and learned BNML by BNRL, EM, IEM, DEBN, TRIP, RPL and ARPL.

## 5. Comparison methods
(1) (filepath: \EM.pyIEM)   
EM  
(2) (filepath: \Improved_EM.py)  
IEM  
(3) (filepath: \DEBN.pyTRIP)  
DEBN  
(4) (filepath: \TRIP.py)  
TRPL  
(5) (filepath: \dynRNN_BN.py)  
RPL  
(6) (filepath: \dynAERNN_BN.py)  
ARPL

## 6. Probabilistic inference
(1) (filepath: \BNML\toy-5-inference)   
Inplement 3 approximate probabilistic inference methods (i.e., FS, GS, and RS) on the BNML.  
(2) (filepath: \BNML\toy-6-inference)   
Calculate 4 effectiveness metrics (i.e, F1, AUC, mAP, and NDGC) based on the inference results of VE and 4 other methods (i.e., our BNRL, FS, GS, and RS).
