# DL_PSYCHE
A deep learning method for obtaining sensitivity-enhanced pure shift NMR spectra. 

# Profiles
PSYCHE experimental data of azithromycin, estradiol and ibuprofen:
1. exp_azithromycin.mat
2. exp_estradiol.mat
3. exp_ibuprofen.mat

Model weight file (model_rtzs.h5) can be download from the weblink: https://www.dropbox.com/s/1yfoj91r7n2d505/model_rtzs.h5

Model can be tested or used by the file:
1. modelTest.py

Model training code is shown in the file：
1. train.py
2. generator0.py
3. model_logic2.py

The network code is shown in the file：
1. HRNet_1D.py

# Dependencies
1. keras == 2.2.4
2. numpy == 1.19.2
3. tensorfolw == 1.14.0
4. h5py == 2.10.0

Model has been written and tested with the above dependencies. Performance with other module versions has not been tested.

# Preparing Data
The input to the model must be in '.mat' format with variable name of 'data', which can be edited in MATLAB or Python. 

Prior to input into the network model, the FID data containing less than 4096 complex points needs to be zero-filled to 4096 complex points and Fourier transformed to the spectrum. Then, the spectrum is phased, taken as the real part, and normalized to 1. 

If FID contains more than 4096 complex points or the spectral width is larger than 4096 Hz, FID can be zero-filled to 8192 or more (integer multiple of 4096) complex points, then the corresponding spectrum can be divided into two or more spectra with 4096 real points as inputs, and finally the processed spectra can be concatenated into a complete spectrum.
