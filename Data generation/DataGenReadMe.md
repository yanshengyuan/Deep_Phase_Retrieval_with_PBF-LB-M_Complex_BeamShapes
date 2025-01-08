# 10/29/2024, Shengyuan Yan, PhD Candidate, IRIS group, DP Mathematics&ComputerScience, Eindhoven University of Technology, Eindhoven, Netherlands.

This directory contains official standard implementation of the simulation of the PBF-LB/M laser beam shaping system in the Optics Express journal paper "Deep learning based phase retrieval with complex beam shapes for beam shape correction" of EU InShaPe project funded by the European Union (EU Funding Nr.: 101058523 — InShaPe).
project URL: https://inshape-horizoneurope.eu/




Library and version requirements:
numpy==1.26.3
python==3.11.9
matplotlib==3.5.2
lightpipes==2.1.4




Reproduce the training and testing dataset generation via the simulation of the PBF-LB/M laser beam shaping system:
1, For each beam shape there is a dedicated folder named 'densebeamshape30k/'. Let's use Chair shape as the example to demonstrate how to run the simulation.
2, For Chair shape, enter the folder 'densechair30k/'
3, The parameters of the simulation are taken in the files "Input_Data/Config_AI_Data_Generator-train" and "Input_Data/Config_AI_Data_Generator-test". Because of the confidential regulations of InShaPe consortium, some key parameters of the optical simulation are given obviously random values. However, since the mathematical relationship between them is still preserved, the randomness in the values of these parameters will not influence the results of the simulation at all (in fact, they cooperating together do not even influence the mathematical numerical process of the entire simulation at all.)
4, The only png gray-scale image in the folder "Input_Data/" is the target aberration-free LCOS-SLM phase mask for that beam shape, however, stored in computer as a gray-scale image with [-pi, +pi] phase values normalized to [0, 255] pixel values. They will be transformed back to corresponding phase values when they are input into the optical simulation.
5, To start the simulations, run commands:
                                        python3 train.py   to simulate the training set
                                        python3 test.py   to simulate the test set
6, The simulated intensity images of the distorted beam shapes will be in the folders 'Output_Data/' for the training set and 'Val_Data/' for the test set.

Explanation of the simulated dataset:
For both the training set and test set, there are 7 gray-scale png images paired with a numpy array file for one simulated optical sample, for example, the following files comprise a simulated sample:
run0001_intensity_01-pre6.png
run0001_intensity_02-pre5.png
run0001_intensity_03-pre4.png
run0001_intensity_04-foc.png
run0001_intensity_05-pre4.png
run0001_intensity_06-pre5.png
run0001_intensity_07-pre6.png
run0001_zernikeCoeff.npy

In this sample, which has 8 files, run0001_intensity_01-pre6.png, run0001_intensity_02-pre5.png, and run0001_intensity_03-pre4.png are the pre-Fourier planes images, run0001_intensity_04-foc.png is the Fourier plane image, and run0001_intensity_05-pre4.png, run0001_intensity_06-pre5.png, run0001_intensity_07-pre6.png are the post-Fourier planes images; run0001_zernikeCoeff.npy is the randomly sampled Zernike Coefficients numpy array that works as the groundtruth label for supervised learning.
