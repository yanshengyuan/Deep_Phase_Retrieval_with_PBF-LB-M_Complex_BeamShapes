# 10/29/2024, Shengyuan Yan, PhD Candidate, IRIS group, DP Mathematics&ComputerScience, Eindhoven University of Technology, Eindhoven, Netherlands.

This directory contains official standard implementation of aberration detection and beam shape correction in the Optics Express journal paper "Deep learning based phase retrieval with complex beam shapes for beam shape correction" of EU InShaPe project funded by the European Union (EU Funding Nr.: 101058523 — InShaPe).
project URL: https://inshape-horizoneurope.eu/




Library and version requirements:
lightpipes==2.1.4
numpy==1.26.3
opencv==4.9.0
python==3.11.9




Reproduce the numbers reported in the tables Tab. 1. and Tab. 2. of the paper:
1, Preparation: unzip the 'beamshape_name_testset.zip' files to decompress the testset images of each beamshape. They are mandatorily needed to compute the reconstruction error.

2, For each beamshape there is a dedicated folder named after it. We take Chair shape for example to demonstrate how to operate the scripts.

3, Enter folder 'chair/', you will see such two files named after a benchmarked model. Let's take resnet for example, the two files are 'resnet.py' and 'resnet.npy'.

4, 'resnet.npy' contains the 3000 sets of predicted Zernike Coefficients predicted by the ResNet18 model trained on the Chair beam shape data. This numpy array file will serve as the input of the simulation script of aberration detection and beam shape correction.

5, 'resnet.py' is the simulation script that simulate the process of aberration detection and beam shape correction using the Zernike Coefficients predicted by the ResNet18.

6, There is a 'gt.npy' file that contains the groundtruth Zernike Coefficients for all 3000 testing samples in the path "../chair_testset/". They are the groundtruth needed to compute the MAE(Z), wavefront error, and correction error.

7, If all files mentioned above exist in the indicated path then run command: nohup python3 resnet.py  >resnet.txt 2>&1 &

8, Running 'resnet.py' will generate:
                                     the resnet_Metrics.npy file that contains 3000 records of all four computed metrics
                                     the resnetScatter_Matrix.png that plots the correlation between every two metrics
                                     the visualized reconstructed intensity images in "ReconsErr/"
                                     the visualized corrected beam shapes in "CorrErr/"
                                     the resnet.txt log file that reports the values of the five (one extra as the wrapped-phase wavefront error) computed metrics and the average values of them in the end. The numbers at the end of this log file are the average performance numbers reported in the paper.




Reproduce the visualized examples in the Fig. 9. Worst cases visualization of aberration detection and Fig. 13. Worst cases visualization of beam shapes correction:
1, Run the script 'beamshape_name/PhaseMask_WorstVis.py': python3 PhaseMask_WorstVis.py
2, The visualized worst examples will be in the path "beamshape_name/PhaseMaps"
3, Do step 1 and 2 for every beam shape's root folder to get the worst case visualizations for each beam shape




Reproduce the visualized examples in the Fig. 10. Average cases visualization of aberration detection and Fig. 14. Average cases visualization of beam shape correction:
1, For each beam shape, enter the corresponding folder 'beamshape_name/'. There will be a script PhaseMask_SelectedVis.py in it.
2, Run this script: python3 PhaseMask_SelectedVis.py
3, The visualized average representative examples will be in the path "beamshape_name/PhaseMaps"
4, Do step 1, 2 and 3 for every beam shape's root folder to get the average representative case visualizations for each beam shape