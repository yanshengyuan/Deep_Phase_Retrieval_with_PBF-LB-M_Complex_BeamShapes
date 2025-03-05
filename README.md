# Deep_Phase_Retrieval_with_PBF-LB-M_Complex_BeamShapes
This repository is the code officially published and submitted in OPTICA, USA of our Optics Express journal paper publication in 2025.

Our paper: Deep learning based phase retrieval with complex beam shapes for beam shape correction

Journal: Optics Express of OPTICA, USA.

DOI: https://doi.org/10.1364/OE.547138

Citation: S. Yan, R. Off, A. Yayak, K. Wudy, A. Aghajani-Talesh, M. Birg, J. Grünewald, M. Holenderski, and N. Meratnia, "Deep learning based phase retrieval with complex beam shapes for beam shape correction," Opt. Express  33, 10806-10834 (2025). 

Please go to different folders for different experiments and different functionalities of our paper. There is respective detailed readme file instructions on how to use the scripts in each folder.

Acknowledgements:

This work was supported by the EU InShaPe project (https://inshape-horizoneurope.eu) funded by the European Union (EU Funding Nr.: 101058523 — InShaPe).

The following instructions guide the users how to use the codes in different functionalities in different folders.



1), Data generation (optical simulation codes):

The directory main/"Data generation/" contains official standard implementation of the simulation of the PBF-LB/M laser beam shaping system in the Optics Express journal paper "Deep learning based phase retrieval with complex beam shapes for beam shape correction" of EU InShaPe project funded by the European Union (EU Funding Nr.: 101058523 — InShaPe).
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



2), Deep neural network training and testing codes

The directory main/Deep_Phase_Retrieval_Training/ contains official standard implementation of the training and testing of the deep learning models benchmarked in the Optics Express journal paper "Deep learning based phase retrieval with complex beam shapes for beam shape correction" of EU InShaPe project funded by the European Union (EU Funding Nr.: 101058523 — InShaPe).
project URL: https://inshape-horizoneurope.eu/




Library and version requirements:

lightpipes==2.1.4

numpy==1.26.3

opencv==4.9.0

python==3.11.9

pytorch==2.1.1

torchmetrics==1.1.2

pytorch-lightning==2.1.3

timm==0.9.12

torchvision==0.16.1

mlflow==2.6.0




Training of the models:

1, Specify in the configs_train.py: The path to your phase retrieval training data, DOI of the compared method, architecture name of the employed model, model scale, intended run name of MLFlow project and the experiment name in it, number of epochs, number of training samples, batch size, selection of input caustic planes by focal distance, initial learning rate, and step size of the learning rate decay strategy. Each domain in the configs_train.py script is mandatory!

2, Training command: python3 train.py

3, Pretrained weights (trained model): Updated and stored in the run ID folder in the root folder 'mlruns/'. E.g., "mlruns/189416205877355165/002cb96cd03d41439a8196d63f4c2bf2/artifacts/model/data/model.pth"




Testing of the models:

1, Specify in the configs_test.py: The path to your testing data, the path to the trained model as indicated in the bullet point above, batch size, and selection of input caustic planes by focal distance. Each domain in the configs_test.py script is mandatory!

2, Testing command: python3 test.py

   or the command to test all existing models in the 'mlruns/' folder: python3 test_all.py

3, The predicted Zernike Coefficients will be stored in the 'predictions/' folder in numpy array files.




Extra information:

You can find extra information about the structures and the periodically saved checkpoints of the models in each experiment in the folder 'lightning_logs'



3), Simulation-assisted analysis codes for the evaluation of aberration detection and correction.

The directory main/Simulation_assisted_analysis-Aberration_Detection_and_Correction/ contains official standard implementation of aberration detection and beam shape correction in the Optics Express journal paper "Deep learning based phase retrieval with complex beam shapes for beam shape correction" of EU InShaPe project funded by the European Union (EU Funding Nr.: 101058523 — InShaPe).
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



4). Figure plotting codes

The directory main/"Figure plotting" contains official standard implementation of the figure plotting in the Optics Express journal paper "Deep learning based phase retrieval with complex beam shapes for beam shape correction" of EU InShaPe project funded by the European Union (EU Funding Nr.: 101058523 — InShaPe).
project URL: https://inshape-horizoneurope.eu/




Library and version requirements:

numpy==1.26.3

opencv==4.9.0

python==3.11.9

matplotlib==3.5.2

seaborn==0.11.2




To reproduce all figures other than the two KDE plots (Fig. 8. The kernel density estimate (KDE) of the distribution of MAE(Z) and reconstruction error and Fig. 12. The kernel density estimate (KDE) of the distribution of wavefront error and correction error).

1, Enter the folder 'Benchmarking_and_Ablation_Study/' and you will see a script 'draw_dotchart.py' in it.

2, Open the script 'draw_dotchart.py' you will see several segments of codes, each of which is wrapped between two multi-line commenting characters like the following example

'''

# what analysis this segment of codes is doing, start!

codes......

codes......

# what analysis this segment of codes is doing, end!

'''

3, Each segment is dedicated to plotting one of the figures in the paper. To reproduce a figure, use single-line commenting character # to disable the multi-line commenting characters ''' at the beginning and the end of the segment to activate the function of this segment.

4, Run the script: python3 draw_dotchart.py

5, Remove the character # you added just now in the script to disable the function of the segment again. This step is necessary otherwise the plotting programs will intervene each other and yield chaotic results. The 3D bar chart plotting program is highly complicated and its presence in this script makes this script has to work this way one plotting at a time for each figure.

6, Do the step 2,3,4,5 for all segments in the script.




To reproduce the two KDE figures:

1, Enter the folder 'Error_Distribution/' and you will see a script 'ErrDistr.py'

2, Open it and you will see similar structure with the previously mentioned script 'draw_dotchart.py'.

3, Use # to activate and disable each segment of codes and run the script for according times as already introduced previously in the last section.