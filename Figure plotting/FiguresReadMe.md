# 10/29/2024, Shengyuan Yan, PhD Candidate, IRIS group, DP Mathematics&ComputerScience, Eindhoven University of Technology, Eindhoven, Netherlands.

This directory contains official standard implementation of the figure plotting in the Optics Express journal paper "Deep learning based phase retrieval with complex beam shapes for beam shape correction" of EU InShaPe project funded by the European Union (EU Funding Nr.: 101058523 — InShaPe).
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