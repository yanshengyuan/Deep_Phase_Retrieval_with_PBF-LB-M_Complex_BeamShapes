# 10/29/2024, Shengyuan Yan, PhD Candidate, IRIS group, DP Mathematics&ComputerScience, Eindhoven University of Technology, Eindhoven, Netherlands.

This directory contains official standard implementation of the training and testing of the deep learning models benchmarked in the Optics Express journal paper "Deep learning based phase retrieval with complex beam shapes for beam shape correction" of EU InShaPe project funded by the European Union (EU Funding Nr.: 101058523 — InShaPe).
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