experiment_params = dict(doi='None',      # None, 10.1063/1.5125252, 10.1038/s41592-018-0153-5, 10.1364/OE.401933,
                         arch='ViT', #resnet, mlp, ViT
                         modelsize = 'base', #base, tiny, small, large, huge
                         #run_name='CNN(smNet)-Default_Params',                    # ResNet,                  CNN(smNet),            CNN(PhaseNet)
                         data_folder='../densebench30k_morez',
                         run_name='Submission',                    # ResNet,                  CNN(smNet),            CNN(PhaseNet)
                         seed=3, 
                         train=True, # labels (zernike coeff) included or not
                         num_epochs=1000, # number of epochs for training
                         num_samples=10000, # number of training samples
                         num_zernikes=12,
                         batch_size=128, # batch size for pytorch dataset
                         num_workers=4, # number of workers for data fetching
                         log_every_n_steps=1, # monitoring frequency of training
                         save_top_k=1, # save best k models
                         best_model_metric='mae_val', # metric that the best model selection is based on: 'ARE(_val)', 'mse(_val)', 'mae(_val)'
                         focal_dist=[-2, 0, 2], # list of focal distances [-3, -2, -1, 0, 1, 2, 3] intended to fetch images
                         experiment_name='submission', # the name of MLflow experiment
                         init_lr=2e-4,
                         stepsize=30
                         )