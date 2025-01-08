experiment_params = dict(                   
                         model_path='mlruns/925193670559623593/723e11a9ef0f49e1b28f21b4df2f67a3/artifacts/model',
                         data_folder='../densebench30k_morez',
                         batch_size=1, # batch size for pytorch dataset
                         num_workers=1, # number of workers for data fetching
                         focal_dist=[-2, 0, 2], # list of focal distances intended to fetch images
                         )