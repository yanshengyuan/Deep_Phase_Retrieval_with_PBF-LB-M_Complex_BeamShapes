experiment_params = dict(                   
                         model_path='./mlruns/759329051334174262/9badc21b420246049e6f1025076f0f64/artifacts/model',
                         data_folder='../DenseChair30k',
                         batch_size=128, # batch size for pytorch dataset
                         num_workers=4, # number of workers for data fetching
                         focal_dist=[-2, 0, 2], # list of focal distances intended to fetch images
                         )