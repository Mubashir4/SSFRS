# SSFRS
The code consists of three main files and three supporting files:

1. Main.py
2. User_Module.py 
    - VQ.py
    - ⋅VQ-Trainer.py
    - utility.py
3. Cluster.py


**User_Module:** contains the implementation of clients Embedding model, which uses VQ-VAE to learn embeddings. These embeddings are updated and maintined by each users object in the main file.

**Cluster:** It is the implementation of Server and contains VAE based semantic sampler and the Clustring Unit.

**Main:** For the easy of simulation/processing we have currently uploaded iterative version of the implementation.

Both datasets i.e, MovieLens and Last.fm can be downloaded from the link below. Due to large size of datasets, download links are provided. We also provide implementation to format both datasts. These implementations use word2vec inorder ot convert text into vector format.

Data References:
1. https://grouplens.org/datasets/movielens/20m/
2. https://www.upf.edu/web/mtg/lastfm360k

Code References:
1. https://github.com/HenningBuhl/VQ-VAE_Keras_Implementation
2. https://github.com/keras-team/keras-io/blob/master/examples/generative/vae.py


Version 2, using client and server sockets will be uploaded soon. 
