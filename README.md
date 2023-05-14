# Towards-Multi-Staged-One-Shot-Visual-Imitation

**Installation:**
Much of this work is using robosuite. Install mosaic and robosuite from the instructions listed: https://github.com/rll-research/mosaic

The `models` directory has the models that are used for the latent planner's encoder and decoder. 
The `train_latent.py` file has the code used for training the latent planner.

Much of the behavior cloning code is used from MOSAIC since we used the vanilla MOSAIC behavior cloning code to run the bseline experiments as well as use it in combination with our latent planner. 
