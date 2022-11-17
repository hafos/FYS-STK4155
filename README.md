# FYS-STK4155
Our group consists of Simon Hille, Semya Tønnessen and Oskar Hafstad.

------------------------
PROJECT 1:
We have an extended deadline until monday 17/10/2022 and will continue to update this repo until midnight on the 17th of October.

The changes after midnight was mainly clean up removing files that should not be in repo. We also sadly found a last minute bug that prevent the code from running, which we had to fix after midnight. In addition to that we added back the figures folder with figures in it. We were unaware that we could include figures in the repo thats why we removed it in the first place. To produce plots run LinearRegression.py for Franke function analysis and real_data.py for Terrain data analysis.

------------------------
PROJECT 2:
We again have a extendet deadline

In the folder scrips you find all scripts used to generate outputs we used / that were demanded.
In the following we will briefly summarize what each script does:

compare_time.py: (used data: Francke function)
  Prints out the time per epochs and the time per operation for GD and SGD for a constant learningrate
 
compare_gd_sdg.py: (used data: Francke function)
  Generates two plots for a constant learningrate to compare GD with SGD
  The first is a plot were for two different learningrates the MSE i plotted against the number of epochs.
  The second is a plot were for for different number of batches, the MSE is plotted against the number of epochs.

sgd_lr_batches.py: (used data: Francke function)
  Generates a headmap, were the MSE is plotted for different learningrates and number of batches.
  
  
  
