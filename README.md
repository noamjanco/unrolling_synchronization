# Unrolled algorithms for group synchronization
This is the python & TensorFlow implementation of the experiments conducted in "Unrolled algorithms for group synchronization" (https://arxiv.org/abs/2207.09418).

In this work we developed unrolled algorithms for several group synchronization instances, including synchronization over the group of 3-D rotations: the synchronization problem in cryo-EM. We also apply a similar approach to the multi-reference alignment problem. We show by numerical experiments that the unrolling strategy outperforms existing synchronization algorithms in a wide variety of scenarios.

![unrolling_synchronization](https://github.com/noamjanco/unrolling_synchronization/assets/82662498/ec5376dc-d005-4fde-8b76-1da2f29431a9)

# Usage
The different experiments presented in the paper are under the experiments folder.
For instance, run compare_so3_unrolling.py to reconsturct the results of unrolled synchronization for the SO(3) case, which shows the average error per iteration or depth.
Run compare_so3_unrolling_vs_snr.py to reconstruct the results of unrolled synchronization for the SO(3) case, which shows the average error per SNR, for a given depth.

# Results
Results for SO(3) experiments that shows the average error vs. the number of iterations / depth.
![unrolling_synchronization-so3_results](https://github.com/noamjanco/unrolling_synchronization/assets/82662498/c70e6992-5d4a-483c-819f-f20276fdea36)
Results for SO(3) experiments that shows the average error vs. SNR.
![unrolling_synchronization-so3_results_per_snr](https://github.com/noamjanco/unrolling_synchronization/assets/82662498/a1ca44a9-8094-46d3-b7fa-7118d2c2db18)
