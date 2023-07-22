# Unrolled algorithms for group synchronization
This is the python & TensorFlow implementation of the experiments conducted in "Unrolled algorithms for group synchronization" (https://arxiv.org/abs/2207.09418).

In this work we developed unrolled algorithms for several group synchronization instances, including synchronization over the group of 3-D rotations: the synchronization problem in cryo-EM. We also apply a similar approach to the multi-reference alignment problem. We show by numerical experiments that the unrolling strategy outperforms existing synchronization algorithms in a wide variety of scenarios.

![unrolling_synchronization](https://github.com/noamjanco/unrolling_synchronization/assets/82662498/ec5376dc-d005-4fde-8b76-1da2f29431a9)

The different experiments presented in the paper are under the experiments folder.
For instance, run compare_so3_unrolling.py to reconsturct the results of unrolled synchronization for the SO(3) case.
