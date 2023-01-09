# Multi-Dueling Bandits with Contextual Information

This repository contains Multi-Dueling Bandit algorithms with Contextual Information. This branch of the repository is an implementation of the experimentation on the Explore-Exploit strategy on the UCB and CoLSTIM bandit algorithms with contextual information to be used in the CPPL Algorithm Configurator.

The experiments are ran on the Noctua 2 cluster provided by PC $^2$ from Paderborn University. To run the experiments, we used SAT problem instances for SAPS solver and arbitrary bids version of CA for the CPLEX solver. 

To run the framework, you need to have an access to the Noctua 2 cluster. Make sure you have Python3 installed in your cluster. Clone the repository in your cluster folder. To submit the jobs on the cluster, you need to run the job scripts named 

```
script_parameter_setup_<name-of-the-algorithms>_<dataset>.sh
```

The experiment is actually executed through `test_cluster.py` file.
You can use the following command
```
python3 test_cluster.py --help
```
to get the list of all the arguments used as the input to the CPPL framework.