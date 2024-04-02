# Advanced Cognitive Modeling: Portfolio Assignment 2

## Modelling the Matching Pennies Game

Julia implementation with `Julia v1.10.2`. Using Rescorla-Wagner model to simulate the game and results.

### Model and Simulation

The model is implemented and simulated in [`RWCoins.jl`](./RWCoins.jl).

Simulations were done for single subjects with randomly sampled values for $α$ and $τ$. 100 simulations were done for each of the following number of trials:

 - 10
 - 25
 - 50
 - 75
 - 100
 - 150
 - 200
 - 250
 - 500
 - 1000

### Some Plots


Summary boxplot of parameter recovery

![Summary boxplot of parameter recovery](../../out/parameter_recovery_boxplot.png)

Example chains for $α$ and $τ$ for 10 trials

![Example chains for alpha and tau for 10 trials](../../out/posterior_1-50.png)

Example chains for $α$ and $τ$ for 1000 trials

![Example chains for alpha and tau for 1000 trials](../../out/posterior_10-50.png)


### True vs Posterior Mode

| Trials | Alpha | Tau |
|---:|---|---|
| 10 | ![True vs Mode alpha for 10 trials](../../out/true_vs_mode_alpha_N10.png) | ![True vs Mode alpha for 1000 trials](../../out/true_vs_mode_tau_N10.png) |
| 25 | ![True vs Mode alpha for 25 trials](../../out/true_vs_mode_alpha_N25.png) | ![True vs Mode alpha for 1000 trials](../../out/true_vs_mode_tau_N25.png) |
| 50 | ![True vs Mode alpha for 50 trials](../../out/true_vs_mode_alpha_N50.png) | ![True vs Mode alpha for 1000 trials](../../out/true_vs_mode_tau_N50.png) |
| 75 | ![True vs Mode alpha for 75 trials](../../out/true_vs_mode_alpha_N75.png) | ![True vs Mode alpha for 1000 trials](../../out/true_vs_mode_tau_N75.png) |
| 100 | ![True vs Mode alpha for 100 trials](../../out/true_vs_mode_alpha_N100.png) | ![True vs Mode alpha for 1000 trials](../../out/true_vs_mode_tau_N100.png) |
| 150 | ![True vs Mode alpha for 150 trials](../../out/true_vs_mode_alpha_N150.png) | ![True vs Mode alpha for 1000 trials](../../out/true_vs_mode_tau_N150.png) |
| 200 | ![True vs Mode alpha for 200 trials](../../out/true_vs_mode_alpha_N200.png) | ![True vs Mode alpha for 1000 trials](../../out/true_vs_mode_tau_N200.png) |
| 250 | ![True vs Mode alpha for 250 trials](../../out/true_vs_mode_alpha_N250.png) | ![True vs Mode alpha for 1000 trials](../../out/true_vs_mode_tau_N250.png) |
| 500 | ![True vs Mode alpha for 500 trials](../../out/true_vs_mode_alpha_N500.png) | ![True vs Mode alpha for 1000 trials](../../out/true_vs_mode_tau_N500.png) |
| 1000 | ![True vs Mode alpha for 1000 trials](../../out/true_vs_mode_alpha_N1000.png) | ![True vs Mode alpha for 1000 trials](../../out/true_vs_mode_tau_N1000.png) |

