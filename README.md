# MCTS-VRP: Monte Carlo Tree Search for Dynamic Vehicle Routing

This project implements a Monte Carlo Tree Search (MCTS) based solution for a **Dynamic Vehicle Routing Problem (DVRP)** with time windows and stochastic trip requests. The system supports both greedy and MCTS-based solvers and includes a neural network-based generative model to simulate future demand.

## 🚀 Features
- Generative Model: Neural network generative model for future request simulation
- Greedy baseline for comparison
- MCTS-based decision-making framework for VRP
- Visualization tools for request distributions and evaluation

---

## 📁 Project Structure

```bash
.
├── main.py                          # Entry point to run the solver
├── models/
│   ├── data_loader.py              # Loads request and travel time data
│   ├── generative_model.py        # Trains and samples from the generative model
│   ├── mdp.py                      # Defines vehicle and environment state
├── solvers/
│   ├── greedy_solver.py           # Greedy dispatch baseline
│   ├── mcts_solver.py             # MCTS rollout policy with scoring
├── evaluation/
│   ├── request_distribution.py     # Visualization of request distributions
│   ├── greedy_evaluation.py        # Evaluation plots for greedy solution
├── data/                           # Folder containing daily request data

## Code Running
Simply run python main.py, it includes Data Loading, Generative Model Training (NN), Greedy Algorithm run, MCTS run. 


```

## Meet Preliminary Exam specifications by
1. Building a full pipeline to solve VRP problem
2. Implement a generative model (I tried Neural Network first, but it cause low traning accuracy), then I swithced to Random Forest to sample requests
3. Evaluate the sampled requests by plotting the location distribution, and histogram of the number of the demands (hourly) for days
4. Implement a greedy algorithm which does not consider future incoming requests, but only focus on the current demands
5. Implement MCTS together with the generative model, so it can take advantage of the simulated/sampled future request while it determines optimal actions for each demand.

However, the generative model is not able to detect the spatial information of historical request data very well (which can definitely be improved in the future), even though it can perfectly learn the pattern of number of demands per hour in a day. As aresult, our MCST-based algorithm performance got beaten by the greedy algorithm (~50% VS ~70% in terms of average successful service rate).
In conclusion, this work builds a fundamental framwork of how researchers solve a VRP problem, which usually includes components, such as an offline learner of historical data, an online solver to determine optimal routes for action space, and evaluations modules to validate the results.
