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
