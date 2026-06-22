# Module 2: Reinforcement Learning-Based Sybil Attack Simulation

This module demonstrates a cyberattack scenario in connected transportation systems using SUMO, TraCI, Python, and reinforcement learning. The focus of this module is a Sybil attack, where an attacker creates fake vehicle identities or fake traffic information to mislead an adaptive traffic signal control system.

In this simulation, the reinforcement learning-based attacker learns how to inject fake traffic information into the environment. The goal of the attacker is to influence the traffic signal controller and create congestion, delay, or inefficient signal decisions.

## Overview

Connected transportation systems rely on vehicle information, sensor data, and communication between vehicles and infrastructure. If an attacker creates fake vehicle identities, the traffic control system may believe that there are more vehicles on a road than there actually are.

This can cause the signal controller to make poor decisions, such as giving more green time to a direction with fake congestion while real vehicles in other directions experience delays.

This module helps demonstrate how intelligent cyberattacks can affect traffic cyber-physical systems and why robust defense mechanisms are important for connected and automated transportation networks.

## Main Components

This module includes:

* SUMO traffic simulation files
* Route and traffic demand files
* SUMO configuration file
* Sybil attack environment using TraCI
* Reinforcement learning-based attacker agent
* Training and evaluation scripts
* Attack impact analysis on traffic performance

## Repository Structure

```text
Module_2_RL_SybilAttack/
│
├── nodes.xml
├── edges.xml
├── routes.xml
├── sumocfg.xml
├── gui_settings.xml
├── sybil_env.py
├── attack_agent.py
└── RUN.sh
```

## Objective

The main objective of this module is to understand how an RL-based attacker can learn attack strategies in a traffic simulation environment.

The attacker learns how to:

* Observe the traffic state
* Select an attack action
* Inject fake vehicle information
* Increase traffic delay or congestion
* Mislead the adaptive traffic signal controller

By testing this attack scenario, the module provides a basic foundation for studying cyber-resilience, attack detection, and defense mechanisms in intelligent transportation systems.

## Reinforcement Learning Setup

The RL setup contains the following basic elements:

### State

The state represents the traffic condition observed by the attacker. This may include traffic density, vehicle queues, signal phase information, or other simulation-based traffic features.

### Action

The action represents the attack decision selected by the RL attacker. For example, the attacker may decide how many fake vehicles or fake reports should be injected into the system.

### Reward

The reward tells the attacker whether the attack was successful. A possible reward can be based on increasing vehicle waiting time, increasing congestion, or reducing traffic signal efficiency.

### Agent

The agent is a reinforcement learning model that learns which attack actions are most effective under different traffic conditions. Over multiple training episodes, the attacker improves its strategy.

## Requirements

Make sure the following tools and libraries are installed:

* Python 3.8 or higher
* SUMO
* TraCI
* PyTorch
* NumPy

Install Python dependencies using:

```bash
pip install -r requirements.txt
```

## How to Run

Move into the Module 2 folder:

```bash
cd Module_2_RL_SybilAttack
```

Generate the SUMO network file:

```bash
netconvert -n nodes.xml -e edges.xml -o intersection.net.xml
```

Train the RL-based Sybil attacker:

```bash
python attack_agent.py --train --episodes 20
```

Evaluate the trained attacker with SUMO GUI:

```bash
python attack_agent.py --eval --model sybil_attacker.pth --gui
```

## Expected Output

After training, the RL attacker learns an attack policy that attempts to disrupt traffic signal control. During evaluation, the trained attacker interacts with the SUMO environment and selects attack actions based on the observed traffic state.

The output may include:

* Training reward values
* Episode-level attack performance
* Trained attacker model file
* Traffic delay or congestion changes
* SUMO GUI visualization during evaluation



