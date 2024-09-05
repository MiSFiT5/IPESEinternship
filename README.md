# IPESEinternship

The internship project in EPFL IPESE lab

Author: Heyuan Liu M2 MSc&T AiViC student Email: heyuan.liu@polytechnique.edu 

Title: Identify optimal configuration with a machine learning method in multi-criteria decision analysis

## Shape Algorithms for Typical Solution Identifications
### To Be Completed
Data Format is ruled and we are considering to add a small scale of data to it. So currently there is no valid dataset in any folder, but coming soon...


![Methodology for Typical Solution Identifications](IMGforMD/Progress.png)


In the Folder of 16000Experiment, There are several Python files which are components of the Typical Solution Selectors. 

Including functions for
  - ISS 
  - Mesh Saliency
  - Harris Corner Detector in 3D

And you could find the visualization functions in Visualization.ipynb, which is a Jupyter Notebook for a better visualizations in data sicence.

Before You get started, please make sure about that you have the Python environment on your computer. This part is developed under Python 3.7.8, and no clear differences with higher edition of the Python.

### Install necessary packages 
Go to the Folder 16000Experiment and run the following command in Terminal

```bash
pip install -r requirements.txt
```
### Give it a GO
With the Environment established, 

RUN in Ternimal
```bash
python main.py
```

Remeber to set up the preferences of the iteration numbers. Actually the points selected by each iteration could be considered as Typical, because of the redundancy of the repeated configurations and super close distributions.

## LLM aided Decision Making

Organizing will be deployed soon.

![Methodology for LLM Solution Identifications](IMGforMD/LLM_progress.png)

First, make sure about you have a OpenAI API key.

![Self-iterative LLM aided Decision Making](IMGforMD/LLMinteraction.png)

## Reinforcement Learning for Real-Time Multi-Objective Control

Stored in the RL.ipynb.

It still a raw code file, working on how to make it more clear.

The Multi-head Attention Mechanism and Shared Layer + Individual Layer Design within DDPG framework safeguards the configuration units control with changing market conditions and reach zero violations
![DDPG framework for Multi-Objective Control](IMGforMD/DDPGwithattention.svg)
