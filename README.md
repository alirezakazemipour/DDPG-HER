# DDPG + HER
Implementation of the Deep Deterministic Policy Gradient with Hindsight Experience Replay Extension on the MuJoCo's robotic FetchPickAndPlace environment.   
> Visit [vanilla_DDPG](https://github.com/alirezakazemipour/DDPG-her/tree/vanilla_DDPG) branch for the implementation **without the HER extention**.  

## Dependencies  
- gym == 0.17.2  
- matplotlib == 3.1.2  
- mpi4py == 3.0.3  
- mujoco-py == 2.0.2.13  
- numpy == 1.19.1  
- opencv_contrib_python == 3.4.0.12  
- psutil == 5.4.2  
- torch == 1.4.0  

## Installation
```shell
pip3 install -r requirements.txt
```

## Usage
```shell
mpirun -np $(nproc) python3 -u main.py
```
## Demo
<p align="center">
  <img src="Demo/FetchPickAndPlace.gif" height=250>
</p>  


## Result
<p align="center">
  <img src="Result/Fetch_PickandPlace.png" height=400>
</p>

## Reference
1. [_Continuous control with deep reinforcement learning_, Lillicrap et al., 2015](https://arxiv.org/abs/1509.02971)  
2. [_Hindsight Experience Replay_, Andrychowicz et al., 2017](https://arxiv.org/abs/1707.01495)  
3. [_Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research_, Plappert et al., 2018](https://arxiv.org/abs/1802.09464)  
## Acknowledgement
All the credit goes to [@TianhongDai](https://github.com/TianhongDai) for [his simplified implementation](https://github.com/TianhongDai/hindsight-experience-replay) of [the original OpenAI's code](https://github.com/openai/baselines/tree/master/baselines/her).  

