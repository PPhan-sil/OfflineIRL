# Ranking Games Meet Generative World Models for Offline Inverse Reinforcement Learning

## Abstract from:
### [Ranking Games Meet Generative World Models for Offline Inverse Reinforcement Learning](https://pphan-sil.github.io/Portfolio/offlineIRL/paper.pdf)

> The motivation for Offline Inverse Reinforcement Learning (Offline IRL) is to discern the underlying reward structure and environmental dynamics from a fixed dataset of previously collected experiences. For safety-sensitive applications this paradigm becomes a topic of interest because interacting with the environment may not be possible. Therefore accurate models of the world becomes crucial to avoid compounding errors in estimated rewards. With limited demonstrations data of varying expertise, it also becomes important to be able to extrapolate from beyond these demonstrations in order  to infer high-quality reward functions. We introduce a bi-level optimization approach for offline IRL which accounts for uncertainty in an estimated world model and uses a ranking loss to encourage learning from intent. We demonstrate the algorithm can match state-of-the-art offline IRL frameworks over the continuous control tasks in MuJoCo and different datasets in the D4RL benchmark.

## Installation
- PyTorch 1.13.1
- MuJoCo 2.1.0
- pip install -r requirements.txt


## File Structure
- Experiment result ：`data/`
- Configurations: `args_yml/`
- Expert Demonstrations: `expert_data/`

## Instructions
- All the experiments are to be run under the root folder.
- After running, you will see the training logs in `data/` folder.

## Experiments
All the commands below are also provided in `run.sh`.

### Offline-IRL benchmark (MuJoCo)
Before experiment, you can download our expert demonstrations and our trained world model [here](https://drive.google.com/drive/folders/1BbEZLEKP6HAijeRBXG0V3JLSrB0FIQg6?usp=drive_link).

```bash
python train.py --yaml_file args_yml/model_base_IRL/halfcheetah_v2_medrep.yml --seed 1 --uuid halfcheetah_test1
```
also you can use:
```bash
./run.sh
```
