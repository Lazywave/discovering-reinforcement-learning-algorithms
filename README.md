[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Discovering reinforcement learning algorithms
A jax/stax implementation of the NeurIPS 2020 paper: _Discovering reinforcement learning algorithms_ [[1]](https://proceedings.neurips.cc/paper/2020/file/0b96d81f0494fde5428c7aea243c9157-Paper.pdf)

The agent at `lpg.agent.py` implements the `bsuite.baseline.base.Agent` interface.
The `lpg/environments/*.py` interfaces with a `dm_env.Environment`.
We wrap the [gym-atari](https://github.com/openai/gym) suite using the `bsuite.utils.gym_wrapper.DMEnvFromGym` adapter into a `dqn.AtariEnv` to implement historical observations and actions repeat.


## Installation
To run the algorithm on a GPU, I suggest to [install](https://github.com/google/jax#pip-installation) the gpu version of `jax` [[4]](https://github.com/google/jax). You can then install this repo using [Anaconda python](https://www.anaconda.com/products/individual) and [pip](https://pip.pypa.io/en/stable/installing/).
```sh
conda env create -n lpg
conda activate lpg
pip install git+https://github.com/epignatelli/discovering-reinforcement-learning-algorithms
```

## Note from Student
Pip installing the github link from above will install all requirements you need to run my student.ipynb.

1. To run my experiments you only need to run the code cells in the student.ipynb after pip installing said repo. (Depending on the device the visualizations/rendering might not work. It only works for one of my workstations)
2. a)   Source code (all *.py files) is from https://github.com/epignatelli/discovering-reinforcement-learning-algorithms
b) I've not touched the source code, only read it and applied it.
c) As stated at the top of the student.ipynb, every piece of code that I wrote is in that file.
3. As this was a RL agent, I didn't use "datasets" on it, but instead had different environments, extensively described in my paper.


## References
[1] [_Oh, J., Hessel, M., Czarnecki, W.M., Xu, Z., van Hasselt, H.P., Singh, S. and Silver, D., 2020. Discovering reinforcement learning algorithms. Advances in Neural Information Processing Systems, 33._](https://proceedings.neurips.cc/paper/2020/file/0b96d81f0494fde5428c7aea243c9157-Paper.pdf)
