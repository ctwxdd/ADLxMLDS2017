agent_dir/agent_dqn.py -- dqn
agent_dir/agent_ddqn.py -- double dqn
agent_dir/agent_dqn_dueling.py -- double dqn + dueling
agent_dir/agent_pg_ori.py -- dense network
agent_dir/agent_pg --a3c agent



# ADL HW3
Please don't revise test.py, environment.py, agent_dir/agent.py

## Installation
Type the following command to install OpenAI Gym Atari environment.

`$ pip3 install opencv-python gym gym[atari]`

Please refer to [OpenAI's page](https://github.com/openai/gym) if you have any problem while installing.

## How to run :
training policy gradient:
* `$ python3 main.py --train_pg`

testing policy gradient:
* `$ python3 test.py --test_pg`

training DQN:
* `$ python3 main.py --train_dqn`

testing DQN:
* `$ python3 test.py --test_dqn`
