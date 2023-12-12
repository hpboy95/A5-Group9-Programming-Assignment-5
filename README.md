# Project Name
## Project Summary
<!-- Around 200 Words -->
<!-- Cover (1) What problem you are solving, (2) Who will use this RL module and be happy with the learning, and (3) a brief description of the results -->

I was inspired by the board games that I play. I enjoy playing wargames and I wanted to model an environment where an 
agent could learn to play against small squad tactics. However, multi-agent automata is well beyond the scope of this class.
Therefore instead I modeled an environment where an agent would have to learn to avoid an opposing set of agents and reach 
a goal area. The hope is that the agent would learn the best was to move as to avoid interaction with the hostile agents.

A real game of infinity (the wargame I play) would involve multiple agents on either team and would allow for both teams
to have an action phase to deal damage to the enemy. However, to simply the problem and the game, I have modeled the
environment to consist of one enemy agent and a goal. I initially tried to include a building, but this proved too difficult
for the amount of time effort I was able to put into this project. At each step the enemy agent will move forward in the direction
and then check if the player agent is within 2 tiles. If it is within 2 tiles the player agent dies. 


## State Space
<!-- See the Cart Pole Env example https://gymnasium.farama.org/environments/classic_control/cart_pole/ -->


| Observation     | Description                                    | Min | Max    | Notes                                |
|-----------------|------------------------------------------------|-----|--------|--------------------------------------|
| Agent           | The current position of the agent              | 0   | size-1 | For both x,y                         | 
| Target          | The current position of the target             | 0   | size-1 | For both x,y                         |
| Enemy           | The current position of the enemy              | 0   | size-1 | For both x,y                         |
| Enemy_Direction | The current direction that the enemy is facing | 0   | 3      | Same direction space as player agent |

## Action Space
<!-- See the Cart Pole Env example https://gymnasium.farama.org/environments/classic_control/cart_pole/ -->
| Observation | Direction |
|-------------|-----------|
| 0           | Right     |
| 1           | Up        |
| 2           | Left      |
| 3           | Down      |

## Rewards
<!-- See the Cart Pole Env example https://gymnasium.farama.org/environments/classic_control/cart_pole/ -->
For the game of infinity the worst case scenario is losing a unit, so I set that case to have a reward of -100
Every move taken is worth -1. There is a reward of 1 if the end goal is reached successfully.

## RL Algorithm 
I used the PPO algorithm obtained with Ray lib.

## Starting State [if applicable]
<!-- See the Cart Pole Env example https://gymnasium.farama.org/environments/classic_control/cart_pole/ -->

The map is randomly generated. There is a fixed square size for this iteration of grid world.
The agent's starting position is random. The goal is position is randomly generated and adjusted if the position is the
same as the agent. The enemy's position and starting direction is random.

## Episode End [if applicable]
<!-- See the Cart Pole Env example https://gymnasium.farama.org/environments/classic_control/cart_pole/ -->

The episode ends if the playing agent is captured or the goal is reached.

## Results

Please Refer to the "Results.png". In this graph the following is the legend:
Green: The current number of positive rewards
Red: The current number of negative rewards
Orange: The average reward
Blue:The truescale rewards

As can be seen from this graph, the rate at which positive rewards acrues much faster than the negative rewards (aka the
green line has a higher slope than the red line). Initially when I graphed the average this increase was not clearly apparent, 
it seemed as though the average was stagnant through out. However, when I looked at the positive returns compared to the
negative returns I can clearly see that PPO is improving over time. However, the average was clearly brought down by my 
very low capture reward. 

Over time the agent got better and better at avoiding the enemy agent (as is evidenced, by the low number of negative 
spikes on the right hand side of the plot). However, since the enemy is a random agent with a random spawn it would be 
much better if enemy spawn logic was much more strategic and game accurate. Furthermore, I would like to improve upon 
this by adding additional enemies and obstacles to see how the agent would fare in that environment. Also I could add
shooting logic to see how that would affect the aggressiveness of the agent.

I believe that was able to grasp the basics of creating an openAI gym environment, as well as use the PPO algorithm we 
learned towards the end of the class. Hopefully this last assignment with what I have done, and what could be done 
demonstrates that. Thanks for teaching this semester  I appreciate what I have learned, and I hope to apply it to my 
future classes and work ^.^!

