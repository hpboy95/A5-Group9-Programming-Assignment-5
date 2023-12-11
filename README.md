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
| Observation | Direction                        |
|-------------|----------------------------------|
| 0           | Price to enter the priority lane |
| 0           | Price to enter the priority lane |
| 0           | Price to enter the priority lane |
| 0           | Price to enter the priority lane |

## Rewards
<!-- See the Cart Pole Env example https://gymnasium.farama.org/environments/classic_control/cart_pole/ -->

## RL Algorithm 

## Starting State [if applicable]
<!-- See the Cart Pole Env example https://gymnasium.farama.org/environments/classic_control/cart_pole/ -->

## Episode End [if applicable]
<!-- See the Cart Pole Env example https://gymnasium.farama.org/environments/classic_control/cart_pole/ -->

## Results

