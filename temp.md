how to combine serval grad together?

Assume we have:

g1 from sampled data, using the state and actor and critic calced loss

g2 from auged data using the next_state and actor critic calced loss

ques: g2 the auged data not used in the actor loss, how to tackle it? The core problem lies in the prdc not using the date tuple to update actor.

solution:

1. not using the auged data to update actor

g3 from dc loss using the sampled state

g4 from dc loss using the auged data or rather the state_next

2. the dc loss are applied to the online fintuning process.

Once we 