# Design Inbox (Unprocessed Ideas)

This file is a temporary staging area for design ideas before they become ADRs or are discarded.

---

## Item 1
**Date:** YYYY-MM-DD  
**Source:** AI session / intuition / bug / research

### Idea
Short description.

### Context
Why this came up.

### Notes
- Possible direction
- Open questions

### Status
- [ ] Needs ADR
- [ ] Needs exploration
- [ ] Rejected

--- 

## High Level Components 
**Date:** 2026-04-22
**Source:** Trying to solve replay buffer complexity 

### Idea
High level more defined components. Maybe like sub modules. A data folder that has a reusable replay buffer type and replay buffer components, a network folder that has reusable network components, an algorithm folder that has reusable algorithm components, etc. Components no longer mix but are only for their higher level type. 

### Context
I had a similar system before, maybe though something like that plus the DAG validation system would be a good idea. The trouble is a Replay Buffer is very different to an Actor and very different to a MCTS. These have their own optimizations, things like a replay buffer or a MCTS are not best thought of as Nodes or graphs, but more intuitive as entire components (using a strategy pattern perhaps). 

### Notes
- Should we have high level more defined components like a Learner, Actor, Replay Buffer. These are common in RL and from the perspective of an "RL programming language" maybe these are kind of types like an Array or Database. From a Hardware comparison its like saying I have RAM, CPU, GPU, and each of these have their own considerations (components etc), instead of trying to make everything into Nodes/Components. 
- Makes DAG slightly more complex as different high level components have different contracts and ways of executing.
- Components no longer UNIVERSALLY interchangable
- May lose reusability of certain components like action selectors between learner and actor. 
- May need to keep indefinitely adding high level components, wheras composition would allow you to make a graph representing a component. 


### Status
- [ ] Needs ADR
- [ ] Needs exploration
- [ ] Rejected