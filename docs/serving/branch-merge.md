(branch-merge)=
# Branching and merging steps

You can define a graph that branches into two parallel steps, and the output of both steps merge back together. 

In this basic example, all input goes into both stepA and stepB, and then both stepA and stepB forward the input to stepC. 
This means that a dataset of 5 rows generates an output of 10 rows (barring any filtering or other processing that 
would change the number of rows).

```{admonition} Note
Use this configuration to join the graph branches and **not** to join the events into a single large one.
``` 

Graphs that split and rejoin can also be used for these types of scenarios:
- Steps B and C are filter steps that complement each other. For example B passes events where key < X, and C passes events where key >= X. The resulting DF contains the exact event ingested, since each event was handled once on one of the branches.
- Steps B and C modify the content of the event in different ways. B adds a column col1 with value X, and C adds a column col2 with value X. The resulting DF contains both col1 and col2. Each key is represented twice: once with col1 == X, col2 == null and once with col1 == null, col2 == X.

Example:
```
graph.to("stepB")
graph.to("stepC")
graph.add_step(name="stepD", after=["stepB", "stepC"])


graph = fn.set_topology("flow", exist_ok=True)
dbl = graph.to(name="double", handler="double")
dbl.to(name="add3", class_name="Adder", add=3)
dbl.to(name="add2", class_name="Adder", add=2)
graph.add_step("Gather").after("add2", "add3")
```