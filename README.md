## Solving Maximum Weighted Matching on Large Graphs with Deep Reinforcement Learning

---

#### Datasets

- Synthetic Datasets, to get implementation from [Networkx](https://networkx.org/documentation/stable/index.html).
    1. [erdos_renyi_graph](https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.erdos_renyi_graph.html#networkx.generators.random_graphs.erdos_renyi_graph)
    2. [barabasi_albert_graph](https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.barabasi_albert_graph.html)
    3. [holme_kim_graph](https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.powerlaw_cluster_graph.html)
    4. [watts_strogatz_graph](https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.watts_strogatz_graph.html)

- Real-world Datasets, to get from [SNAP](https://snap.stanford.edu/data/).  

#### Compare with

- <mark>LwD</mark>@[Learning What to Defer for Maximum Independent Sets](https://github.com/sungsoo-ahn/learning_what_to_defer)
- <mark>DGL-TreeSearch</mark>@[WHAT’S WRONG WITH DEEP LEARNING IN TREE SEARCH FOR COMBINATORIAL OPTIMIZATION](https://github.com/MaxiBoether/mis-benchmark-framework)

#### Running

For training,  

    python run_l2m.py --mode train

For testing,  

    python run_l2m.py --mode test
