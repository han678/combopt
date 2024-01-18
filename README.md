This is the code for "Combinatorial optimization for low bit-width neural networks"(https://arxiv.org/pdf/2206.02006.pdf).
#### Single layer network
The following examples generate Figure 3 (a) binary weights and (b) ternary weights in our paper.
```bash
cd singleLayer
python demo1.py
python demo2.py
```
#### Experiments for two-layer networks and MLP models
```bash
cd mlp
python binary_opt.py --opt_strategy
```
'''opt_strategy''' can be greedy or submodular.

