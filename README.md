# D-PSGD
Algorithm: Decentralized Parallel Stochastic Gradient Descent   
* Follow paper [Lian X, Zhang C, Zhang H, et al. Can decentralized algorithms outperform centralized algorithms? a case study for decentralized parallel stochastic gradient descent[C]//Advances in Neural Information Processing Systems. 2017: 5330-5340.]  
## Training
```bash
mpirun -n 5 --hostfile hosts python PSGD.py --epochs 160 --lr 0.5
```

