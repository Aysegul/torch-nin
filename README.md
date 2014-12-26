torch-NetworkInNetwork
======================

a replication of Network in network in torch. 
original code : https://github.com/mavenlin/cuda-convnet

Learning rate 0.1 is used in the original code, but it doesn't make the learning happen in this script. 
Learning rate is set to 1e-3, and the accuracy gets to %86 not %89.6 :(

Another difference is the original code runs on batches of 128, this is 64.

This script works with the last version of optim/sgd.lua
Please update 
```lua
luarocks install optim
```
