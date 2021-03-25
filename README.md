# OperatorsOptim
Optimize gemm on Raspi-4B

CPU: BCM2711

RAM: 8G

OS: Ubuntu server 20.04

This version impl a 4x4 kernel, with a matrix 1024x1024 costs about 2098ms (asm4x4-v1)

This version impl a 16x16 kernel, with a matrix 1024x1024 costs about 1077ms (asm16x16-v0)
