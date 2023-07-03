#!/bin/sh

gcc -O3 -fno-tree-vectorize -Wall -Wextra -o nn nn.c -lm