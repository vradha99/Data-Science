#!/bin/bash                                                                     

g++ myfinal.cpp -o output `pkg-config --cflags --libs opencv` 
 
