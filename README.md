## Visual Tracking with Fully Convolutional Networks

### Introduction
FCNT is an online visual tracking algorithm using fully convolutional neural networks. This package contains the source code to reproduce the experimental results of FCNT reported in our [ICCV 2015 paper](http://202.118.75.4/lu/Paper/ICCV2015/iccv15_lijun.pdf). The source code is mainly written in MATLAB.

### Usage

* Supported OS: the source code was tested on 64-bit Arch Linux OS, and it should also be executable in other linux distributions.

* Dependencies: 
 * Deep learning framework [caffe](http://caffe.berkeleyvision.org/) and all its dependencies. 
 * Cuda enabled GPUs

* Installation: 
 1. Install caffe-fcnt: caffe-fcnt is our customized version of the original caffe. Change directory into ./caffe-fcnt and compile the source code and the matlab interface following the [installation instruction of caffe](http://caffe.berkeleyvision.org/installation.html).
 2. Download the 16-layer VGG network from https://gist.github.com/ksimonyan/211839e770f7b538e2d8, and put the caffemodel file under the ./feature_model directory.
 3. Run the demo code run.m. You can customize your own test sequences following this example.

### Citing Our Work

If you find FCNT useful in your research, please consider to cite our paper:

        @inproceedings{ wang2015deep,
           title={Visual Tracking with Fully Convolutional Networks},
           author={Wang, Lijun and Ouyang, Wanli and Wang, Xiaogang and Lu, Huchuan},
           booktitle={IEEE International Conference on Computer Vision (ICCV)},
           year={2015}
        }

### Liscense

        Copyright (c) 2015, Lijun Wang
        All rights reserved. 
        Redistribution and use in source and binary forms, with or without modification, are 
        permitted provided that the following conditions are met:
    		* Redistributions of source code must retain the above copyright 
      		  notice, this list of conditions and the following disclaimer.
    		* Redistributions in binary form must reproduce the above copyright 
      		  notice, this list of conditions and the following disclaimer in 
      		  the documentation and/or other materials provided with the distribution
        
        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
        AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
        IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
        ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 	
        LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
        CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
        SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
        INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
        CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
        ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
        POSSIBILITY OF SUCH DAMAGE.
