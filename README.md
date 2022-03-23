# Object-tracking-with-Flownet

This project aims to employ optical flow based algorithm to perform object tracking task. The dataset consists of many sequences. Each sequence presents one moving object that is needed to be tracked, and its corresponded masks as the ground truth.

We benmark 3 approaches: Flowner, Deepflow, and Farneback.

## 1. Flownet

Flownet is a deep learning based approach presented in the paper: [flownet](https://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/)
We adopt a pre-trained flownetS into our project for object tracking. The pre-trained weight can be download at: [weight](https://drive.google.com/drive/folders/16eo3p9dO_vmssxRoZCmWkTpNjKRzJzn5). This model is trained on FlyingChair dataset, which contains a huge number of synthesized images (64GB). 
In terms of implementation, we reuse the implementation of [ClementPinard](https://github.com/ClementPinard/FlowNetPytorch)

Since we do not retrain again on our dataset, several pre-processing techniques are employed to increase the accuracy of the tracking. Some of them are Gaussian blurring and Detail Enhancement.

## 2. Deepflow

Deepflow is an optical flow estimation algorithm that aims to handle large displacement problem. It was presented in the paper: [Deepflow](https://hal.inria.fr/hal-00873592/document). 
The main idea is to introduce another term called matching term beside two original terms in traditional optical flow estimation: data term and flow smoothing term.
The matching term benefits from the keypoint matching algorithm. Briefly, the traditional Horn-Schunk optical flow estimation relies on two main hypothese: brightness constancy and small motion.



