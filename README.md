# Object-tracking-with-Flownet

This project aims to employ optical flow based algorithms to perform object tracking task. The dataset consists of 9 sequences. Each sequence presents one moving object that is needed to be tracked, and its corresponded masks as the ground truth. 

The object is tracked by propagating the given mask of the **reference frame** to the current frame via *to-the-reference* optical flow estimation strategy. 
The flow between the current frame and the reference one can be computed via 2 integration methods: *direct* or *sequential*.

Then, the pipeline to track an object is composed of: 

1. Load the mask of the reference frame (usually the first frame in the sequence)
2. Computing the optical flow between
    - the current frame *t* and the reference frame (in direct integration), or 
    - the current frame *t* and the previous frame *t-1* then concatenate the computed flow to the previously integrated flow between frame *t-1* and reference frame (in sequential integration)
3. The mask of the current frame is the mask of the reference frame propagated to the current frame via the estimated flow between it and the reference frame.

For optical flow estimation, we benchmark 3 approaches: **FlowNet**, **Deepflow**, and **Farneback**.

## 1. Flownet

Flownet is a deep learning based approach presented in the paper: [FlowNet: Learning Optical Flow with Convolutional Networks](https://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/).

We adopt a pre-trained FlowNetS into our project for object tracking. The pre-trained weight can be download at: [weight](https://drive.google.com/drive/folders/16eo3p9dO_vmssxRoZCmWkTpNjKRzJzn5). This model is trained on FlyingChair dataset, which contains a huge number of synthesized images (64GB). 

In terms of implementation, we reuse the implementation of [@ClementPinard](https://github.com/ClementPinard/FlowNetPytorch).

### Input processing

Since we do not retrain again on our dataset, several pre-processing techniques, including Gaussian blurring, Detail enhancing, Edge preserving and Pencil sketching are employed to increase the accuracy of the tracking. The effect of each pre-processing on the input image is shown below:
![preprocessed_inputs](images/flownet_preprocessing.png)

### Results

The tracking results based on the optical flow estimated by FlowNetS on pre-processed as well as original inputs are demonstrated in the following figure:
![preprocessed_input_results_on_swan_sequence](images/flownet_preprocessing_1.png)

![preprocessed_input_results_on_swan_sequence](images/flownet_preprocessing_2.png)

## 2. Deepflow

Deepflow is an optical flow estimation algorithm that aims to handle large displacement problem. It was presented in the paper: [Deepflow](https://hal.inria.fr/hal-00873592/document).

The main idea is to introduce another term called matching term beside two original terms in traditional optical flow estimation: data term and flow smoothing term.
The matching term benefits from the keypoint matching algorithm. Briefly, the traditional Horn-Schunk optical flow estimation relies on two main hypothese: brightness constancy and small motion.



