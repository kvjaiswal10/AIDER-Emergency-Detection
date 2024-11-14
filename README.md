# AIDER-Emergency-Detection

<h3>Dataset</h3>
Dataset obtained from <a href="https://github.com/ckyrkou/AIDER?tab=readme-ov-file">this link</a> (AIDER dataset).

<h3>Model</h3>
A deep convolutional neural network called EmergencyNet is trained on this dataset referencing <a href="https://github.com/ckyrkou/EmergencyNet">this repo</a>
To build efficient networks for devices with limited resources (like drones), the goal is to design lightweight models that can still perform well. 

<u>Efficient Design with Atrous Convolutional Feature Fusion (ACFF):</u>
The model uses a special type of layer called an ACFF block to gather and combine information from different parts of an image without increasing the model size.
This ACFF block uses a technique called atrous (or dilated) convolution. Atrous convolutions capture details from various scales (like zooming in and out) by adjusting the spacing of the convolution operation.

<u>Depth-Wise Convolution:</u>
Each atrous convolution is broken down into smaller parts called depth-wise convolutions. This approach applies one filter per channel, making it lighter on computation.

<u>Reducing Feature Map Size:</u>
To keep the model efficient, the number of channels in the input data is cut in half before applying atrous convolutions. This lets the model use multiple atrous convolution layers without becoming too large.

<u>Combining Multi-Scale Features:</u>
Each atrous convolution captures information at a different scale. All these features are then combined so that the model can learn from a wider area of the image. After combining these features, a small 1x1 convolution is applied to mix the channels and create a richer, more complex representation. This 1x1 convolution also reduces complexity, keeping the model efficient.

