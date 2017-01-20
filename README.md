# Explain output of image classifier

VGG-16 [1] trained on  ILSVRC-2014 data [2].
The algorithm used for identifying the sub-images that gets the highest probability for
the top class is inspired by [3].

Weights for VGG16 can be downloaded here: [vgg16_weights.h](https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing)

Usage
```
python explain.py -i dog.jpg -m vgg16_weights.h5 -o out.jpg
```

### Input image:
![Image of dog playing guitar](dog.jpg)
### Output image:
![Image of dog playing guitar, only showing dog](out.jpg)


[1] Very Deep Convolutional Networks for Large-Scale Image Recognition
K. Simonyan, A. Zisserman
arXiv:1409.1556

[2] http://image-net.org/challenges/LSVRC/2014/

[3] "Why Should I Trust You?": Explaining the Predictions of Any Classifier
Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin
arXiv:1602.04938
