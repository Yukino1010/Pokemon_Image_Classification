# Pokemon_Image_Classification
<p align="center"><img width="200px" src="https://github.com/Yukino1010/Pokemon_Image_Classification/blob/master/centered-sprites/gen05_black-white/644.png" />&emsp;&emsp;&emsp;&emsp;&emsp;<img width="200px" src="https://github.com/Yukino1010/Pokemon_Image_Classification/blob/master/centered-sprites/gen05_black-white/643.png" /></p>
<br>
<p align="center"><img width="200px" src="https://github.com/Yukino1010/Pokemon_Image_Classification/blob/master/centered-sprites/gen05_black-white/483.png" />&emsp;&emsp;&emsp;&emsp;&emsp;<img width="200px" src="https://github.com/Yukino1010/Pokemon_Image_Classification/blob/master/centered-sprites/gen05_black-white/484.png" /></p>

## Introduce


## Network Structure

**architecture**  

![image](https://github.com/Yukino1010/Pokemon_Image_Classification/blob/master/model/structure.png)

### network design
- use structure similar to AlexNet but reduce the filter of each layers
- add AveragePooling behind the last Conv2d to reduce the parameter of Dense

## Loss and Accuracy

![image](https://github.com/Yukino1010/Pokemon_Image_Classification/blob/master/model/structure.png)

At the begining I use the origin data to train my model.Although I got 60% accuracy on testing data ,but it was obviously overfitting to the training data,
so I consider to do data augumentation to slove this problem.

![image](https://github.com/Yukino1010/Pokemon_Image_Classification/blob/master/model/structure.png)

From the results of the picture, it seems that it can indeed solve the overfitting problem, but at the same time it also reduces the accuracy rate.

## Result
