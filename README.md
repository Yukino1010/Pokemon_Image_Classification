# Pokemon_Image_Classification
<br>
<p align="center"><img width="200px" src="https://github.com/Yukino1010/Pokemon_Image_Classification/blob/master/centered-sprites/gen05_black-white/644.png" />&emsp;&emsp;&emsp;&emsp;&emsp;<img width="200px" src="https://github.com/Yukino1010/Pokemon_Image_Classification/blob/master/centered-sprites/gen05_black-white/643.png" /></p>
<br>
<p align="center"><img width="200px" src="https://github.com/Yukino1010/Pokemon_Image_Classification/blob/master/centered-sprites/gen05_black-white/483.png" />&emsp;&emsp;&emsp;&emsp;&emsp;<img width="200px" src="https://github.com/Yukino1010/Pokemon_Image_Classification/blob/master/centered-sprites/gen05_black-white/484.png" /></p>

## Introduce
The topic this time is pokemon attribute classification,because many pokemons have dual attributes, only one of them will be used this time.<br>
In this implementation,I seperate train and test data randomly instead of seperate by the unique sprite to avoid the situation that some kinds of attribute only appear in testing data or training data.<br>
Alough there is a little bit overfitting during the training process, but I think the performence of the model is already enough for the classification.

## Network Structure

**architecture**  

![image](https://github.com/Yukino1010/Pokemon_Image_Classification/blob/master/model/structure.png)

### network design
- use structure similar to AlexNet but reduce the filter of each layers
- add AveragePooling behind the last Conv2d to reduce the parameter of Dense
## Data
this data set was collected from [Veekun](https://veekun.com/) and processed by  [Journal of Geek Studies](https://jgeekstudies.org/)


## Loss and Accuracy

![image](https://github.com/Yukino1010/Pokemon_Image_Classification/blob/master/accracy_loss/origin.png)

At the begining I use the origin data to train my model.Although I got 60% accuracy on testing data ,but it was obviously overfitting to the training data,
so I consider to do data augumentation to slove this problem.

**after data augumentation** 
![image](https://github.com/Yukino1010/Pokemon_Image_Classification/blob/master/accracy_loss/data_aug.png)

From the results of the picture, it seems that it can indeed solve the overfitting problem, but at the same time it also reduces the accuracy rate.

## Result
 
<p align="center"><img width="450px" src="https://github.com/Yukino1010/Pokemon_Image_Classification/blob/master/result/train_result.png"><img width="450px"  src="https://github.com/Yukino1010/Pokemon_Image_Classification/blob/master/result/result.png"></p>
<p align="center"><img width="700px" src="https://github.com/Yukino1010/Pokemon_Image_Classification/blob/master/result/pictur_10.png"><img width="700px"  src="https://github.com/Yukino1010/Pokemon_Image_Classification/blob/master/result/pictur_3.png"></p>

## References
Journal of Geek Studies (https://jgeekstudies.org/)<br>
Neuralmon  (https://github.com/hemagso/Neuralmon)
