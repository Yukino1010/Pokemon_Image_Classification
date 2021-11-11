# Pokemon_Image_Classification
<p align="center"><img width="200px" src="https://github.com/Yukino1010/Pokemon_Image_Classification/blob/master/centered-sprites/gen05_black-white/644.png" />&emsp;&emsp;&emsp;&emsp;&emsp;<img width="200px" src="https://github.com/Yukino1010/Pokemon_Image_Classification/blob/master/centered-sprites/gen05_black-white/643.png" /></p>
<br>
<p align="center"><img width="200px" src="https://github.com/Yukino1010/Pokemon_Image_Classification/blob/master/centered-sprites/gen05_black-white/483.png" />&emsp;&emsp;&emsp;&emsp;&emsp;<img width="200px" src="https://github.com/Yukino1010/Pokemon_Image_Classification/blob/master/centered-sprites/gen05_black-white/484.png" /></p>

## Introduce

## Network Structure

**architecture**  
&emsp;
![image](https://github.com/Yukino1010/Pokemon_Image_Classification/blob/master/model/structure.png)
&emsp;
### network design
- use structure similar to AlexNet but reduce the filter of each layers
- add AveragePooling behind the last Conv2d to reduce the parameter of Dense
