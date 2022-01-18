# DLCV2021 final project
## Description
 - checkout [Final Project.pdf challenge 1](<https://github.com/Huan80805/DLCV2021_proj/tree/master/Final Project.pdf>)   
## method  
 - train AlexNet on slice level class label
 - predict case level class label
 - train Faster RCNN to predict bbox using only skull with fractures
 - predict bbox using case level label predicted by AlexNet
## Reproduce  

    cd reproduce
    bash final.sh <input_image_file_path> <out_csv_path>

## Result 
  - [Model Architecture and Comparison](<Result.pdf>)