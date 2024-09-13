# ComfyUI_DeepFakeDefenders
You can using DeepFakeDefenders in comfyUI to Prediction image is a DeepFake img or not.

from 1st place solution for The Global Multimedia Deepfake Detection (Image Track) by "JTGroup" team.

**DeepFakeDefenders From: [link](https://github.com/HighwayWu/DeepFakeDefenders)** 

Update
---
**2024/09/13**
* Add the function of distinguishing images based on thresholds/增加根据阈值区分图片的功能
* Unmatched image will get a empty output/如果没有匹配的图片，会得到一张带有文字提示的空输出。
* Add image cropping function to better match the model/加入图片裁切，方法内置是512*512的分块，识别精度有轻微提升

1.Installation
-----
  In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_DeepFakeDefenders.git
```  
2.requirements  
----
For ComfyUI users, all libraries in the requirements file should be available. If not, please uncomment the # and reinstall
```
pip install -r requirements.txt
```
3 Need  model 
----    

download models [百度云](https://pan.baidu.com/s/1hh6Rub60T7UXok5rqACffQ?pwd=gxu5) or  [google drive](https://drive.google.com/drive/folders/1OQkkBn-Wv9PTHaxhXs_JF1IdkIES64pm)
```
├── ComfyUI/models/DeepFakeDefender
|             ├── ema.state
|             ├── weight.pth
```
4 Example
----
Notice,example img are all deepfake img,so  we still need to improve recognition accuracy
![](https://github.com/smthemex/ComfyUI_DeepFakeDefenders/blob/main/example/example.png)

5 Citation
------
This work is licensed under a [link](https://creativecommons.org/licenses/by-nc/4.0/) License   
**DeepFakeDefenders From: [link](https://github.com/HighwayWu/DeepFakeDefenders)** 

