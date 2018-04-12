# CNTK_Realtime_Multi-Person_Pose_Estimation

This is a [CNTK](https://github.com/Microsoft/CNTK) implementation of **Realtime Multi-Person Pose Estimation**, origin code is here <https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation>

```
git clone --depth 1 https://github.com/Hzzone/CNTK_Realtime_Multi-Person_Pose_Estimation.git
```

## Model Download
* [onedrive](https://1drv.ms/f/s!AsLTqNyoZKl9jVvtiUeXwdDDOz4c)
* [坚果云](https://www.jianguoyun.com/p/DQDP1SMQlPvoBhizgk0)

## Requirements
```
CNTK>=2.3
matplotlib
opencv-python
```
or run `pip3 install -r requirements.txt`.

## Results
Run `python3/python demo.py` to see the result if you have cntk and other requirements installed.   <br>
![](sample/ski.jpg)
![](sample/preview.jpg)


------

![](sample/demo.gif)

## Citation
Please cite the paper in your publications if it helps your research:

    @inproceedings{cao2017realtime,
      author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
      year = {2017}
      }
      
    @inproceedings{wei2016cpm,
      author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Convolutional pose machines},
      year = {2016}
      }