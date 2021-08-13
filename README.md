README
===
pytorch 版本:
torch 1.7.1
torchaudio 0.7.2
torchvision 0.8.2
## 照片標記
使用labelme標記會產生json檔，需要把他轉乘maskRCNN看得懂的mask
1. 工具: [labelme](https://chtseng.wordpress.com/2020/06/18/labelme%E8%88%87coco%E5%BD%B1%E5%83%8F%E5%88%86%E5%89%B2%E6%A8%99%E8%A8%98/)
:::info
multiple class標記方式: 假設有cat跟dog兩種class分別有好多個，標記物件的lable我的方法是這樣
```
cat: cat1, cat2, cat3...
dog: dog1, dog2, dog3...
```
這樣再將json轉成label.png的時候才會將每個dog(或class)視為不同物件，給予不同的像素標記

我也有看到有人是給所有dog都標dog但給不同group id但如果用我轉換資料的方式會變成所有狗都是同一個物件，可能有別種轉換方式我不知道XD
:::

2.  批量轉換
建立.bat檔於產生.json的同層資料夾下，並執行.bat檔
執行後每個json檔都會產生一個資料夾，裡面包含img.png、label.png、label_names.txt、lable_viz.png
```bat=
@echo off
for %%i in (*.json) do labelme_json_to_dataset "%%i"
pause
```
:::warning
若資料集為multiple class需要多產生一個info.yaml，續修改labelme_json_to_dataset[參考資料](https://blog.csdn.net/winter616/article/details/104426111)
:::
3. 16bits轉8bits
將每個資料夾中label.png轉換成8bits的.png
```python=
def img_16to8():
    from PIL import Image
    import numpy as np
    import shutil
    import os

    src_dir = r'E:\code\Tongue_detect\train_data\labelme_json'
    dest_dir = r'E:\code\Tongue_detect\train_data\cv2_mask'
    for child_dir in os.listdir(src_dir):
        new_name = child_dir.split('_')[0] + '.png'
        old_mask = os.path.join(os.path.join(src_dir, child_dir), 'label.png')
        img = Image.open(old_mask)
        img = Image.fromarray(np.uint8(np.array(img)))
        new_mask = os.path.join(dest_dir, new_name)
        img.save(new_mask)

if __name__ == "__main__":
    img_16to8()
```
:::warning
圖片務必為png，jpg會出事
:::



## 自訂義資料集

* input 
    * image.png
    * mask.png
    * info.yaml
* output
    * image(影像本生資訊)
    * target
        * labels
        * masks
        * boxes(標記框)
        * area (標記框大小)
        * iscrowd (iscrowd=1忽略此標記框)
        * image_id
:::info
剛剛標記的時候把每個物件label都是unique的，所以為了得到每個物件正確的類別需要多一個info.yaml
```
dog1, dog2, dog3 -> 1
cat1, cat2, cat3 -> 2
```
:::

## 畫bbox
maskRCNN output的bbox是以mask的最左上角和最右下角的點畫出來的，不是旋轉的bbox
用cv2.findContours可以框出圍住物體的最小矩形(旋轉bbox)