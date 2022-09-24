# **YOLOV4 İLE TABELA TESPİT**

###  Yolo algoritmasi kullanılarak geliştirilmiştirdiğim Utility Pole Information Plate (elektrik direği bilgi tabelası ) ve SignStore (dükkan tabelası) objelerini tanıyan algoritmayı tanıtacağım


---
<br/><br/>

## **Data Set**
---
### [visiosoft](http://www.visiosoft.com.tr) tarafından toplam 482 tane data hazırlandı.Bu data Utility Pole Information Plate (elektrik direği bilgi tabelası ) ve SignStore (dükkan tabelası) adlı iki obje içeriyor .datadaki verilerden bazılırı  geniş açılı ve yüksek çözünürlükteki verilerdi.
### Bu veriler <font  color="aqua "> x1 y1 w h </font> formatı ile etiketlenmişti . Verileri eyitmek için gereken yolo modeli forma olarak  <font  color="aqua "> x y  w h </font>  sıralaması ile ilerliyordu verilerin uygun formatlanması için python dilinden  yararlandım .
<br/>

```PowerShell
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm as tqdm

data = json.load(open('annotations/annotation.json')) # load jeson data into code
```

### Veriler <font  color="pink "> json</font> formatı ile verilmişti verileri buzeltmek için  <font  color="pink "> json</font> kütüphanesinden yararlandım. Verileri okumak için  <font  color="pink "> data = json.load(open('annotations/annotation.json'))</font> kodundan yararlandım
<br/>

```PowerShell
def format_label(data,image_about,image_path):
    #   format=[file_name,w,h,category,boxs]

    filname=[]
    for i in data:
        for k in image_about:
            if i["id"] == k["image_id"]:
                formatla=[os.path.join(image_path,i["file_name"]),
                          i["width"],
                          i["height"],
                          k["category_id"],
                          k["bbox"]
                ]
                filname.append(formatla)
                break

    return np.array(filname)

images=format_label(x,segment,"images")
```

### Verileri yolo formatına çevirmeden önce etiket içerisini inceleme ve gerekli etiketleri almak için <font  color="pink ">format=[file_name,w,h,category,boxs] </font> olarak  yeniden formatladım.


<br/>

```PowerShell
def x1_y1_convert(boxx):

        x,y,w,h=boxx # orginal format x1,y1,w,h

        x1=int(x)
        x2=int(x+(w))
        y1=int(y)
        y2=int(y+(h))

        return [(x1,y1),(x2,y2)]

images[:,4]=[x1_y1_convert(i) for i in images[:,4]]

```

### Veriler yetersiz kalabileceğinden verileri çoğaltmak için her bir resime kırpma uyguladım . Veri seti  üzerinden kırpma  işlemini kulanmak için obje konumunu belirleyen etiketleri  <font  color="pink "> boxx = x1 y1 x2 y2 </font>  olarak tekrar formatladım

<br/>

```PowerShell
def resim_kırpma(img,img_box=("x1","x2")):
    img_shape=img.shape

    image_center_point=(img_shape[0]/2,img_shape[1]/2)

    image_center_x=image_center_point[1]

    if image_center_x<img_box[0] :
        img=img[:,int(image_center_x):]
        nex_points_x=(img_box[0]-image_center_x,img_box[1]-image_center_x)

    elif image_center_x>img_box[1]:
        img=img[:,:int(image_center_x)]
        nex_points_x=img_box
    else:
        img=img[:,:int(image_center_x)+(img_box[1]-img_box[0])]
        nex_points_x=img_box
    return img,nex_points_x

```