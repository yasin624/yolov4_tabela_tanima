# **YOLOV4 İLE TABELA TESPİT**

###  Yolo algoritmasi kullanılarak geliştirilmiştirdiğim Utility Pole Information Plate (elektrik direği bilgi tabelası ) ve SignStore (dükkan tabelası) objelerini tanıyan algoritmayı tanıtacağım


---
<br/><br/>

## **Data Set Düzenlenmesi**
---
#### [visiosoft](http://www.visiosoft.com.tr) tarafından toplam 482 tane data hazırlandı.Bu data Utility Pole Information Plate (elektrik direği bilgi tabelası ) ve SignStore (dükkan tabelası) adlı iki obje içeriyor .datadaki verilerden bazılırı  geniş açılı ve yüksek çözünürlükteki verilerdi.
#### Bu veriler <font  color="aqua "> x1 y1 w h </font> formatı ile etiketlenmişti . Verileri eyitmek için gereken yolo modeli forma olarak  <font  color="aqua "> x y  w h </font>  sıralaması ile ilerliyordu verilerin uygun formatlanması için python dilinden  yararlandım .
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

#### Veriler <font  color="pink "> json</font> formatı ile verilmişti verileri buzeltmek için  <font  color="pink "> json</font> kütüphanesinden yararlandım. Verileri okumak için  <font  color="pink "> data = json.load(open('annotations/annotation.json'))</font> kodundan yararlandım
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

#### Verileri yolo formatına çevirmeden önce etiket içerisini inceleme ve gerekli etiketleri almak için <font  color="pink ">format=[file_name,w,h,category,boxs] </font> olarak  yeniden formatladım.


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

#### Veriler yetersiz kalabileceğinden verileri çoğaltmak için her bir resime kırpma uyguladım . Veri seti  üzerinden kırpma  işlemini kulanmak için obje konumunu belirleyen etiketleri  <font  color="pink "> boxx = x1 y1 x2 y2 </font>  olarak tekrar formatladım

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

#### Resim dosyalarını  <font  color="pink "> resim_kırpma() </font> fonksiyonu ile resimleri ortadan ikiye ayırdım. Bu sayede verileri bozmadan ,ayrıntıları fazla kaçırmadan objenin bulunduğu bölgeyi aldım.


<br/>

```PowerShell
def save_image(img_label,s,test=False):
    img=plt.imread(img_label[0])
    if test:
        img_label[0]=f"hazir_veri\\test_veri\\{s}.jpg"
    else:
        img_label[0]=f"hazir_veri\\{s}.jpg"

    img_label[-1]=yolov_format(img_label[1:3],img_label[-1])
    plt.imsave(img_label[0],img)


    save_for_txt(img_label)

```
#### Son olarak kırpılmış verileri kaydetmek için <font  color="pink "> save_image() </font> fonksiyonundan yararlanıldım.


<br/>

```PowerShell

def yolov_format(image_size=("H","W"),boxxes=[("x1","y1"),("x2","y2")]):

    IW,IH=image_size[0],image_size[1]

    try:
        (x1,y1),(x2,y2)=boxxes
    except:
        x1,y1,x2,y2=boxxes


    w,h=abs(x2-x1),abs(y2-y1)
    x,y=x1+(w/2),y1+(h/2)

    #   yolo format x/IW  y/IH w/IW h/IH
    return [x/IW,y/IH,w/IW,h/IH]

```

#### Etiketleri kaydetme den önce boxx değerlerini <font  color="pink "> x y w h </font> formatına dönuştürmek için <font  color="pink "> yolov_format() </font> fonksiyonundan  yararlandım.Veriler yolo modeline  aktarılmaya hazır hale getirdim.

<br/>

```PowerShell
def shuffle(array_num,shufle_num):
    array=list(range(array_num))
    for i in range(shufle_num):
        np.random.shuffle(array)

    return array

```
#### Modelin verimliliğini artırmak için veri setini karıştırmak için etiketleri  <font  color="pink "> shuffle() </font> foksiyonundan geçirdim.

<br/>

```PowerShell
def save_for_txt(img_label):
     with open(img_label[0].split(".")[0]+".txt","w") as file:
        # yolov format pc x,y,w,h
        pc=img_label[-2]

        x,y,w,h=img_label[-1]
        yolo=str(pc-1)+" "+str(x)+" "+str(y)+" "+str(w)+" "+str(h)
        file.write(yolo)


```

####  Son olarak etiketleri kaydetme işlemi ile veri düzenleme kısmını bitirmiş oldum.

<br/><br/><br/>


# **YOLOV4 Model Eğitimi**
