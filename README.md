{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "konumum :  E:\\dataset\n"
     ]
    }
   ],
   "source": [
    "import json\n",
    "import os\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import cv2\n",
    "from tqdm import tqdm as tqdm\n",
    "\n",
    "data = json.load(open('annotations/annotation.json')) # load jeson data into code\n",
    "print(\"konumum : \",os.getcwd())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "y=data[\"categories\"]\n",
    "x=data[\"images\"]\n",
    "segment=data[\"annotations\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def format_label(data,image_about,image_path):\n",
    "    #   format=[file_name,w,h,category,boxs]\n",
    "\n",
    "    filname=[]\n",
    "    for i in data:\n",
    "        for k in image_about:\n",
    "            if i[\"id\"] == k[\"image_id\"]:\n",
    "                formatla=[os.path.join(image_path,i[\"file_name\"]),\n",
    "                          i[\"width\"],\n",
    "                          i[\"height\"],\n",
    "                          k[\"category_id\"],\n",
    "                          k[\"bbox\"]\n",
    "                ]\n",
    "                filname.append(formatla)\n",
    "                break\n",
    "\n",
    "    return np.array(filname)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "<ipython-input-3-e766c6b117e1>:17: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray\n",
      "  return np.array(filname)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(484, 5)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "images=format_label(x,segment,\"images\")\n",
    "images.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def kare_draw(image,start_point,end_point,color):\n",
    "    # Line thickness of 2 px\n",
    "    thickness = 2\n",
    "\n",
    "    # Using cv2.rectangle() method\n",
    "    # Draw a rectangle with blue line borders of thickness of 2 px\n",
    "    image = cv2.rectangle(image, start_point, end_point, color, thickness)\n",
    "\n",
    "    # return the new image\n",
    "    return image\n",
    "\n",
    "def x1_y1_convert(boxx):\n",
    "        \n",
    "        x,y,w,h=boxx # orginal format x1,y1,w,h\n",
    "        \n",
    "        x1=int(x)\n",
    "        x2=int(x+(w))\n",
    "        y1=int(y)\n",
    "        y2=int(y+(h))\n",
    "        \n",
    "        return [(x1,y1),(x2,y2)]\n",
    "\n",
    "    \n",
    "def yolov_format(image_size=(\"H\",\"W\"),boxxes=[(\"x1\",\"y1\"),(\"x2\",\"y2\")]):\n",
    "    \n",
    "    IW,IH=image_size[0],image_size[1]\n",
    "    \n",
    "    try:\n",
    "        (x1,y1),(x2,y2)=boxxes\n",
    "    except:\n",
    "        x1,y1,x2,y2=boxxes\n",
    "    \n",
    "    \n",
    "    w,h=abs(x2-x1),abs(y2-y1)\n",
    "    x,y=x1+(w/2),y1+(h/2)\n",
    "    \n",
    "    #   yolo format x/IW  y/IH w/IW h/IH\n",
    "    return [x/IW,y/IH,w/IW,h/IH]\n",
    "    \n",
    "    \n",
    "    \n",
    "def resim_kırpma(img,img_box=(\"x1\",\"x2\")):\n",
    "    img_shape=img.shape\n",
    "    \n",
    "    image_center_point=(img_shape[0]/2,img_shape[1]/2)\n",
    "    \n",
    "    image_center_x=image_center_point[1]\n",
    "    \n",
    "    if image_center_x<img_box[0] :\n",
    "        img=img[:,int(image_center_x):]\n",
    "        nex_points_x=(img_box[0]-image_center_x,img_box[1]-image_center_x)\n",
    "        \n",
    "    elif image_center_x>img_box[1]:\n",
    "        img=img[:,:int(image_center_x)]\n",
    "        nex_points_x=img_box\n",
    "    else:\n",
    "        img=img[:,:int(image_center_x)+(img_box[1]-img_box[0])]\n",
    "        nex_points_x=img_box\n",
    "    return img,nex_points_x\n",
    "\n",
    "def save(path,image_label):\n",
    "    #   format=[file_name,w,h,category,boxs]\n",
    "    np.save(path,image_label)\n",
    "    \n",
    "    \n",
    "    \n",
    "def save_for_txt(img_label):\n",
    "     with open(img_label[0].split(\".\")[0]+\".txt\",\"w\") as file:\n",
    "        # yolov format pc x,y,w,h\n",
    "        pc=img_label[-2]\n",
    "\n",
    "        x,y,w,h=img_label[-1]\n",
    "        yolo=str(pc-1)+\" \"+str(x)+\" \"+str(y)+\" \"+str(w)+\" \"+str(h)\n",
    "        file.write(yolo)\n",
    "    \n",
    "\n",
    "def save_image(img_label,s,test=False):\n",
    "    img=plt.imread(img_label[0])\n",
    "    if test:\n",
    "        img_label[0]=f\"hazir_veri\\\\test_veri\\\\{s}.jpg\"\n",
    "    else:\n",
    "        img_label[0]=f\"hazir_veri\\\\{s}.jpg\"\n",
    "    \n",
    "    img_label[-1]=yolov_format(img_label[1:3],img_label[-1])\n",
    "    plt.imsave(img_label[0],img)\n",
    "    \n",
    "    \n",
    "    save_for_txt(img_label)\n",
    "    \n",
    "    \n",
    "def load(path):\n",
    "    return np.load(path,allow_pickle=True)\n",
    "\n",
    "\n",
    "def name_change(img_label,during):\n",
    "    \"\"\"yol=img[:,0]             # paths  separator  \n",
    "    cate=img[:,3]            # category seperator\n",
    "    boxx=img[:,-1]           # boxxes seperator\"\"\"\n",
    "    \n",
    "    for s,i in enumerate(tqdm(img_label)):\n",
    "        if int((len(during)*20)/100)>s+1:\n",
    "            save_image(i,during[s],test=True)\n",
    "        else:\n",
    "            save_image(i,during[s],test=False)\n",
    "\n",
    "def shuffle(array_num,shufle_num):\n",
    "    array=list(range(array_num))\n",
    "    for i in range(shufle_num):\n",
    "        np.random.shuffle(array)\n",
    "        \n",
    "    return array\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "images[:,4]=[x1_y1_convert(i) for i in images[:,4]]\n",
    "#change= [yolov_format(i[1:3],i[-1]) for i in images]\n",
    "\n",
    "#images[:,4]=change\n",
    "#save_for_txt(images)  # save orjinal data at disk"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "#i=kare_draw(i,(x1, y1), (x2, y2),(255,0,0))              #  this  code draws  rectangle\n",
    "#print (\"category : \",c,\"  boxxes : \",b)\n",
    "    #plt.imsave(f\"images/cut_image/cut_{image_name}\",i)         # this code  save cut_images at disk\n",
    "    #plt.imshow(i)\n",
    "    #plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "cut_img_label=[]\n",
    "def data_cut(img_label,during):\n",
    "    \n",
    "    \n",
    "    \n",
    "    for s,i in enumerate(tqdm(img_label)):\n",
    "        try:\n",
    "            (x1,y1),(x2,y2)=i[-1]\n",
    "        except:\n",
    "            x1,y1,x2,y2=i[-1]\n",
    "        img=plt.imread(i[0])\n",
    "        img,(x1,x2)=resim_kırpma(img,(x1,x2))                         #  this  code  cuts images  from center\n",
    "        \n",
    "        \"\"\"\n",
    "        image_name=str(during[s])+\".jpg\"\n",
    "        cut_img_label.append([os.path.join(\"images\\\\cut_image\",image_name),\n",
    "                              img.shape[1],\n",
    "                              img.shape[0],\n",
    "                              i[-2],\n",
    "                              [(x1,y1),(x2,y2)]])\n",
    "        \"\"\"\n",
    "        \n",
    "        if int((len(during)*20)/100)>s+1:\n",
    "            save_image(i,during[s],test=True)\n",
    "        else:\n",
    "            save_image(i,during[s],test=False)\n",
    "        \n",
    "    \n",
    "    \n",
    "def veri_hazırlığı(img_label):\n",
    "    cut=True\n",
    "    img_num=shuffle(len(img_label)*2,20)\n",
    "    \n",
    "    #name_change(img_label,img_num[:len(img_label)])\n",
    "    print(\"saved orginal data\")\n",
    "    \n",
    "    data_cut(img_label,img_num[len(img_label):])\n",
    "    print(\"saved cut data\")\n",
    "        \n",
    "    \n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "  0%|▏                                                                                 | 1/484 [00:00<00:50,  9.52it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "saved orginal data\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|████████████████████████████████████████████████████████████████████████████████| 484/484 [05:02<00:00,  1.60it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "saved cut data\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "veri_hazırlığı(images)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 201,
   "metadata": {},
   "outputs": [],
   "source": [
    "dene=np.array(cut_img_label)\n",
    "for i in dene:\n",
    "    #box=yolov_format(i[1:3],i[-1])\n",
    "    cut_img_label=np.array(cut_img_label)\n",
    "    cut_img_label.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 195,
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "not enough values to unpack (expected 4, got 2)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-195-79be5113e724>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0msave_for_txt\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcut_img_label\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m<ipython-input-193-79cb317b11fd>\u001b[0m in \u001b[0;36msave_for_txt\u001b[1;34m(image_label)\u001b[0m\n\u001b[0;32m     70\u001b[0m                 \u001b[1;31m# yolov format pc x,y,w,h\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     71\u001b[0m                 \u001b[0mpc\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mi\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m-\u001b[0m\u001b[1;36m2\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 72\u001b[1;33m                 \u001b[0mx\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0my\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mw\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mh\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mi\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;33m-\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     73\u001b[0m                 \u001b[0myolo\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mstr\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mpc\u001b[0m\u001b[1;33m-\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m+\u001b[0m\u001b[1;34m\" \"\u001b[0m\u001b[1;33m+\u001b[0m\u001b[0mstr\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mx\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m+\u001b[0m\u001b[1;34m\" \"\u001b[0m\u001b[1;33m+\u001b[0m\u001b[0mstr\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0my\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m+\u001b[0m\u001b[1;34m\" \"\u001b[0m\u001b[1;33m+\u001b[0m\u001b[0mstr\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mw\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m+\u001b[0m\u001b[1;34m\" \"\u001b[0m\u001b[1;33m+\u001b[0m\u001b[0mstr\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mh\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     74\u001b[0m                 \u001b[0mfile\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mwrite\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0myolo\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mValueError\u001b[0m: not enough values to unpack (expected 4, got 2)"
     ]
    }
   ],
   "source": [
    "save_for_txt(cut_img_label)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "a=list(range(100))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[0,\n",
       " 1,\n",
       " 2,\n",
       " 3,\n",
       " 4,\n",
       " 5,\n",
       " 6,\n",
       " 7,\n",
       " 8,\n",
       " 9,\n",
       " 10,\n",
       " 11,\n",
       " 12,\n",
       " 13,\n",
       " 14,\n",
       " 15,\n",
       " 16,\n",
       " 17,\n",
       " 18,\n",
       " 19,\n",
       " 20,\n",
       " 21,\n",
       " 22,\n",
       " 23,\n",
       " 24,\n",
       " 25,\n",
       " 26,\n",
       " 27,\n",
       " 28,\n",
       " 29,\n",
       " 30,\n",
       " 31,\n",
       " 32,\n",
       " 33,\n",
       " 34,\n",
       " 35,\n",
       " 36,\n",
       " 37,\n",
       " 38,\n",
       " 39,\n",
       " 40,\n",
       " 41,\n",
       " 42,\n",
       " 43,\n",
       " 44,\n",
       " 45,\n",
       " 46,\n",
       " 47,\n",
       " 48,\n",
       " 49,\n",
       " 50,\n",
       " 51,\n",
       " 52,\n",
       " 53,\n",
       " 54,\n",
       " 55,\n",
       " 56,\n",
       " 57,\n",
       " 58,\n",
       " 59,\n",
       " 60,\n",
       " 61,\n",
       " 62,\n",
       " 63,\n",
       " 64,\n",
       " 65,\n",
       " 66,\n",
       " 67,\n",
       " 68,\n",
       " 69,\n",
       " 70,\n",
       " 71,\n",
       " 72,\n",
       " 73,\n",
       " 74,\n",
       " 75,\n",
       " 76,\n",
       " 77,\n",
       " 78,\n",
       " 79,\n",
       " 80,\n",
       " 81,\n",
       " 82,\n",
       " 83,\n",
       " 84,\n",
       " 85,\n",
       " 86,\n",
       " 87,\n",
       " 88,\n",
       " 89,\n",
       " 90,\n",
       " 91,\n",
       " 92,\n",
       " 93,\n",
       " 94,\n",
       " 95,\n",
       " 96,\n",
       " 97,\n",
       " 98,\n",
       " 99]"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
