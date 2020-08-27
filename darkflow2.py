#!/usr/bin/env python
# coding: utf-8

# In[2]:


from darkflow.net.build import TFNet
import cv2
import numpy as np


# In[3]:


options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}
tfnet = TFNet(options)


# In[4]:


cap = cv2.VideoCapture(0)


# In[5]:


class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
              'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor', 'keyboard', "book", "laptop"]

num_classes = len(class_names)
class_colors = []


# In[6]:


for i in range(num_classes):
    hue = 255 * i / num_classes
#     hue = 64
    col = np.zeros((1, 1, 3)).astype("uint8")
    col[0][0][0] = hue
    col[0][0][1] = 128
    col[0][0][2] = 255
    cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
    col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
    class_colors.append(col)


# In[ ]:


while True:
    ret, frame = cap.read()
    result = tfnet.return_predict(frame)
    for item in result:
        tlx = item["topleft"]["x"]
        tly = item["topleft"]["y"]
        brx = item["bottomright"]["x"]
        bry = item["bottomright"]["y"]
        label = item["label"]
        conf = item["confidence"]

        if conf > 0.6:
            for class_name in class_names:
                if label == class_name:
                    class_num = class_names.index(class_name)
                    break

            cv2.rectangle(frame, (tlx, tly), (brx, bry), class_colors[class_num], 5)

            text = label + " " + ("%.2f" % conf)
            cv2.rectangle(frame, (tlx, tly - 60), (tlx + 500, tly + 5), class_colors[class_num], -1)
            cv2.putText(frame, text, (tlx, tly), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.imshow("Show FLAME Image", frame)
    k = cv2.waitKey(10);
    if k == ord('q'): break  

cap.release()
cv2.destroyAllWindows()


# In[ ]:


cap.release()



# %%


# %%
