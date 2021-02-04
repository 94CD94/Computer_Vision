import numpy as np
from PIL import Image 
from PIL import ImageShow as Is
import os 
os.chdir(r'C:\Users\LTM0110User\Desktop\vs_code')



# In this program several operation to an image are performed in order to learn how to modify the images
# using OpenCV



arr=np.random.randint(0,256,(100,100))
arr[0]=np.zeros(100)
arr[:,-1]=np.zeros(100)
arr[-1]=np.zeros(100)
arr[:,0]=np.zeros(100)
img=Image.fromarray(arr)
img.show() #this showld be black all zeros

withe= np.ones((20,20))*255
x_o=  [int(arr.shape[0]/2)-int(withe.shape[0]/2),int(arr.shape[0]/2)+int(withe.shape[0]/2)]           
y_o=  [int(arr.shape[1]/2)-int(withe.shape[1]/2),int(arr.shape[1]/2)+int(withe.shape[1]/2)]           
# Withe, all 256
arr[x_o[0]:x_o[1],y_o[0]:y_o[1]]=withe
img=Image.fromarray(arr)

#___________________________________________________________________________________________________

img=Image.open(r'./cv00-master/A/res/cat.jpg') # loading the picture of a cat
#img=img.convert('L') # Passo da una matrice in 3D a una in 1D (Bianco e nero)
arr=np.array(img)
img=Image.fromarray(arr[:,:,:])
img.show()
#____________________________________________________________________________________________________
import cv2 # Opencv
print(cv2.__version__)

img= cv2.imread(r'./cv00-master\2 - Operare sulle Immagini\res\elon.jpg', cv2.IMREAD_GRAYSCALE) # Loading picture
cv2.imshow('image',img) #Displaying picture
 
print(type(img))
img2= cv2.imread(r'./cv00-master\2 - Operare sulle Immagini\res\elon.jpg')
cv2.imshow('ROBA',img2)
cv2.waitKey(0) # Compulsory!
cv2.destroyAllWindows()
#____________________________________________________________________________________________________
os.chdir(r'./cv00-master\2 - Operare sulle Immagini')

img=cv2.imread(r'./res\elon.jpg', cv2.IMREAD_COLOR)
print(img.shape)

# Resize
img_h,img_w=550, 500
img_resized= cv2.resize(img, (img_h, img_w))
cv2.imshow('Redim', img_resized)
 

# Cropping 
size = 200
img_cropped= img_resized[img_h//2-size//2:img_h//2+size//2 ,\
     img_w//2-size//2:img_w//2+size//2 ]
cv2.imshow('Cropped',img_cropped)
 

# Rotation
angle=180
center= (img_h//2, img_w//2)
rot_img= cv2.getRotationMatrix2D(center, angle, 1)
img_rotated= cv2.warpAffine(img_resized,rot_img,(img_w,img_h))
cv2.imshow('Ruotata', img_rotated)
cv2.waitKey(0)

cv2.destroyAllWindows()


cv2.imwrite('./img_rotated.jpg', img_rotated) 
cv2.imwrite('./img_resized.jpg', img_resized) 
cv2.imwrite('./img_cropped.jpg', img_cropped)

#______________________________________________________________________________________________-
import cv2 
RED= (0, 0, 255) # B, G, R 
TIK=4
img = cv2.imread(  r'./res/elon_resized.jpg' )
#cv2.imshow('cacca', img  )
#cv2.waitKey(0)
img_h, img_w =img.shape[0:2]
l=200
center= (img_w//2, img_h//2)
#Quadrato
cv2.rectangle(img, (center[0]-l//2 , center[1]-l//2), (center[0]+l//2, center[1]+l//2),RED,-1)
cv2.imshow("rosso" ,img )
cv2.waitKey(0)
#Cerchio
img = cv2.imread(  r'./res/elon_resized.jpg' )
GRE=(0,255,0)
R=10
cv2.circle(img, center, R, GRE, -1)
cv2.imshow("verde", img) 
cv2.waitKey(0)
#Due linee
img = cv2.imread(  r'./res/elon_resized.jpg' )
cv2.line(img, (center[0],0), (center[0], img_h ), (255,0,0), 1.5 )
cv2.line(img, (0,center[1]), (img_w, center[1]), (255,0,0), 1.5 )

cv2.imshow("linea", img)
cv2.waitKey(0)

#Testo nella immagine 
img = cv2.imread(  r'./res/elon_resized.jpg' )
cv2.putText(img, "@Carniani Davide ", (center[0]-70, img_h-20 ),cv2.FONT_HERSHEY_PLAIN,  1 ,(0,0,0), 2  )
cv2.imshow("linea", img)
cv2.waitKey(0)
