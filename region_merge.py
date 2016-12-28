import cv2
import numpy as np
from scipy import signal
import PIL
from PIL import Image
np.seterr(over='ignore')

image = cv2.imread('MixedVegetables.jpg',0)

img2_row = len(image)
img2_col =  len(image[0])

cv2.imshow('Original Image',image)
cv2.waitKey(0)
Image.fromarray(image).save("Original Image.jpg")

def crack_edge(img):
	row = len(img)
	col = len(img[0])
	
	result = np.zeros(shape = (row*2,col*2),dtype=np.float)
	
	row_r = len(result)
	col_r = len(result[0])
	
	img_x = 0
	
	for i in range(row_r):
		img_y = 0
		diff=0
		for j in range(col_r):
			if(img_x==0):
				if(img_y ==0):
					diff= img[img_x+1][img_y]-img[img_x][img_y+1]
				elif(img_y ==col-1):
					diff = img[img_x][img_y-1]-img[img_x+1][img_y]
				else:
					diff= float(img[img_x][img_y-1]-img[img_x+1][img_y]-img[img_x][img_y+1])
			elif(img_x==row-1):
				if(img_y==0):
					diff= img[img_x-1][img_y]-img[img_x][img_y+1]
				elif(img_y ==col-1):
					diff= img[img_x-1][img_y]-img[img_x][img_y-1]
				else:
					diff= float(img[img_x-1][img_y]-img[img_x][img_y-1]-img[img_x][img_y+1])
			elif((img_y==0 and img_x!=0) or (img_y==0 and img_x!=row-1)):
				if(img_y==0):
					diff= img[img_x-1][img_y]-img[img_x][img_y+1]-img[img_x+1][img_y]
				elif(img_y==col-1):
					diff= img[img_x-1][img_y]-img[img_x][img_y-1]-img[img_x+1][img_y]
			else:
				diff = img[img_x][img_y-1]-img[img_x][img_y+1]-img[img_x-1][img_y]-img[img_x+1][img_y]
			
			
			if(i%2==0 and j%2==0):
				result[i][j] = img[img_x][img_y]
			elif(i%2==0 and j%2==1 and img_y < col-2):
				result[i][j] = diff
				result[i-1][j] = diff
				img_y +=1
			elif(i%2==1 and j%2==0):
				result[i][j] = result[i-1][j+1]
			elif(i%2==1 and j == col_r-1 and img_x < row-1):
				img_x +=1
	res = np.uint8(result)
	cv2.imshow('Crack Edge Image',res)
	cv2.waitKey()
	Image.fromarray(res).save("Crack Edge Image.jpg")
	return np.uint8(result)

def threshold(img):
	thres = 165
	row = len(img)
	col = len(img[0])
	result = np.zeros(shape = (row,col),dtype=np.float)
	
	for i in range(row):
		for j in range(col):
			if(img[i][j] < thres):
				result[i][j] = 0
			else:
				result[i][j] = img[i][j]
	result = np.uint8(result)
	cv2.imshow('Merged Image',result)
	cv2.waitKey()
	Image.fromarray(result).save("Merged Image.jpg")
	return result

ce = crack_edge(image)
t1 = threshold(ce)

cv2.destroyAllWindows()
