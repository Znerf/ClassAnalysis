import cv2
import os

class Annotate():
	def __init__(run=True):
		pass
	
	def saveImage(self,name,img,location='annotateData/output/data1/'):
		name=location+name
		result=cv2.imwrite(name, img)
		if result==True:
			print("File saved successfully")
		else:
			print("Error in saving file")

	@staticmethod
	def directorycheck(name):
		if (not(os.path.exists(name))):
			os.mkdir(name)
			return True
		else:
			return False