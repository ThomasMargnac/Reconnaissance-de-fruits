import os
from natsort import natsorted
import numpy as np
import pandas as pd
import cv2 as cv
from skimage import morphology
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

def load_images_from_folder(folder):
	"""
	Description : Importe chaque image dans une liste

	Paramètre
	---------------
	folder : Chaine de caractère désignant le fichier où se trouvent les images

	Retour
	---------------
	images : Liste contenant toutes les images du fichier
	"""
	images = []
	files = natsorted(os.listdir(folder))
	for filename in files:
		img = cv.imread(os.path.join(folder,filename))
		if img is not None:
			images.append(img)
	return images

if __name__ == '__main__' :

	#Importation des images
	images = load_images_from_folder("img")

	#Attributs pour chaque image
	blue = []
	green = []
	red = []
	elongation = []
	compacite = []
	conca = []

	#Extraction des attributs
	for i in range(0,len(images)) :

		print("Extraction d'attributs de l'image n°{}/{}".format(i+1,len(images)))

		#Conversion en HSV
		hsv = cv.cvtColor(images[i], cv.COLOR_BGR2HSV)

		#Flou Gaussien et Thresholding avec OTSU
		blur = cv.GaussianBlur(hsv[:,:,1],(5,5),0)
		ret,thresholded = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

		#Fermeture sur le Threshold
		kernel = np.ones((13,13),np.uint8)
		closing = cv.morphologyEx(thresholded,cv.MORPH_CLOSE,kernel)

		#Region Filling
		max_area = int((thresholded.shape[0]*thresholded.shape[1])/16)
		holes_filling = np.array(cv.bitwise_not(closing),dtype=bool)
		holes_filling = morphology.remove_small_holes(ar=holes_filling, area_threshold=max_area)
		holes_filling = np.invert(holes_filling)
		holes_filling = morphology.remove_small_holes(ar=holes_filling, area_threshold=max_area)
		holes_filling = (holes_filling*1).astype("uint8")

		#Détermination de la couleur dominante
		masked = cv.bitwise_and(images[i],images[i],mask = holes_filling)
		blue_hist = []
		green_hist = []
		red_hist = []
		for k in range(1,256):
			blue_hist.append(np.count_nonzero(masked[:,:,0] == k))
			green_hist.append(np.count_nonzero(masked[:,:,1] == k))
			red_hist.append(np.count_nonzero(masked[:,:,2] == k))
		blue.append(np.argmax(blue_hist)+1)
		green.append(np.argmax(green_hist)+1)
		red.append(np.argmax(red_hist)+1)

		#Plus petit cercle circonscrit
		contours,_ = cv.findContours(holes_filling,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

		areas = [cv.contourArea(c) for c in contours]
		sorted_areas = np.sort(areas)

    	#Le plus grand contour
		cnt=contours[areas.index(sorted_areas[-1])]

		(x,y),radius = cv.minEnclosingCircle(cnt)
		center = (int(x),int(y))
		radius = int(radius)

		#Compacité
		area = cv.contourArea(cnt)
		perimeter = cv.arcLength(cnt,True)
		compacite.append((perimeter**2)/area)

		#Concavité
		hull = cv.convexHull(cnt)
		conca_area = [cv.contourArea(c) for c in hull]
		sorted_conca_areas = np.sort(conca_area)
		cnt_conca = hull[conca_area.index(sorted_conca_areas[-1])]
		conca_perimeter = cv.arcLength(cnt_conca,True)
		conca.append(conca_perimeter/perimeter)

		#Plus grand cercle inscrit
		distance_from_black_pixel = cv.distanceTransform(holes_filling,cv.DIST_L2,cv.DIST_MASK_3)
		max_ = np.max(distance_from_black_pixel)
		for j in range(0,distance_from_black_pixel.shape[0]):
			if np.max(distance_from_black_pixel[j]) == max_:
				y = j
				x = np.argmax(distance_from_black_pixel[j])
				break

		#Calcul elongation
		current_elongation = max_/radius
		elongation.append(current_elongation)

		#Affichage (Décommenter les lignes suivantes)
		"""cv.circle(images[i],center,radius,(0,255,0),2)
		cv.circle(images[i],(x,y),max_,(255,0,0),2)
		fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(10,10))
		ax1.imshow(holes_filling,'gray',vmin=0,vmax=1)
		ax2.imshow(images[i][:,:,::-1])
		plt.show()"""

	#Si les images ont été importées par ordre alpha numérique, l'ordre des labels est celui-ci : 1-Avocat, 2-Banane, 3-Citron, 4-Fraise, 5-Orange, 6-Tomate
	labels = [1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6]
	d = {'Bleu': blue, 'Vert': green, 'Rouge': red, 'Concavité': conca, 'Élongation': elongation, 'Compacité': compacite, 'Label': labels}
	df = pd.DataFrame(data=d)
	df.to_csv("dataframe.csv")
    print("\nFin")
