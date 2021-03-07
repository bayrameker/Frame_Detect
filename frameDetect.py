import cv2
import numpy as np


aranan_resim = cv2.imread('images/a.jpg')
cv2.imshow("Original Template", aranan_resim)
cv2.waitKey(0)


height, width = aranan_resim.shape[:2]

aranilan_resim_org = cv2.imread('images/aa.jpg')
cv2.imshow("Original Image", aranilan_resim_org)
cv2.waitKey(0)


gray_template = cv2.cvtColor(aranan_resim, cv2.COLOR_BGR2GRAY)
gray_original = cv2.cvtColor(aranilan_resim_org, cv2.COLOR_BGR2GRAY)


match = cv2.matchTemplate(gray_original, gray_template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)


top_left = max_loc
bottom_right = (top_left[0]+height, top_left[1]+width)
cv2.rectangle(aranilan_resim_org, top_left, bottom_right, (0,0,255), 5)

cv2.imshow("Bulunan Kare", aranilan_resim_org)
cv2.waitKey(0)

cv2.destroyAllWindows()
