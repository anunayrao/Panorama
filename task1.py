print("---------------------------------------------------")	
print("Implementing Task 1 : Image Features and Homography")
print("---------------------------------------------------")	
import cv2
import numpy as np

np.random.seed(sum([ord(c) for  c in 'anunayra']))

img1 = cv2.imread('mountain1.jpg',0)
img2 = cv2.imread('mountain2.jpg',0)
image1 = cv2.imread('mountain1.jpg')
image2 =  cv2.imread('mountain2.jpg')
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

im1 = cv2.drawKeypoints(image1, kp1, None)
im2 = cv2.drawKeypoints(image2, kp2, None)
cv2.imwrite("task1_sift1.jpg", im1)
cv2.imwrite("task1_sift2.jpg",im2)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

#cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
cv2.imwrite("task1_matches_knn.jpg",img3)

good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)

good = np.random.choice(good,10)
if len(good)>=10:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(image1,kp1,image2,kp2,good,None,**draw_params)
cv2.imwrite("task1_matches.jpg",img3)

def warpImages(img1, img2, H):
	rows1, cols1 = img1.shape[:2]
	rows2, cols2 = img2.shape[:2]

	list_of_points_1 = np.float32([[0,0], [0,rows1], [cols1, rows1], [cols1,0]]).reshape(-1,1,2)
	temp_points = np.float32([[0,0], [0,rows2], [cols2, rows2], [cols2,0]]).reshape(-1,1,2)

	list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
	list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

	[x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
	[x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

	translation_dist = [-x_min, -y_min]
	H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])

	output_img = cv2.warpPerspective(img1, H_translation.dot(H), (x_max - x_min, y_max - y_min))
	output_img[translation_dist[1]:rows1+translation_dist[1],translation_dist[0]:cols1+translation_dist[0]] = img2
	return output_img

result = warpImages(image1,image2,M)
cv2.imwrite("task1_pano.jpg",result)
print("---------------------------------------------------")
print("---------Homography Matrix Generated::-------------")
print("---------------------------------------------------")
print(M)
