import cv2
import numpy as np

filename = './testing_img/integral x+3 dx.png'
src = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
cv2.imshow('gray', src)
k = cv2.waitKey(0)
cv2.destroyAllWindows()

ret , binary = cv2.threshold(src,125,255,cv2.THRESH_BINARY_INV) #영상 이진화
cv2.imshow('binary',binary)
k = cv2.waitKey(0)
cv2.destroyAllWindows()

contours , hierarchy = cv2.findContours(binary , cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE) #외곽선 검출
color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR) #이진화 이미지를 color이미지로 복사
cv2.drawContours(color , contours , -1 , (0,255,0),3) #초록색으로 외곽선을 그린다

bR_arr = []
digit_arr = []
digit_arr2 = []
count = 0

#검출한 외곽선에 사각형을 그려서 배열에 추가
for i in range(len(contours)) :
    bin_tmp = binary.copy()
    x,y,w,h = cv2.boundingRect(contours[i])
    bR_arr.append([x,y,w,h])

bR_arr = sorted(bR_arr, key=lambda num : num[0], reverse = False) #순서대로 값 나누기 위해 x값을 기준으로 배열을 정렬

for x,y,w,h in bR_arr :
    tmp_y = bin_tmp[y - 2:y + h + 2, x - 2:x + w + 2].shape[0]
    tmp_x = bin_tmp[y - 2:y + h + 2, x - 2:x + w + 2].shape[1]

    if tmp_x and tmp_y > 10:
        count += 1
        cv2.rectangle(color, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 0, 255), 1)
        digit_arr.append(bin_tmp[y - 2:y + h + 2, x - 2:x + w + 2])

        if count == 1:
            digit_arr2.append(digit_arr)
            digit_arr = []
            count = 0

cv2.imshow('contours',color)
k = cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range(0,len(digit_arr2)) : #이미지 자르고 45x45로 저장
    for j in range(len(digit_arr2[i])) :
        count += 1
        if i == 0 :
            width = digit_arr2[i][j].shape[1]
            height = digit_arr2[i][j].shape[0]
            tmp = (height - width)/2
            mask = np.zeros((height,height))
            mask[0:height,int(tmp):int(tmp)+width] = digit_arr2[i][j]
            digit_arr2[i][j] = cv2.resize(mask,(45,45))
        else:
            digit_arr2[i][j] = cv2.resize(digit_arr2[i][j],(45,45))

        cv2.imshow('gray', digit_arr2[i][j])
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
        ret, det = cv2.threshold(digit_arr2[i][j], 0, 255, cv2.THRESH_BINARY_INV) #이진화 이미지 반전
        cv2.imshow('white', det)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('./testing_img_extracting/integral x+3 dx/' + str(i + 1) + '.jpg', det)



















