import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

df = pd.read_csv('examp10.txt', header=None)
a, b = [], []
for i in range(len(df.loc[:, 2])):
    a.append(float(df.loc[:, 2][i].split(';')[0]))
    b.append(float(df.loc[:, 2][i].split(';')[1]))
a, b = np.array(a), np.array(b)
names = df.columns.tolist()
names.append(names[-1] + 1)
df.drop([2], inplace=True, axis = 1)
df.insert(2, 2, a)
df.insert(3, 2.1, b)
df.columns = names

points = []
rob = []
for i in range(len(df)):
    points.append(np.array(list(df.loc[i, 3:])))
    rob.append(list(df.loc[i, :2]))
points = np.array(points)
rob = np.array(rob)

angles = np.radians(np.linspace(-120, 120, 681))
points_x = []
points_y = []
traectoria_x = []
traectoria_y = []
for i in range(100):
    traectoria_x.append(rob[i][0])
    traectoria_y.append(rob[i][1])
    for j in range(len(points[i])):
        if 1 < points[i][j] < 4.5:
            x = rob[i][0] + 0.3 * math.cos(rob[i][2]) + points[i][j] * math.cos(rob[i][2] - angles[j])
            y = rob[i][1] + 0.3 * math.sin(rob[i][2]) + points[i][j] * math.sin(rob[i][2] - angles[j])
            points_x.append(x)
            points_y.append(y)

fig = plt.figure()
map_ = plt.scatter(points_x, points_y,  s=2)
plt.xlim(-2, 12)
plt.ylim(-7, 4)
fig.canvas.draw()
img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
img  = cv2.cvtColor(img.reshape(fig.canvas.get_width_height()[::-1] + (3,), ),cv2.COLOR_RGBA2BGR)
img = img[60:420,83:575,]
t = np.zeros(img.shape)
t2 = t.copy()


fig = plt.figure()
plt.scatter(traectoria_x, traectoria_y, color = 'red', s = 1)
plt.xlim(-2, 12)
plt.ylim(-7, 4)
fig.canvas.draw()
img_t = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
img_t = cv2.cvtColor(img_t.reshape(fig.canvas.get_width_height()[::-1] + (3,), ),cv2.COLOR_RGBA2BGR)
img_t = img_t[60:420,83:575,]

# Делаем красный на черном фоне чтобы можно было наложить траекторию на карту
for i in range(len(img_t)):
    for j in range(len(img_t[i])):
        # print(img_t[i][j] == [255,255,255])
        if list(img_t[i][j]) == [255,255,255]:
            img_t[i][j] = [0,0,0]

im = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(img, 120, 950)

cnts, hierarchy = cv2.findContours(edged,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(t, cnts, -1, (0, 0, 255), 2)

conturs= []
for cnt in cnts:

    epsilon = 0.002*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)

    conturs.append(approx) if len(approx) >35 else None


cv2.drawContours(t2, conturs, -1, (255, 255, 0), 1)

cv2.imshow("3", cv2.resize(t2+img_t, (700,700)))
cv2.waitKey(0)
cv2.destroyAllWindows()