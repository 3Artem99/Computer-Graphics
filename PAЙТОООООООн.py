#================================================����������������������================================================= 
# img = Image.new('RGB', (1000, 900), 'black')
# draw_flag(img)
# imshow(np.asarray(img))
# img.save('test.png')

# import matplotlib.pyplot as plt  # ����������� ��� plt ��� ������ plt.show()
# from matplotlib.pyplot import imshow
# from PIL import Image
# import numpy as np

# # ������� ��� ��������� ������� ����� ����� �������
# def draw_line(img, x0, y0, x1, y1, color):
#     dx = abs(x1 - x0)
#     dy = abs(y1 - y0)
#     sx = 1 if x0 < x1 else -1
#     sy = 1 if y0 < y1 else -1
#     err = dx - dy

#     while True:
#         img.putpixel((x0, y0), color)
#         if x0 == x1 and y0 == y1:
#             break
#         e2 = err * 2
#         if e2 > -dy:
#             err -= dy
#             x0 += sx
#         if e2 < dx:
#             err += dx
#             y0 += sy

# # ������� ��� ��������� �����
# def draw_grid(img, step, color):
#     width, height = img.size
#     # ������������ �����
#     for x in range(0, width, step):
#         draw_line(img, x, 0, x, height-1, color)
#     # �������������� �����
#     for y in range(0, height, step):
#         draw_line(img, 0, y, width-1, y, color)

# # �������� ���������� �� ������������
# x0 = int(input("Input first x0-coordinate: "))
# y0 = int(input("Input first y0-coordinate: "))
# x1 = int(input("Input second x1-coordinate: "))
# y1 = int(input("Input second y1-coordinate: "))

# # ������ ������ �����������
# img = Image.new('RGB', (1000, 900), 'white')

# # ������ ����� � ����� 50 �������� 
# draw_grid(img, 50, (200, 200, 200))  # ����� �����

# # ������ ����� �� ������ �������� ������������� ���������
# draw_line(img, x0, y0, x1, y1, (0, 0, 0))

# # ���������� �����������
# imshow(np.asarray(img))

# # ���� ���������� �����������
# plt.show()

# # ��������� �����������
# img.save('Linia.png')

# #===================================����������=====================================================================
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np

# # ������� ��� ��������� ������� �� ����������� � ���������� ����������
# def draw_circle_pixels(img, xc, yc, x, y, color):
#     img.putpixel((xc + x, yc + y), color)
#     img.putpixel((xc - x, yc + y), color)
#     img.putpixel((xc + x, yc - y), color)
#     img.putpixel((xc - x, yc - y), color)
#     img.putpixel((xc + y, yc + x), color)
#     img.putpixel((xc - y, yc + x), color)
#     img.putpixel((xc + y, yc - x), color)
#     img.putpixel((xc - y, yc - x), color)

# # �������� ���������� ��� ��������� ����������
# def draw_circle(img, xc, yc, r, color):
#     x = 0
#     y = r
#     d = 3 - 2 * r
#     draw_circle_pixels(img, xc, yc, x, y, color)
    
#     while y >= x:
#         x += 1
        
#         # ��������� ������� �������� �������������
#         if d > 0:
#             y -= 1
#             d = d + 4 * (x - y) + 10
#         else:
#             d = d + 4 * x + 6
        
#         # ������ ������������ ������� ����������
#         draw_circle_pixels(img, xc, yc, x, y, color)

# # ������� ��� ��������� �����
# def draw_grid(img, step, color):
#     width, height = img.size
#     # ������������ �����
#     for x in range(0, width, step):
#         draw_line(img, x, 0, x, height - 1, color)
#     # �������������� �����
#     for y in range(0, height, step):
#         draw_line(img, 0, y, width - 1, y, color)

# # �������� ������
# r = int(input("Enter radius: "))

# # ������ ����������� ������ ���� ���� ������ �������, ����� ���������� ����� �������� ����
# image_size = r * 2 + 20  # ��������� �� 10 �������� ������� � ������ �������

# # ����� �����������
# xc, yc = image_size // 2, image_size // 2

# # ������ ������ ����������� � ����� �����
# img = Image.new('RGB', (image_size, image_size), 'white')

# # ������ ����������
# draw_circle(img, xc, yc, r, (255, 0, 0))

# plt.imshow(np.asarray(img))
# plt.show()

# img.save('Krug.png')


#===========���������==============
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math

# ������� ��� ��������� ������� �� ����������� � ���������� ����������
def draw_circle_pixels(img, xc, yc, x, y, color):
    img.putpixel((xc + x, yc + y), color)
    img.putpixel((xc - x, yc + y), color)
    img.putpixel((xc + x, yc - y), color)
    img.putpixel((xc - x, yc - y), color)
    img.putpixel((xc + y, yc + x), color)
    img.putpixel((xc - y, yc + x), color)
    img.putpixel((xc + y, yc - x), color)
    img.putpixel((xc - y, yc - x), color)

# �������� ���������� ��� ��������� ����������
def draw_circle(img, xc, yc, r, color):
    x = 0
    y = r
    d = 3 - 2 * r
    draw_circle_pixels(img, xc, yc, x, y, color)
    
    while y >= x:
        x += 1
        
        # ��������� ������� �������� �������������
        if d > 0:
            y -= 1
            d = d + 4 * (x - y) + 10
        else:
            d = d + 4 * x + 6
        
        # ������ ������������ ������� ����������
        draw_circle_pixels(img, xc, yc, x, y, color)

# ������� ��� ��������� �����
def draw_line(img, x0, y0, x1, y1, color):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        img.putpixel((x0, y0), color)
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

# ������� ��� ��������� �����
def draw_grid(img, step, color):
    width, height = img.size
    # ������������ �����
    for x in range(0, width, step):
        draw_line(img, x, 0, x, height - 1, color)
    # �������������� �����
    for y in range(0, height, step):
        draw_line(img, 0, y, width - 1, y, color)

# ������� ��� ��������� ������� �� ����������
def draw_ticks(img, xc, yc, r, num_ticks, color):
    for i in range(num_ticks):
        angle = 2 * math.pi * i / num_ticks
        x_end = int(xc + r * math.cos(angle))
        y_end = int(yc - r * math.sin(angle))  # ��������� ����� �� ����������
        x_start = int(xc + (r - 10) * math.cos(angle))  # ������� ������� ������ ����������
        y_start = int(yc - (r - 10) * math.sin(angle))
        draw_line(img, x_start, y_start, x_end, y_end, color)

# �������� ������
r = int(input("Enter radius: "))

# ������ ����������� ������ ���� ���� ������ �������, ����� ���������� ����� �������� ����
image_size = r * 2 + 20  # ��������� �� 10 �������� ������� � ������ �������

# ����� �����������
xc, yc = image_size // 2, image_size // 2

# ������ ������ ����������� � ����� �����
img = Image.new('RGB', (image_size, image_size), 'white')

# ������ ����� � ����� 50 ��������
draw_grid(img, 50, (200, 200, 200))  # ����� �����

# ������ ����������
draw_circle(img, xc, yc, r, (255, 0, 0))  # ������� ����������

# ������ ������� �� ���������� (��������, 12 �������, ��� �� �����)
draw_ticks(img, xc, yc, r, 12, (0, 0, 0))  # ������ �������

plt.imshow(np.asarray(img))
plt.show()

img.save('Clock.png')

