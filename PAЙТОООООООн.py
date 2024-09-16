#================================================ЛинияМеждуДвумяТочками================================================= 
# img = Image.new('RGB', (1000, 900), 'black')
# draw_flag(img)
# imshow(np.asarray(img))
# img.save('test.png')

# import matplotlib.pyplot as plt  # Импортируем как plt для вызова plt.show()
# from matplotlib.pyplot import imshow
# from PIL import Image
# import numpy as np

# # Функция для рисования отрезка между двумя точками
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

# # Функция для рисования сетки
# def draw_grid(img, step, color):
#     width, height = img.size
#     # Вертикальные линии
#     for x in range(0, width, step):
#         draw_line(img, x, 0, x, height-1, color)
#     # Горизонтальные линии
#     for y in range(0, height, step):
#         draw_line(img, 0, y, width-1, y, color)

# # Получаем координаты от пользователя
# x0 = int(input("Input first x0-coordinate: "))
# y0 = int(input("Input first y0-coordinate: "))
# x1 = int(input("Input second x1-coordinate: "))
# y1 = int(input("Input second y1-coordinate: "))

# # Создаём пустое изображение
# img = Image.new('RGB', (1000, 900), 'white')

# # Рисуем сетку с шагом 50 пикселей 
# draw_grid(img, 50, (200, 200, 200))  # Серая сетка

# # Рисуем линию на основе введённых пользователем координат
# draw_line(img, x0, y0, x1, y1, (0, 0, 0))

# # Показываем изображение
# imshow(np.asarray(img))

# # Явно отображаем изображение
# plt.show()

# # Сохраняем изображение
# img.save('Linia.png')

# #===================================Окружность=====================================================================
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np

# # Функция для рисования пикселя на изображении с симметрией окружности
# def draw_circle_pixels(img, xc, yc, x, y, color):
#     img.putpixel((xc + x, yc + y), color)
#     img.putpixel((xc - x, yc + y), color)
#     img.putpixel((xc + x, yc - y), color)
#     img.putpixel((xc - x, yc - y), color)
#     img.putpixel((xc + y, yc + x), color)
#     img.putpixel((xc - y, yc + x), color)
#     img.putpixel((xc + y, yc - x), color)
#     img.putpixel((xc - y, yc - x), color)

# # Алгоритм Брезенхема для рисования окружности
# def draw_circle(img, xc, yc, r, color):
#     x = 0
#     y = r
#     d = 3 - 2 * r
#     draw_circle_pixels(img, xc, yc, x, y, color)
    
#     while y >= x:
#         x += 1
        
#         # Проверяем текущее значение дискриминанта
#         if d > 0:
#             y -= 1
#             d = d + 4 * (x - y) + 10
#         else:
#             d = d + 4 * x + 6
        
#         # Рисуем симметричные пиксели окружности
#         draw_circle_pixels(img, xc, yc, x, y, color)

# # Функция для рисования сетки
# def draw_grid(img, step, color):
#     width, height = img.size
#     # Вертикальные линии
#     for x in range(0, width, step):
#         draw_line(img, x, 0, x, height - 1, color)
#     # Горизонтальные линии
#     for y in range(0, height, step):
#         draw_line(img, 0, y, width - 1, y, color)

# # Получаем радиус
# r = int(input("Enter radius: "))

# # Размер изображения должен быть чуть больше радиуса, чтобы окружность почти касалась краёв
# image_size = r * 2 + 20  # Добавляем по 10 пикселей отступа с каждой стороны

# # Центр изображения
# xc, yc = image_size // 2, image_size // 2

# # Создаём пустое изображение с белым фоном
# img = Image.new('RGB', (image_size, image_size), 'white')

# # Рисуем окружность
# draw_circle(img, xc, yc, r, (255, 0, 0))

# plt.imshow(np.asarray(img))
# plt.show()

# img.save('Krug.png')


#===========Циферблат==============
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math

# Функция для рисования пикселя на изображении с симметрией окружности
def draw_circle_pixels(img, xc, yc, x, y, color):
    img.putpixel((xc + x, yc + y), color)
    img.putpixel((xc - x, yc + y), color)
    img.putpixel((xc + x, yc - y), color)
    img.putpixel((xc - x, yc - y), color)
    img.putpixel((xc + y, yc + x), color)
    img.putpixel((xc - y, yc + x), color)
    img.putpixel((xc + y, yc - x), color)
    img.putpixel((xc - y, yc - x), color)

# Алгоритм Брезенхема для рисования окружности
def draw_circle(img, xc, yc, r, color):
    x = 0
    y = r
    d = 3 - 2 * r
    draw_circle_pixels(img, xc, yc, x, y, color)
    
    while y >= x:
        x += 1
        
        # Проверяем текущее значение дискриминанта
        if d > 0:
            y -= 1
            d = d + 4 * (x - y) + 10
        else:
            d = d + 4 * x + 6
        
        # Рисуем симметричные пиксели окружности
        draw_circle_pixels(img, xc, yc, x, y, color)

# Функция для рисования линии
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

# Функция для рисования сетки
def draw_grid(img, step, color):
    width, height = img.size
    # Вертикальные линии
    for x in range(0, width, step):
        draw_line(img, x, 0, x, height - 1, color)
    # Горизонтальные линии
    for y in range(0, height, step):
        draw_line(img, 0, y, width - 1, y, color)

# Функция для рисования засечек на окружности
def draw_ticks(img, xc, yc, r, num_ticks, color):
    for i in range(num_ticks):
        angle = 2 * math.pi * i / num_ticks
        x_end = int(xc + r * math.cos(angle))
        y_end = int(yc - r * math.sin(angle))  # Вычисляем точку по окружности
        x_start = int(xc + (r - 10) * math.cos(angle))  # Засечка немного внутри окружности
        y_start = int(yc - (r - 10) * math.sin(angle))
        draw_line(img, x_start, y_start, x_end, y_end, color)

# Получаем радиус
r = int(input("Enter radius: "))

# Размер изображения должен быть чуть больше радиуса, чтобы окружность почти касалась краёв
image_size = r * 2 + 20  # Добавляем по 10 пикселей отступа с каждой стороны

# Центр изображения
xc, yc = image_size // 2, image_size // 2

# Создаём пустое изображение с белым фоном
img = Image.new('RGB', (image_size, image_size), 'white')

# Рисуем сетку с шагом 50 пикселей
draw_grid(img, 50, (200, 200, 200))  # Серая сетка

# Рисуем окружность
draw_circle(img, xc, yc, r, (255, 0, 0))  # Красная окружность

# Рисуем засечки на окружности (например, 12 засечек, как на часах)
draw_ticks(img, xc, yc, r, 12, (0, 0, 0))  # Черные засечки

plt.imshow(np.asarray(img))
plt.show()

img.save('Clock.png')

