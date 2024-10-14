# Computer-Graphics

## 1.1 Создать программу, которая рисует отрезок между двумя точками, заданными пользователем

```sh
import matplotlib.pyplot as plt  # Импортируем как plt для вызова plt.show()
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np

# Функция для рисования отрезка между двумя точками
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
        draw_line(img, x, 0, x, height-1, color)
    # Горизонтальные линии
    for y in range(0, height, step):
        draw_line(img, 0, y, width-1, y, color)

# Получаем координаты от пользователя
x0 = int(input("Input first x0-coordinate: "))
y0 = int(input("Input first y0-coordinate: "))
x1 = int(input("Input second x1-coordinate: "))
y1 = int(input("Input second y1-coordinate: "))

# Создаём пустое изображение
img = Image.new('RGB', (1000, 900), 'white')

# Рисуем сетку с шагом 50 пикселей 
draw_grid(img, 50, (200, 200, 200))  # Серая сетка

# Рисуем линию на основе введённых пользователем координат
draw_line(img, x0, y0, x1, y1, (0, 0, 0))

# Показываем изображение
imshow(np.asarray(img))

# Явно отображаем изображение
plt.show()

# Сохраняем изображение
img.save('Linia.png')
```

## 1.2 Создать программу, которая рисует окружность с заданным пользователем радиусом

```sh
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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

# Функция для рисования сетки
def draw_grid(img, step, color):
    width, height = img.size
    # Вертикальные линии
    for x in range(0, width, step):
        draw_line(img, x, 0, x, height - 1, color)
    # Горизонтальные линии
    for y in range(0, height, step):
        draw_line(img, 0, y, width - 1, y, color)

# Получаем радиус
r = int(input("Enter radius: "))

# Размер изображения должен быть чуть больше радиуса, чтобы окружность почти касалась краёв
image_size = r * 2 + 20  # Добавляем по 10 пикселей отступа с каждой стороны

# Центр изображения
xc, yc = image_size // 2, image_size // 2

# Создаём пустое изображение с белым фоном
img = Image.new('RGB', (image_size, image_size), 'white')

# Рисуем окружность
draw_circle(img, xc, yc, r, (255, 0, 0))

# Показываем изображение
plt.imshow(np.asarray(img))
plt.show()

# Сохраняем изображение
img.save('Krug.png')
```
 ## 1.3 Циферблат

```sh
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

# Функция для рисования линии (используется для сетки и засечек)
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

# Рисуем сетку
draw_grid(img, 50, (200, 200, 200))  # Серая сетка

# Рисуем окружность
draw_circle(img, xc, yc, r, (255, 0, 0))  # Красная окружность

# Рисуем засечки на окружности (например, 12 засечек, как на часах)
draw_ticks(img, xc, yc, r, 12, (0, 0, 0))  # Черные засечки

plt.imshow(np.asarray(img))
plt.show()

img.save('Clock.png')

``` 

## Правлю 1.2

```sh
import matplotlib.pyplot as plt
import numpy as np

def bresenham_circle(radius):
    x = 0
    y = radius
    d = 3 - 2 * radius
    points = []

    def draw_circle_points(x, y):
        points.extend([
            (x, y), (-x, y), (x, -y), (-x, -y),
            (y, x), (-y, x), (y, -x), (-y, -x)
        ])

    while x <= y:
        draw_circle_points(x, y)
        if d <= 0:
            d = d + 4 * x + 6
        else:
            d = d + 4 * (x - y) + 10
            y -= 1
        x += 1

    return points

def plot_circle(radius):
    points = bresenham_circle(radius)
    
    # Удаляем дубликаты и сортируем точки по углам
    unique_points = list(set(points))
    unique_points.sort(key=lambda p: np.arctan2(p[1], p[0]))

    # Добавляем первую точку в конец для замыкания контура
    unique_points.append(unique_points[0])
    
    # Разворачиваем список точек в x и y для построения графика
    x_coords = [point[0] for point in unique_points]
    y_coords = [point[1] for point in unique_points]

    # Рисуем
    plt.plot(x_coords, y_coords, color='black')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'My MEGA GIPER ULTRA CIRCLE {radius}')
    plt.grid(True)
    plt.show()

#Enter
radius = int(input("R: "))
plot_circle(radius)
```

## Правлю 1.3

```sh
import matplotlib.pyplot as plt
import numpy as np

def bresenham_circle(radius):
    x = 0
    y = radius
    d = 3 - 2 * radius
    points = []

    def draw_circle_points(x, y):
        points.extend([
            (x, y), (-x, y), (x, -y), (-x, -y),
            (y, x), (-y, x), (y, -x), (-y, -x)
        ])

    while x <= y:
        draw_circle_points(x, y)
        if d <= 0:
            d = d + 4 * x + 6
        else:
            d = d + 4 * (x - y) + 10
            y -= 1
        x += 1

    return points

def plot_circle_with_ticks(radius, num_ticks):
    points = bresenham_circle(radius)
    
    # Удаляем дубликаты и сортируем точки по углам
    unique_points = list(set(points))
    unique_points.sort(key=lambda p: np.arctan2(p[1], p[0]))

    # Добавляем первую точку в конец для замыкания контура
    unique_points.append(unique_points[0])
    
    # Разворачиваем список точек в x и y для построения графика
    x_coords = [point[0] for point in unique_points]
    y_coords = [point[1] for point in unique_points]

    fig, ax = plt.subplots()
    ax.plot(x_coords, y_coords, color='blue')

    # Добавляем засечки как на циферблате
    tick_length = 0.1 * radius
    for i in range(num_ticks):
        angle = 2 * np.pi * i / num_ticks
        x_tick_start = (radius - tick_length) * np.cos(angle)
        y_tick_start = (radius - tick_length) * np.sin(angle)
        x_tick_end = radius * np.cos(angle)
        y_tick_end = radius * np.sin(angle)
        
        # Рисуем засечки
        ax.plot([x_tick_start, x_tick_end], [y_tick_start, y_tick_end], color='red', lw=1.5)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Bresenham BASING {radius} and {num_ticks} ULTRAIMBA')
    ax.grid(True)
    plt.show()

radius = int(input("R: "))
num_ticks = 12
plot_circle_with_ticks(radius, num_ticks)
```

## Реализация алгоритма Сезерленда-Коэна
```sh
import matplotlib.pyplot as plt

# Opredelyaem kody regionov dlya otsecheniya
INSIDE = 0  # 0000
LEFT = 1    # 0001
RIGHT = 2   # 0010
BOTTOM = 4  # 0100
TOP = 8     # 1000

# Funktsiya dlya vychisleniya koda tochki
def compute_code(x, y, x_min, y_min, x_max, y_max):
    code = INSIDE
    if x < x_min:    # Sleva ot okna
        code |= LEFT
    elif x > x_max:  # Sprava ot okna
        code |= RIGHT
    if y < y_min:    # Nizhe okna
        code |= BOTTOM
    elif y > y_max:  # Vyshe okna
        code |= TOP
    return code

# Algoritm Sazerlenda-Koena dlya otsecheniya otrezkov
def cohen_sutherland_clip(x1, y1, x2, y2, x_min, y_min, x_max, y_max):
    code1 = compute_code(x1, y1, x_min, y_min, x_max, y_max)
    code2 = compute_code(x2, y2, x_min, y_min, x_max, y_max)
    accept = False

    while True:
        if code1 == 0 and code2 == 0:  # Obe tochki vnutri okna
            accept = True
            break
        elif code1 & code2 != 0:  # Obe tochki snaruji, otrezok vne okna
            break
        else:
            x, y = 0.0, 0.0
            # Vyberaem tochku, nahodyashchuyusya snaruji
            if code1 != 0:
                code_out = code1
            else:
                code_out = code2

            # Naydemy peresechenie s granitsami okna
            if code_out & TOP:  # Peresechenie s verhney granitsey
                x = x1 + (x2 - x1) * (y_max - y1) / (y2 - y1)
                y = y_max
            elif code_out & BOTTOM:  # Peresechenie s nizhney granitsey
                x = x1 + (x2 - x1) * (y_min - y1) / (y2 - y1)
                y = y_min
            elif code_out & RIGHT:  # Peresechenie s pravoy granitsey
                y = y1 + (y2 - y1) * (x_max - x1) / (x2 - x1)
                x = x_max
            elif code_out & LEFT:  # Peresechenie s levoy granitsey
                y = y1 + (y2 - y1) * (x_min - x1) / (x2 - x1)
                x = x_min

            # Zamenim tochku snaruji na tochku peresecheniya i pereschitaem kod
            if code_out == code1:
                x1, y1 = x, y
                code1 = compute_code(x1, y1, x_min, y_min, x_max, y_max)
            else:
                x2, y2 = x, y
                code2 = compute_code(x2, y2, x_min, y_min, x_max, y_max)

    if accept:
        return x1, y1, x2, y2
    else:
        return None

# Funktsiya dlya vizualizatsii otsecheniya linii
def draw_plot(lines, x_min, y_min, x_max, y_max):
    fig, ax = plt.subplots()

    # Risuyem okno otsecheniya
    ax.plot([x_min, x_max, x_max, x_min, x_min],
            [y_min, y_min, y_max, y_max, y_min], 'k-', lw=2)

    # Risuyem otrezki do otsecheniya
    for line in lines:
        x1, y1, x2, y2 = line
        ax.plot([x1, x2], [y1, y2], 'r--', label='Do otsecheniya')

    # Otsechenie linii
    for line in lines:
        result = cohen_sutherland_clip(*line, x_min, y_min, x_max, y_max)
        if result:
            x1, y1, x2, y2 = result
            ax.plot([x1, x2], [y1, y2], 'g-', lw=2, label='Posle otsecheniya')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Otsechenie otrezkov algoritmom Sazerlenda-Koena')
    plt.grid(True)
    plt.show()

# Primer ispolzovaniya
if __name__ == "__main__":
    # Zadaem okno otsecheniya
    x_min, y_min = 10, 10
    x_max, y_max = 100, 100

    # Otrezki dlya otsecheniya
    lines = [
        (5, 5, 120, 120),
        (50, 50, 60, 70),
        (70, 80, 120, 140),
        (10, 110, 110, 10),
        (0, 50, 200, 50)
    ]

    # Vizualizatsiya
    draw_plot(lines, x_min, y_min, x_max, y_max)
```
