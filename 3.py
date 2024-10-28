# import matplotlib.pyplot as plt
# import numpy as np

# def fill_polygon(vertices):
#     # Sozdaniye pustogo polya
#     x_min, x_max = min(vertices[:, 0]), max(vertices[:, 0])
#     y_min, y_max = min(vertices[:, 1]), max(vertices[:, 1])
    
#     # Spisok 4 zapolneniya
#     fill_points = []
    
#     # Prohod po gorizontal lines
#     for y in range(int(y_min), int(y_max) + 1):
#         intersections = []
        
#         # Nahodim peresecheniya s ryobrami
#         for i in range(len(vertices)):
#             v1, v2 = vertices[i], vertices[(i + 1) % len(vertices)]
#             if (v1[1] > y) != (v2[1] > y):
#                 x = (v2[0] - v1[0]) * (y - v1[1]) / (v2[1] - v1[1]) + v1[0]
#                 intersections.append(x)
        
#         # Sortirue, peresecheniya
#         intersections.sort()
        
#         # Zapolnyaem Oblast'
#         for i in range(0, len(intersections), 2):
#             fill_points.append((intersections[i], y))
#             fill_points.append((intersections[i + 1], y))
    
#     return fill_points

# # Vershini
# vertices = np.array([(1, 1), (5, 0.5), (4, 4), (2, 3), (1, 4)])
# fill_points = fill_polygon(vertices)

# # Visualizatsia
# plt.fill(vertices[:, 0], vertices[:, 1], 'lightgrey')
# plt.xlim(0, 6)
# plt.ylim(0, 5)
# plt.show()







import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def create_polygon_image(vertices, shape=(100, 100)):
    fig, ax = plt.subplots()
    fig.set_size_inches(shape[0] / fig.dpi, shape[1] / fig.dpi)
    ax.set_xlim(0, shape[1])
    ax.set_ylim(0, shape[0])
    ax.invert_yaxis()
    ax.axis('off')

    # Рисуем многоугольник
    polygon = Polygon(vertices, closed=True, edgecolor='black', facecolor='white')
    ax.add_patch(polygon)

    # Преобразуем в массив
    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').reshape(shape[0], shape[1], 4)
    plt.close(fig)

    return image[:, :, :3].copy()

def is_background(color, threshold=68):
    # Считаем белыми пиксели с яркостью выше 68
    return np.mean(color) > threshold

def boundary_fill(image, x, y, fill_color):
    if not is_background(image[x, y]):
        return

    stack = [(x, y)]

    while stack:
        cx, cy = stack.pop()
        if is_background(image[cx, cy]):
            image[cx, cy] = fill_color

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1] and is_background(image[nx, ny]):
                    stack.append((nx, ny))

# Определяем вершины 7-угольника (хочу вот семиугольник закрасить)
vertices = [(30, 20), (70, 15), (90, 40), (80, 70), (50, 90), (20, 70), (10, 40)]
image = create_polygon_image(vertices)

fill_color = np.array([139, 0, 0], dtype=np.uint8)  # ЦВЕТ КРОВИ

# Убираем темные серые пиксели между границей и заливкой
gray_threshold = 100
image[np.all((image[:, :, 0] < gray_threshold) & 
             (image[:, :, 1] < gray_threshold) & 
             (image[:, :, 2] < gray_threshold), axis=-1)] = [255, 255, 255]

# Отображаем исходное изображение
plt.subplot(1, 2, 1)
plt.title("Исходное изображение")
plt.imshow(image)

# Применяем Boundary Fill с начальной точкой внутри многоугольника
boundary_fill(image, 50, 50, fill_color)

# Отображаем результат
plt.subplot(1, 2, 2)
plt.title("После Boundary Fill")
plt.imshow(image)
plt.show()
