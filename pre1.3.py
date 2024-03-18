import sympy
import numpy as np
import matplotlib.pyplot as plt


def getDerivada(variables, funcion):
    respectoX1 = sympy.diff(funcion, variables[0])
    respectoX2 = sympy.diff(funcion, variables[1])
    return respectoX1, respectoX2


x1, x2 = sympy.symbols('x1 x2')
func = 10 - sympy.exp(-(x1 ** 2 + 3 * x2 ** 2))
derivadas = getDerivada((x1, x2), func)

# Modificar los límites para los valores iniciales de x1 y x2
x1_inicial = np.random.uniform(-1, 1)
x2_inicial = np.random.uniform(-1, 1)

# Modificar el learning rate
lr = 0.1
num_iteraciones = 20
x1_historial = []
x2_historial = []

for i in range(num_iteraciones):
    grad_x1, grad_x2 = derivadas

    # Actualizar los valores de x1 y x2
    x1_inicial -= lr * grad_x1.subs({x1: x1_inicial, x2: x2_inicial})
    x2_inicial -= lr * grad_x2.subs({x1: x1_inicial, x2: x2_inicial})

    x1_historial.append(x1_inicial)
    x2_historial.append(x2_inicial)

    valor_funcion = func.subs({x1: x1_inicial, x2: x2_inicial}).evalf()
    print(f"Iteración {i + 1}: x1 = {x1_inicial:.6f}, x2 = {x2_inicial:.6f}, f(x1, x2) = {valor_funcion:.6f}")

print(f"Valor mínimo estimado: f({x1_inicial}, {x2_inicial}) = {func.subs({x1: x1_inicial, x2: x2_inicial})}")

# Graficar la función y el proceso de optimización
x1_val = np.linspace(-1, 1, 400)
x2_val = np.linspace(-1, 1, 400)
X1, X2 = np.meshgrid(x1_val, x2_val)
Z = 10 - np.exp(-(X1 ** 2 + 3 * X2 ** 2))

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)

ax1.plot_surface(X1, X2, Z, cmap='viridis')
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('f(X1, X2)')
ax1.set_title('Función')

contour_levels = np.linspace(np.min(Z), np.max(Z), 20)
ax2.contour(X1, X2, Z, levels=contour_levels, cmap='viridis')
ax2.scatter(x1_historial, x2_historial, color='red', marker='x', label='Descenso del Gradiente')
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_title('Descenso del Gradiente')
ax2.legend()

plt.show()
 