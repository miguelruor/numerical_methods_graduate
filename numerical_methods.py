import numpy as np
from typing import Tuple, Optional, List, Callable

####################### Álgebra lineal


def solveUpperTriangular(A: np.array, b: np.array, TOL: float = 10 ** (-8)) -> np.array:
    """Resuelve Ax = b cuando A es una matriz triangular superior de nxn, y b es un vector de tamaño n, mediante sustitución regresiva.
    Regresa la solución x si existe, y en otro caso levanta una excepción.

    :param A: Matriz triangular superior de nxn
    :type A: np.array
    :param b: Vector de tamaño n
    :type b: np.array
    :param TOL: Toleracia, i.e., si |x|<TOL, entonces x se considera como 0
    :type TOL: float
    :return: Vector solución x de tamaño n (si existe).
    :rtype: np.array
    """

    n: int = b.size
    x: np.array = np.zeros(n)

    for i in reversed(range(n)):
        if abs(A[i, i]) < TOL:  # checar si A[i,i] es prácticamente 0
            raise Exception("No existe una solución única al sistema")

        s = (
            A[i, i + 1 :] @ x[i + 1 :]
        )  # \sum_{j=i+1}^{n-1} a_{ij}x_j corresponde al producto punto entre A[i,i+1:] y x[i+1:]
        # notar que cuando i == n-1, el producto punto (de arrays vacíos) da 0

        x[i] = (b[i] - s) / A[i, i]

    return x


def solveLowerTriangular(A: np.array, b: np.array, TOL: float = 10 ** (-8)) -> np.array:
    """Resuelve Ax = b cuando A es una matriz triangular inferior de nxn, y b es un vector de tamaño n, mediante sustitución progresiva.
    Regresa la solución x si existe, y en otro caso levanta una excepción.

    :param A: Matriz triangular inferior de nxn
    :type A: np.array
    :param b: Vector de tamaño n
    :type b: np.array
    :param TOL: Toleracia, i.e., si |x|<TOL, entonces x se considera como 0
    :type TOL: float
    :return: Vector solución x de tamaño n (si existe).
    :rtype: np.array
    """

    n: int = b.size
    x: np.array = np.zeros(n)

    for i in range(n):
        if abs(A[i, i]) < TOL:  # checar si A[i,i] es prácticamente 0
            raise Exception("No existe una solución única al sistema")

        s = (
            A[i, :i] @ x[:i]
        )  # \sum_{j=1}^{i-1} a_{ij}x_j corresponde al producto punto entre A[i,:i] y x[:i]
        # notar que cuando i == 0, el producto punto (de arrays vacíos) da 0

        x[i] = (b[i] - s) / A[i, i]

    return x


def solveByGaussianElimination(
    A: np.array, b: np.array, TOL: float = 10 ** (-8)
) -> np.array:
    """Resuelve Ax=b mediante eliminación Gaussiana con pivoteo parcial y sustitución regresiva. Regresa la solución x si existe, y en otro caso levanta una excepción.

    :param A: Matriz de nxn
    :type A: np.array
    :param b: Vector independiente de tamaño n
    :type b: np.array
    :param TOL: Toleracia, i.e., si |x|<TOL, entonces x se considera como 0
    :type TOL: float
    :return: Vector solución x de tamaño n (si existe)
    :rtype: np.array
    """
    # copiamos A, b y casteamos a flotante, para no modificar la matriz y vector originales mientras hacemos eliminación Gaussiana,
    # y también para no hacer división entera si A tiene puros enteros (dtype 'int64')
    A_c = A.astype("float64")
    b_c = b.astype("float64")

    n: int = b.size

    for i in range(n):
        i_max = (
            np.abs(A_c[i:, i]).argmax() + i
        )  # encontramos la fila del elemento de mayor magnitud en la columna i después de o en la fila i
        if i_max != i:
            A_c[[i, i_max]] = A_c[[i_max, i]]  # swap de fila i y fila i_max
            b_c[i], b_c[i_max] = b_c[i_max], b_c[i]

        if abs(A_c[i, i]) < TOL:  # checar si A_c[i,i] es prácticamente 0
            raise Exception(
                "No existe una solución única al sistema"
            )  # no existe porque si el elemento de mayor magnitud es cercano a 0, los demás también

        for k in range(i + 1, n):
            # hacer 0's debajo del pivote A_c[i,i]
            m_ki = A_c[k, i] / A_c[i, i]
            A_c[k, i:] = A_c[k, i:] - m_ki * A_c[i, i:]
            b_c[k] = b_c[k] - m_ki * b_c[i]

    # aqui ya tendriamos un sistema Ux = c, donde U es triangular superior
    return solveUpperTriangular(A_c, b_c, TOL)


def FactorizacionLU(A: np.array, TOL: float = 1e-8) -> Tuple[np.array, np.array]:
    """Factorización LU sin pivoteo parcial de una matriz A

    :param A: Matriz que se quiere factorizar
    :type A: np.array
    :param TOL: Tolerancia para saber si un pivote es cercano a 0, defaults to 1e-8
    :type TOL: float, optional
    :raises Exception: Se levanta una excepción si se encuentra un pivote muy cercano a 0 (de acuerdo a la tolerancia)
    :return: Regresa las matrices L y U
    :rtype: Tuple[np.array, np.array]
    """
    n = A.shape[0]
    L = np.identity(n)
    U = A.astype("float64")

    for i in range(n):
        if abs(U[i, i]) < TOL:  # checar si U[i,i] es prácticamente 0
            raise Exception(
                "Se encontró un pivote muy cercano a 0 de acuerdo a la tolerancia especificada."
            )

        for k in range(i + 1, n):
            # hacer 0's debajo del pivote U[i,i]
            m_ki = U[k, i] / U[i, i]
            L[k, i] = m_ki
            U[k, i:] = U[k, i:] - m_ki * U[i, i:]

    return L, U


def FactorizacionLU_pivoteo(
    A: np.array, TOL: float = 1e-8
) -> Tuple[np.array, np.array, np.array]:
    """Factorización LU con pivoteo de una matriz A

    :param A: Matriz que se quiere factorizar
    :type A: np.array
    :param TOL: Tolerancia para saber si un pivote es cercano a 0, defaults to 1e-8
    :type TOL: float, optional
    :raises Exception: Se levanta una excepción si se encuentra un pivote muy cercano a 0 (de acuerdo a la tolerancia)
    :return: Regresa las matrices L y U, y un vector p que determina la matriz de permutación (1 en entradas (i, p(i)))
    :rtype: Tuple[np.array, np.array, np.array]
    """
    n = A.shape[0]
    L = np.identity(n)
    U = A.astype("float64")
    p = np.array(range(n))

    for i in range(n):
        i_max = (
            np.abs(U[i:, i]).argmax() + i
        )  # encontramos la fila del elemento de mayor magnitud en la columna i después de o en la fila i
        if i_max != i:
            U[[i, i_max]] = U[[i_max, i]]  # swap de fila i y fila i_max
            p[i], p[i_max] = p[i_max], p[i]
            L[[i, i_max], :i] = L[
                [i_max, i], :i
            ]  # swap de fila i y fila i_max de L de la columna 0 hasta la columna i-1

        if abs(U[i, i]) < TOL:  # checar si U[i,i] es prácticamente 0
            raise Exception(
                "Se encontró un pivote muy cercano a 0 de acuerdo a la tolerancia especificada."
            )

        for k in range(i + 1, n):
            # hacer 0's debajo del pivote U[i,i]
            m_ki = U[k, i] / U[i, i]
            L[k, i] = m_ki
            U[k, i:] = U[k, i:] - m_ki * U[i, i:]

    return L, U, p


def determinante(A: np.array) -> float:
    """Determinante de una matriz A

    :param A: Matriz a la que se calculará determinante. Se utiliza factorización LU sin pivoteo parcial. Si A no tiene dicha factorización, se levantará una excepción.
    :type A: np.array
    :return: Determinante de A
    :rtype: float
    """
    L, U, p = FactorizacionLU_pivoteo(A)

    # signo de permutación, igual -1 elevado al número de inversiones (#{(i,j) | i<j y p(i) > j})
    det_P = 1
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                det_P *= -1

    return det_P * np.diag(U).prod()


def SOR(
    A: np.array,
    b: np.array,
    x_0: np.array,
    w: float,
    MAX_ITER: int = 1000,
    TOL: float = 10 ** (-5),
) -> np.array:
    """Método para resolver el sistema Ax=b utilizando el método de sobre-relajación sucesiva (SOR) a partir del método de Gauss-Jacobi

    :param A: Matriz de nxn
    :type A: np.array
    :param b: Vector independiente de dimensión n
    :type b: np.array
    :param x_0: Vector inicial
    :type x_0: np.array
    :param w: Factor de relajación, 0<w<2
    :type w: float
    :param MAX_ITER: Número máximo de iteraciones para el método, defaults MAX_ITER = 1000
    :type MAX_ITER: int, optional
    :param TOL: Toleracia para la condición de paro, i.e., el algoritmo se considera que converge si ||x^(k+1)-x^(k)||<TOL, defaults TOL = 10**(-5)
    :type TOL: float, optional
    :return: Vector solución x de tamaño n si el algoritmo converge.
    :rtype: np.array
    """

    def divide_by(
        x: float,
    ) -> (
        float
    ):  # función auxiliar para calcular 1/x, levantando una excepción si x es cercano a 0
        if abs(x) < TOL:  # si x es practicamente 0, se levanta una excepción
            raise Exception("Error, existe un elemento en la diagonal de A que es 0")

        return 1 / x

    D_inv = np.diag(
        [divide_by(x) for x in np.diag(A)]
    )  # inversa de matriz diagonal D, D con la diagonal de A

    L = np.tril(
        A, k=-1
    )  # matriz triangular inferior con elementos debajo de la diagonal
    U = np.triu(
        A, k=1
    )  # matriz triangular superior con elementos arriba de la diagonal
    x_k = x_0

    for iter in range(MAX_ITER):
        x_k1 = (1 - w) * x_k + w * D_inv @ (-L @ x_k - U @ x_k + b)

        if np.linalg.norm(x_k1 - x_k) < TOL:  # checar si se cumple la condición de paro
            print(f"Algoritmo converge en {iter+1} iteraciones")
            return x_k1

        x_k = x_k1

    raise Exception("No converge el algoritmo SOR")


def MetodoPotencia(
    A: np.array, x0: np.array, TOL: float = 10 ** (-5), MAX_ITER: int = 1000
) -> Tuple[float, np.array]:
    xk = x0 / np.linalg.norm(x0)

    lambda_k = np.inf

    for iter in range(MAX_ITER):
        yk = A @ xk

        j = np.argmax(abs(xk))
        lambda_k = yk[j] / xk[j]

        xk = yk

        v1 = xk / (
            lambda_k ** (iter + 1)
        )  # se calcula eigenvector como x_{k}/lambda^{k}
        v1 = v1 / np.linalg.norm(v1)  # normalizamos

        if np.linalg.norm(A @ v1 - lambda_k * v1) < TOL:  # checar condición de paro
            print(f"Converge en {iter+1} iteraciones")
            return lambda_k, v1

    raise Exception("No converge")


def MetodoPotenciaRayleigh(
    A: np.array,
    x0: np.array,
    TOL: float = 10 ** (-5),
    MAX_ITER: int = 1000,
    iterations: Optional[List[float]] = None,
) -> Tuple[float, np.array]:
    """Método de la Potencia utilizando el coeficiente de Rayleigh (x^TAx, cuando ||x||=1) para calcular el eigenvalor de mayor magnitud y su eigenvector
    asociado de una matriz A de nxn.

    :param A: Matriz de nxn a la cual se le quiere calcular el eigenvalor de mayor magnitud y su eigenvector asociado.
    :type A: np.array
    :param x0: Vector inicial de tamaño n para el algoritmo iterativo
    :type x0: np.array
    :param TOL: Tolerancia para saber cuando convergió el algoritmo, i.e., cuando ||Ax-lambda*x|| < TOL donde x es el eigenvector y lambda el eigenvalor, defaults to 10**(-5)
    :type TOL: float, optional
    :param MAX_ITER: Número máximo de iteraciones del algoritmo, defaults to 1000
    :type MAX_ITER: int, optional
    :param iterations: Lista vacía opcional para guardar "inplace" las iteraciones de las lambdas, defaults to None
    :type iterations: Optional[List[float]], optional
    :raises Exception: Si no converge el algoritmo, i.e., se pasa del número de iteraciones máximo, se levanta una excepción
    :return: Tupla de eigenvalor de mayor magnitud y su eigenvector asociado normalizado
    :rtype: Tuple[float, np.array]
    """
    A = A.astype("float64")
    xk = x0 / np.linalg.norm(x0)

    lambda_k = np.inf

    for iter in range(MAX_ITER):
        yk = A @ xk
        xk = yk / np.linalg.norm(yk)

        lambda_k = xk.T @ A @ xk  # cociente de Rayleigh

        if iterations is not None:
            iterations.append(lambda_k)

        if np.linalg.norm(A @ xk - lambda_k * xk) < TOL:
            # print(f"Converge en {iter+1} iteraciones")
            return lambda_k, xk

    raise Exception("No converge")


def MetodoPotenciaInversaDesplazamiento(
    A: np.array,
    x0: np.array,
    p: float = 0,
    TOL: float = 10 ** (-5),
    MAX_ITER: int = 1000,
    iterations: Optional[List[float]] = None,
) -> Tuple[float, np.array]:
    """Método de la Potencia Inversa con desplazamiento para calcular el eigenvalor de una matriz A más cercano a un valor dado p. p por default es 0,
    por lo que si no se especifica el algoritmo regresa el eigenvalor de A más pequeño en magnitud (Método de la Potencia Inversa normal).

    :param A:  Matriz de nxn a la cual se le quiere calcular el eigenvalor más cercano a p (si p no se especifica, el de menor magnitud) y su eigenvector asociado.
    :type A: np.array
    :param x0: Vector inicial de tamaño n para el algoritmo iterativo
    :type x0: np.array
    :param p: Target para encontrat el eigenvalor más cercano a p, si p no es dado se obtiene el eigenvalor más pequeño en magnitud, defaults to 0
    :type p: float, optional
    :param TOL: Tolerancia para saber cuando convergió el algoritmo, i.e., cuando ||Ax-lambda*x|| < TOL donde x es el eigenvector y lambda el eigenvalor, defaults to 10**(-5)
    :type TOL: float, optional
    :param MAX_ITER: Número máximo de iteraciones del algoritmo, defaults to 1000
    :type MAX_ITER: int, optional
    :param iterations: Lista vacía opcional para guardar "inplace" las iteraciones de las lambdas, defaults to None
    :type iterations: Optional[List[float]], optional
    :raises Exception: Si no converge el algoritmo, i.e., se pasa del número de iteraciones máximo, se levanta una excepción
    :return: Tupla de eigenvalor más cercano a p (o eigenvalor más pequeño en magnitud si p no es dado) y su eigenvector asociado normalizado
    :rtype: Tuple[float, np.array]
    """
    A = A.astype("float64")
    xk = x0 / np.linalg.norm(x0)
    n = x0.size
    A_shift = A - p * np.identity(n)

    lambda_k = np.inf

    for iter in range(MAX_ITER):
        yk = solveByGaussianElimination(A_shift, xk)  # resolvemos A@yk = xk
        xk = yk / np.linalg.norm(yk)

        lambda_k = xk.T @ A_shift @ xk  # cociente de Rayleigh

        if iterations is not None:
            iterations.append(lambda_k + p)

        if np.linalg.norm(A_shift @ xk - lambda_k * xk) < TOL:
            # print(f"Converge en {iter+1} iteraciones")
            return lambda_k + p, xk

    raise Exception("No converge")


def HouseholderMatrix(v: np.array) -> np.array:
    return np.identity(v.size) - (2 / (np.linalg.norm(v) ** 2)) * np.outer(v, v.T)


def canonical_vector(size: int, index: int) -> np.array:
    e = np.zeros(size)
    e[index] = 1.0
    return e


def MetodoPotenciaConHouseholder(
    A: np.array, TOL: float = 10 ** (-5), MAX_ITER: int = 1000
) -> np.array:
    """Método de la potencia mediante deflación utilizando matrices de Householder. Regresa el vector de valores propios de la matriz A dada.

    :param A: _description_
    :type A: np.array
    :param TOL: _description_, defaults to 10**(-5)
    :type TOL: float, optional
    :param MAX_ITER: _description_, defaults to 1000
    :type MAX_ITER: int, optional
    :return: _description_
    :rtype: np.array
    """
    n = A.shape[0]

    valores_propios = np.zeros(n)
    B = A.copy()

    for i in range(n):
        if i == n - 1:
            # ultima iteración, B sólo tiene el último valor propio
            valores_propios[i] = B[0, 0]
            continue

        valores_propios[i], vk = MetodoPotenciaRayleigh(
            B, np.random.random(B.shape[0]), TOL=TOL, MAX_ITER=MAX_ITER
        )
        e1 = canonical_vector(vk.size, 0)
        v = vk - e1
        H = HouseholderMatrix(v)
        B1 = H @ B @ H
        B = B1[1:, 1:]

    return valores_propios


####################### Solución a ecuaciones no lineales


def MetodoBiseccion(
    f: Callable[[float], float], a: float, b: float, epsilon: float, TOL: float = 1e-8
) -> float:
    """Método de la bisección para localización de raíces de una función f continua en un intervalo [a, b] tal que f(a)f(b) < 0. Si se necesita encontrar un intervalo [d, c] que contenga a la raíz
    de tamaño menor que epsilon, entonces se regresa el punto medio del intervalo. Si la función encuentra la raíz m en algún momento (|f(m)| < TOL), entonces se regresa m.

    :param f: Función continua f a la que se le quiere localizar una raíz.
    :type f: Callable[[float], float]
    :param a: Extremo izquierdo del intervalo en donde se buscará la raíz.
    :type a: float
    :param b: Extremo derecho del intervalo en donde se buscará la raíz.
    :type b: float
    :param epsilon: Tamaño del intervalo que contendrá a la raíz.
    :type epsilon: float
    :param TOL: Tolerancia para saber si se ha encontrado una raíz m, i.e., si |f(m)| < TOL (un número pequeño), defaults to 1e-8
    :type TOL: float, optional
    :raises Exception: Se levanta una excepción si no se cumple que f(a)f(b) < 0.
    :return: Punto medio del intervalo de tamaño epsilon que contendrá a la raíz (puede coincidir con la raíz).
    :rtype: float
    """

    if f(a) * f(b) > 0:
        raise Exception("Se necesita que f(a)f(b) < 0.")

    while abs(b - a) > epsilon:
        m = (a + b) / 2

        if abs(f(m)) < TOL:
            return m

        if f(m) * f(a) > 0:
            a = m
        else:
            b = m

    return (a + b) / 2


def MetodoNewton(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    epsilon: float,
    MAX_ITER: int = 10**5,
    iterates: Optional[List[float]] = None,
) -> float:
    """Método de Newton para encontrar una raíz de una función diferenciable f. Es un método iterativo cuyo valor inicial x0 debe ser cercano a la raíz.

    :param f: Función diferenciable a la cual se le quiere encontrar una raíz.
    :type f: Callable[[float], float]
    :param df: Derivada de f.
    :type df: Callable[[float], float]
    :param x0: Valor inicial del algoritmo iterativo.
    :type x0: float
    :param epsilon: Tolerancia para condición de paro, i.e., se detienen las iteraciones cuando |x_{k+1} - x_k| < epsilon
    :type epsilon: float
    :param MAX_ITER: Número máximo de iteraciones en el algoritmo
    :type MAX_ITER: int
    :param iterates: Lista opcional para guardar los iterados, defaults to None
    :type iterates: Optional[List[float]], optional
    :raises Exception: Se levanta una excepción si se llega al número máximo de iteraciones sin haber llegado a la condición de paro
    :return: Raíz de la función f
    :rtype: float
    """

    xk = x0

    if iterates is not None:
        iterates.append(xk)

    for _ in range(MAX_ITER):
        xk_1 = xk - f(xk) / df(xk)

        if iterates is not None:
            iterates.append(xk_1)

        if abs(xk_1 - xk) < epsilon:
            return xk_1

        xk = xk_1

    raise Exception(
        "El método de Newton no convergió (máximo número de iteraciones alcanzado)"
    )


def MetodoSecante(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    TOL: float,
    rel_error: bool = False,
    MAX_ITER: int = 10000,
    iterates: Optional[List[float]] = None,
) -> float:
    """Método de la secante para encontrar una raíz de una función f

    :param f: Función a la que se le quiere encontrar la raíz
    :type f: Callable[[float], float]
    :param x0: primer punto inicial
    :type x0: float
    :param x1: segundo punto inicial
    :type x1: float
    :param TOL: Condición de paro, i.e., cuando |f(x_k)| <= TOL
    :type TOL: float
    :param MAX_ITER: Máximo número de iteraciones, defaults to 10000
    :type MAX_ITER: int, optional
    :param iterates: Lista opcional para guardar los iterados, defaults to None
    :type iterates: Optional[List[float]], optional
    :raises Exception: Se levanta una excepción si se llega al número máximo de iteraciones.
    :return: Aproximación de una raíz de f
    :rtype: float
    """
    xk = x0  # x_k
    xk_1 = x1  # x_{k+1}

    if iterates is not None:
        iterates.append(xk)
        iterates.append(xk_1)

    for _ in range(MAX_ITER):
        aux = xk_1
        xk_1 = xk_1 - f(xk_1) * (xk_1 - xk) / (f(xk_1) - f(xk))
        xk = aux

        if iterates is not None:
            iterates.append(xk_1)

        if rel_error:
            if abs(xk_1 - xk) / abs(xk_1) <= TOL:
                return xk_1
        else:
            if abs(f(xk_1)) <= TOL:
                return xk_1

    print(abs(xk_1 - xk) / abs(xk_1))
    raise Exception(
        "El método de la secante no convergió (máximo número de iteraciones alcanzado)"
    )


def AceleracionAitken(x_n: List[float]) -> List[float]:
    """Aplica el método de Aitken a una sucesión dada para acelerar su convergencia.

    :param x_n: Sucesión de interés, i.e., una lista de números.
    :type x_n: List[float]
    :raises Exception: Se levanta una excepción si la lista tiene menos de 3 elementos. El método necesita al menos 3.
    :return: Sucesión de Aitken
    :rtype: List[float]
    """
    n = len(x_n)

    if n < 3:
        raise Exception(
            "El método de aceleración de Aitken necesita al menos 3 elementos de la sucesión."
        )

    aitken = [
        x_n[i] - (x_n[i + 1] - x_n[i]) ** 2 / (x_n[i + 2] - 2 * x_n[i + 1] + x_n[i])
        for i in range(n - 2)
    ]

    return aitken


def MetodoNewtonMultivariado(
    F: Callable[[np.array], np.array],
    J_F: Callable[[np.array], np.array],
    x0: np.array,
    epsilon: float,
    MAX_ITER: int = 10**5,
    iterates: Optional[List[np.array]] = None,
) -> np.array:
    """Método de Newton para encontrar una raíz de una función diferenciable F de R^n a R^n, i.e., resolver F(X) = 0. Es un método iterativo cuyo vector inicial x0 debe ser cercano a la raíz.

    :param F: Función de R^n a R^n diferenciable a la cual se le quiere encontrar una raíz (F(X) = 0).
    :type F: Callable[[np.array], np.array]
    :param J_F: Jacobiano de f. Función que recibe un vector x n-dimensional y regresa la matriz Jacobiana de F evaluada en x (de nxn).
    :type J_F: Callable[[np.array], np.array]
    :param x0: Vector n-dimensional inicial del algoritmo iterativo.
    :type x0: np.array
    :param epsilon: Tolerancia para condición de paro, i.e., se detienen las iteraciones cuando ||x_{k+1} - x_k|| < epsilon
    :type epsilon: float
    :param MAX_ITER: Número máximo de iteraciones en el algoritmo
    :type MAX_ITER: int
    :param iterates: Lista opcional para guardar los iterados, defaults to None
    :type iterates: Optional[List[np.array]], optional
    :raises Exception: Se levanta una excepción si se llega al número máximo de iteraciones sin haber llegado a la condición de paro
    :return: Vector raíz X tal que F(X) = 0.
    :rtype: np.array
    """

    xk = x0

    if iterates is not None:
        iterates.append(xk)

    for _ in range(MAX_ITER):
        # x_{k+1} = x_k - J_F^{-1}(x_k)F(x_k)
        # Para evitar calcular J_F^{-1}(x_k), resolvemos el sistema  J_F(x_k)z = -F(x_k) para z y hacemos x_{k+1} = z + x_k
        z = solveByGaussianElimination(J_F(xk), -F(xk))
        xk_1 = z + xk

        if iterates is not None:
            iterates.append(xk_1)

        if np.linalg.norm(xk_1 - xk, ord=np.inf) < epsilon:
            return xk_1

        xk = xk_1

    raise Exception(
        "El método de Newton multivariado no convergió (máximo número de iteraciones alcanzado)"
    )


####################### Interpolación


class CubicSpline:
    """Clase para calcular y evaluar un spline cúbico. Sólo soporta spline cúbico natural por ahora."""

    x: np.array  # vector de puntos x_i (ordenado)
    y: np.array  # vector de puntos y_i

    # vectores que determinan al spline en el intervalo [x_k, x_{k+1}]:
    # S(x) = a_k + b_k(x-x_k) + c_k(x-x_k)^2 + d_k(x-x_k)^3
    a: np.array
    b: np.array
    c: np.array
    d: np.array

    def __init__(self, x: np.array, y: np.array):
        """Inicialización de la clase que recibe los puntos (x_i, y_i) que se van a interpolar mediante el spline cúbico. El vector x se supone ordenado.

        :param x: Vector de puntos x_i
        :type x: np.array
        :param y: Vector de valores y_i
        :type y: np.array
        """
        self.x = x
        self.y = y

    def fit(self) -> Tuple[np.array, np.array, np.array, np.array]:
        """Método de la clase para realizar la interpolación de los puntos. Regresa los vectores a, b, c, d tales que el spline S(x) en el intervalo [x_k, x_{k+1}] está dado por
        S(x) = a_k + b_k(x-x_k) + c_k(x-x_k)^2 + d_k(x-x_k)^3

        :return: Regresa la tupla de vectores a, b, c, d con los coeficientes que determinan el spline en los intervalos [x_k, x_{k+1}].
        :rtype: Tuple[np.array, np.array, np.array, np.array]
        """
        h = self.x[1:] - self.x[:-1]  # vector de diferencias h_k = x_{k+1} - x_k
        beta = (
            self.y[1:] - self.y[:-1]
        ) / h  # vector de valores beta_k = (y_{k+1} - y_k)/h_k
        b = 6 * (beta[1:] - beta[:-1])  # vector de valores 6(beta_k - beta_{k-1})
        D = np.diag(
            2 * (h[1:] + h[:-1])
        )  # matriz diagonal con valores (h_{k-1} + h_k)/3
        D_up = np.diag(
            h[1:-1], k=1
        )  # matriz con valores h_k en la diagonal encima de la diagonal principal
        D_down = np.diag(
            h[1:-1], k=-1
        )  # matriz con valores h_{k-1} en la diagonal encima de la diagonal principal
        A = (
            D_down + D + D_up
        )  # matriz tridiagonal que se define para construir el spline

        M = solveByGaussianElimination(
            A, b
        )  # sistema para encontrar los valores M_k = S''(x_k) (segunda derivada del spline evaluada en los puntos x_k) 1<= k <= n-1
        M = np.concatenate(
            ([0], M, [0])
        )  # se agregan condiciones de los splines cúbicos naturales, i.e., M_0 = 0 = M_n

        # vectores a, b, c, d que determinan al spline S(x) en el intervalo [x_k, x_{k+1}]: S_k(x) = a_k + b_k(x-x_k) + c_k(x-x_k)^2 + d_k(x-x_k)^3
        self.a = self.y[:-1]
        self.b = beta - h * (2 * M[:-1] + M[1:]) / 6
        self.c = M[:-1] / 2
        self.d = (M[1:] - M[:-1]) / (6 * h)

        return self.a, self.b, self.c, self.d

    def evaluate(self, t: float) -> float:
        """Método para evaluar el spline en un punto t. Sólo se permite evaluar dentro del intervalo [x_0, x_n]

        :param t: Punto en donde se quiere evaluar el spline
        :type t: float
        :raises Exception: Se lanza una excepción si se quiere evaluar en un punto fuera del intervalo [x_0, x_n]
        :return: Evaluación del spline en el punto dado
        :rtype: float
        """
        if t < self.x[0] or t > self.x[-1]:
            raise Exception(
                "No se puede evaluar el spline en un punto fuera de [x_0, x_n]"
            )

        i = (
            (self.x[:-1] <= t) & (t <= self.x[1:])
        ).argmax()  # buscar i tal que x_i <= x <= x_{i+1}

        return (
            self.a[i]
            + self.b[i] * (t - self.x[i])
            + self.c[i] * (t - self.x[i]) ** 2
            + self.d[i] * (t - self.x[i]) ** 3
        )

    def evaluate_der(self, t: float) -> float:
        """Método para evaluar la derivada del spline en un punto t. Sólo se permite evaluar dentro del intervalo [x_0, x_n]

        :param t: Punto donde se quiere evaluar la derivada del spline
        :type t: float
        :raises Exception: Se lanza una excepción si se quiere evaluar en un punto fuera del intervalo [x_0, x_n]
        :return: Evaluación de la derivada del spline en el punto dado
        :rtype: float
        """
        if t < self.x[0] or t > self.x[-1]:
            raise Exception(
                "No se puede evaluar la derivada del spline en un punto fuera de [x_0, x_n]"
            )

        i = (
            (self.x[:-1] <= t) & (t <= self.x[1:])
        ).argmax()  # buscar i tal que x_i <= x <= x_{i+1}

        return (
            self.b[i]
            + 2 * self.c[i] * (t - self.x[i])
            + 3 * self.d[i] * (t - self.x[i]) ** 2
        )

    def evaluate_snd_der(self, t: float) -> float:
        """Método para evaluar la segunda derivada del spline en un punto t. Sólo se permite evaluar dentro del intervalo [x_0, x_n]

        :param t: Punto donde se quiere evaluar la segunda derivada del spline
        :type t: float
        :raises Exception: Se lanza una excepción si se quiere evaluar en un punto fuera del intervalo [x_0, x_n]
        :return: Evaluación de la segunda derivada del spline en el punto dado
        :rtype: float
        """
        if t < self.x[0] or t > self.x[-1]:
            raise Exception(
                "No se puede evaluar la segunda derivada del spline en un punto fuera de [x_0, x_n]"
            )

        i = (
            (self.x[:-1] <= t) & (t <= self.x[1:])
        ).argmax()  # buscar i tal que x_i <= x <= x_{i+1}

        return 2 * self.c[i] + 6 * self.d[i] * (t - self.x[i])


class PolinomioNewton:
    """Clase para calcular y evaluar el polonomio interpolador de Newton mediante diferencias divididas f[x_0, x_1, ..., x_k]."""

    x: np.array  # vector de puntos x_i (ordenado)
    y: np.array  # vector de puntos y_i

    A: np.array  # Vector con los coeficientes de las diferencias divididas f[x_0, x_1, ..., x_k]

    def __init__(self, x: np.array, y: np.array):
        """Inicialización de la clase que recibe los puntos (x_i, y_i) que se van a interpolar mediante el spline cúbico. El vector x se supone ordenado.

        :param x: Vector de puntos x_i
        :type x: np.array
        :param y: Vector de valores y_i
        :type y: np.array
        """
        self.x = x
        self.y = y

    def fit(self) -> np.array:
        """Método para realizar la interpolación.

        :return: Vector con los coeficientes de las diferencias divididas f[x_0, x_1, ..., x_k]
        :rtype: np.array
        """
        # Obtiene los coeficientes del polinomio de Newton a través de diferencias divididas
        n = self.x.size
        F = np.zeros((n, n))

        F[0] = self.y

        for i in range(1, n):
            for j in range(i, n):
                F[i, j] = (F[i - 1, j] - F[i - 1, j - 1]) / (self.x[j] - self.x[j - i])

        self.A = np.diag(F)

        return self.A

    def evaluate(self, x0: float) -> float:
        """Método para evaluar el polinomio de Newton en un punto dado x0.

        :param x0: Punto en el que se quiere evaluar el polinomio.
        :type x0: float
        :return: Evaluación del polinomio en el punto x0.
        :rtype: float
        """
        n = self.x.size

        prod = 1

        P = self.A[0]

        for i in range(1, n):
            prod = prod * (x0 - self.x[i - 1])
            P = P + self.A[i] * prod

        return P


def PolinomioLagrange(x: np.array, y: np.array, t: float) -> float:
    """Evaluación del polinomio de Lagrange interpolador de puntos (x_i, y_i) en un punto t

    :param x: Vector de valores x_i
    :type x: np.array
    :param y: Vector de valores y_i
    :type y: np.array
    :param t: Punto en donde se evaluará el polinomio de Lagrange
    :type t: float
    :return: Evaluación del polinomio de Lagrange en t
    :rtype: float
    """
    prod = 1
    n = x.size

    P = 0

    for i in range(n):
        prod = 1
        for k in range(n):
            if i != k:
                prod *= (t - x[k]) / (x[i] - x[k])

        P += y[i] * prod

    return P


####################### Integración numérica


def IntegracionTrapecio(x: np.array, y: np.array) -> float:
    """Función para calcular la integral de una función mediante la regla del Trapecio. Esta función está dada por puntos x=[x_0, x_1, ..., x_n], equiespaciados y ordenados, y sus respectivas imágenes
    y=[f(x_0), f(x_1), ..., f(x_n)]. Notar que cuando n = 2, se tiene la regla para integración simple, y para n > 3 se tiene la regla para integral compuesta.

    :param x: Puntos equiespaciados y ordenados x=[x_0, x_1, ..., x_n] en donde se evalúa la función f.
    :type x: np.array
    :param y: Imágenes de los puntos x=[x_0, x_1, ..., x_n] bajo f, i.e., y=[f(x_0), f(x_1), ..., f(x_n)]
    :type y: np.array
    :raises Exception: Se levanta una excepción si se reciben menos de 3 puntos.
    :return: Integral utilizando regla del Trapecio.
    :rtype: float
    """
    if len(x) < 2:
        raise Exception("Se necesitan más de dos puntos.")

    h = x[1] - x[0]
    I = y[0] + y[-1] + 2 * (y[1:-1].sum())
    I = I * h / 2

    return I


def IntegracionSimpson(x: np.array, y: np.array) -> float:
    """Función para calcular la integral de una función mediante la regla de Simpson. Esta función está dada por puntos x=[x_0, x_1, ..., x_n], equiespaciados y ordenados, y sus respectivas imágenes
    y=[f(x_0), f(x_1), ..., f(x_n)]. Notar que cuando n = 2, se tiene la regla para integración simple, y para n > 3 se tiene la regla para integral compuesta.

    :param x: Puntos equiespaciados y ordenados x=[x_0, x_1, ..., x_n] en donde se evalúa la función f.
    :type x: np.array
    :param y: Imágenes de los puntos x=[x_0, x_1, ..., x_n] bajo f, i.e., y=[f(x_0), f(x_1), ..., f(x_n)]
    :type y: np.array
    :raises Exception: Se levanta una excepción si se reciben menos de 3 puntos o si se tiene una cantidad par de puntos. Para la regla de Simpson se necesita una cantidad de puntos impar mayor o igual a 3.
    :return: Integral utilizando regla de Simpson.
    :rtype: float
    """
    if len(x) < 3 or len(x) % 2 == 0:
        raise Exception(
            "Se necesita una cantidad de puntos impar mayor o igual a 3 para este método."
        )

    h = x[1] - x[0]
    i = np.array(range(len(x)))
    I = y[0] + y[-1] + 2 * (y[i % 2 == 0][1:-1].sum()) + 4 * (y[i % 2 == 1].sum())
    I = I * h / 3

    return I


def IntegracionRomberg(
    f: Callable[[float], float], a: float, b: float, l: int, TOL: float = 1e-8
) -> Tuple[float, np.array]:
    R = np.zeros((l + 1, l + 1))

    last_iteration = None

    for j in range(1, l + 1):
        for i in range(1, j + 1):
            if i == 1:
                nj = 2**j
                x_arr = np.linspace(a, b, nj + 1)
                y_arr = np.array([f(t) for t in x_arr])
                R[j, i] = IntegracionTrapecio(x_arr, y_arr)

            else:
                R[j, i] = R[j, i - 1] + (R[j, i - 1] - R[j - 1, i - 1]) / (
                    4 ** (i - 1) - 1
                )

            if i > 1 and j > 1:
                if abs(last_iteration - R[j, i]) <= TOL:
                    return R[i, j], R[1:, 1:]

            last_iteration = R[j, i]

    return R[-1, -1], R[1:, 1:]


def GaussianQuadrature(
    f: Callable[[float], float], a: float, b: float, n: int
) -> float:
    """Función para integrar una función f en [a, b] utilizando el método de cuadratura Gaussiana.

    :param f: Función a integrar (recibe un flotante y regresa un flotante).
    :type f: Callable[[float], float]
    :param a: Extremo izquierdo del intervalo de integración.
    :type a: float
    :param b: Extremo derecho del intervalo de integración.
    :type b: float
    :param n: Número de nodos a usar en la cuadratura Gaussiana.
    :type n: int
    :return: Estimación de la integral
    :rtype: float
    """

    roots, w = np.polynomial.legendre.leggauss(
        n
    )  # función que regresa los nodos y los pesos de la cuadratura Gaussiana (los nodos son las raíces del n-ésimo polinomio de Legendre)

    def change_variable(t: float) -> float:
        return ((b - a) * t + (b + a)) / 2

    return (
        (b - a)
        / 2
        * (np.array([w[i] * f(change_variable(roots[i])) for i in range(n)]).sum())
    )
