# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 12:34:44 2019

@author: Briac
"""

# Procédure
def table7():
    n = 1
    while n <= 10:
        print(n, "x 7 =", n * 7)
        n += 1

def table(base, start=1, length=10):
    n = start
    while n < start + length:
        print(n, "x", base, "=", n * base)
        n += 1

def multiply(a, b):
    return a * b

def abs(x):
    if x < 0:
        return -x
    else:
        return x

def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def table(base, start=1, length=10, symbol="*", op=multiply):
    n = start
    while n < start + length:
        print(n, symbol, base, "=", op(n, base))
        n += 1

table(4, length=2)
table(4, length=5, symbol="+", op=add)


#########" Fonction récursive
def sum(n):
    result = 0
    while n > 0:
        result += n
        n -= 1
    return result

def sum(n):
    if n == 1:
        return 1
    return sum(n - 1) + n

def pow(a, n):   #a puissance n
    if n == 1:
        return a
    if n % 2 == 0:
        return pow(a * a, n / 2)
    return a * pow(a * a, (n - 1) / 2)

##### Utilisation d'un module
import turtle
turtle.forward(90)
turtle.done()

