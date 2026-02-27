import turtle


def koch(t, leng, depth):
    if depth == 0:
        t.forward(leng)
    else:
        leng /= 3
        koch(t, leng, depth - 1)
        t.left(60)
        koch(t, leng, depth - 1)
        t.right(120)
        koch(t, leng, depth - 1)
        t.left(60)
        koch(t, leng, depth - 1)

scr = turtle.Screen()
scr.bgcolor('white')
t = turtle.Turtle()
t.speed(0)
t.pensize(2)

t.penup()
t.goto(-300, -50)
t.pendown()

for i in range(3):
    koch(t, 200, 3)
    t.left(120)

turtle.done()