#lower triangle
def lower_triangular(n):
    for i in range(n):
        for j in range(i + 1):
            print("*", end=" ")
        print()

n = 5 
lower_triangular(n)

#upper triangle
def upper_triangular(n):
    for i in range(n):
        for j in range(n):
            if j < i:
                print(" ", end=" ")
            else:
                print("*", end=" ")
        print()

n = 5
upper_triangular(n)

#pyramid pattern
def pyramid(n):
    for i in range(n):
        print(" " * (n - i - 1) + "* " * (i + 1))

n = 5  # Number of rows
pyramid(n)