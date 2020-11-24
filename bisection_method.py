#Bisection Method: Find the root of polynomial f(x)=x*x*x-x*x-x-1
def polynomial(x):
    fx=x*x*x-x*x-x-1
    return fx;

print("enter a and b")
a=float(input())
b=float(input())
fa = polynomial(a)
fb = polynomial(b)
print(fa,fb)
while(fa*fb>0):
    print("chosen a and b values are not in the correct interval\n")
    print("Enter a and b again\n")
    a = float(input())
    b = float(input())
    fa = polynomial(a)
    fb = polynomial(b)
    print(fa, fb)

while(fa*fb<=0):
        c=(a+b)/2.0
        fc=polynomial(c)
        if (fc <=0.00001):
            print("The root of polynomial f(x)=x*x*x-x*x-x-1 is", c)
            break;
        else:
            if (fc <= 0):
                a = c
            else:
                b = c
        fa = polynomial(a)
        fb = polynomial(b)
        if(fa*fb>0):
            break;










