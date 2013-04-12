import matplotlib.pyplot as pylab
import sys

hessian = []
control = []
imax = 100
f = open(sys.argv[1], "r")
for line in f:
    if line.strip() != "":
        s = line.strip().split()
        if s[0] == "Control=":
            control.append(float(s[1]))
        elif s[0] == "Accuracy=":
            hessian.append(float(s[1]))
print hessian
imax = min(imax, len(control))

pylab.plot(hessian[:imax])
pylab.plot(control[:imax])
pylab.show()
