import numpy as np
import matplotlib.pyplot as plt
import timeit
import scipy.special as sc

#exercise 1

#a

#Golden section search becaus N(x) is one dimensional
def Golden(func,a,b,c, maxiter, accur):
    """Golden section search for finding the minimum of func with the initial bracket [a,b,c]. It iterates either over maxiter iterations or stops when the bracket becomes smaller than accur."""
    #golden ratio and weight
    phi = (1+np.sqrt(5))*0.5
    w = 2-phi
    
    
    for k in range(0,maxiter):
#see which bracket is smaller
        if np.abs(c-b) > np.abs(a-b):
            x = c
        else:
            x = a
#get the new guess for the minimum
        d = b + (x-b)*w
        #print("a,b,c,d", a,b,c,d)
#if d is a worse guess than b, keep b as best guess & make bracket smaller
        if func(b) < func(d):
            if np.abs(c-a) < accur:
                print("accur return b", np.abs(c-a))
                return b
            if np.abs(c-b) > np.abs(a-b):
                c = d
            else:
                a = d
#otherwise, shift bracket to center d
        if func(d) < func(b):
            if np.abs(c-a) < accur:
                print("accur return d", np.abs(c-a))
                return d
        
            if np.abs(c-b) < np.abs(a-b):
                c,b = b,d
            else:
                a,b = b,d

    return b



#getting the function from handin 2 and add Nsat
#N(x)dx is 4*pi*n(x)*x^2 dx

def N(x,A=256/(5*np.pi**(3/2)),a=2.4,b=0.25,c=1.6, Nsat=100):
	"""Returns N(x)"""
	return 4*np.pi* Nsat * A*(x**(a-1)/(b**(a-3)))*np.exp(-(x/b)**c)

#make a function to find the minimum of

def Nmin(x,A=256/(5*np.pi**(3/2)),a=2.4,b=0.25,c=1.6, Nsat=100):
	"""Returns N(x)"""
	return -1*(4*np.pi* Nsat * A*(x**(a-1)/(b**(a-3)))*np.exp(-(x/b)**c))

#we want to find the maximum so we need to find the minimum of -N(x)
#plotting the function shows that the maximum is between 0 and 1 so do initial bracket 0,0.5,5
maxim = Golden(Nmin,0,0.5,5,maxiter=100,accur=10**(-15))

print(maxim)
np.savetxt("NUR3Q1maxim.txt", [maxim])

xes = np.linspace(0,5,1000)

plt.plot(xes,N(xes))
#plt.yscale('log')
plt.scatter(maxim, N(maxim), label='maximum found by Golden')
plt.ylabel("N(x)")
plt.xlabel("x")
plt.legend()
plt.savefig("NUR3Q1plot1.pdf")
plt.close()




#1b

#the code given
def readfile(filename):
    f = open(filename, 'r')
    data = f.readlines()[3:] #Skip first 3 lines 
    nhalo = int(data[0]) #number of halos
    radius = []
    
    for line in data[1:]:
        if line[:-1]!='#':
            radius.append(float(line.split()[0]))
    
    radius = np.array(radius, dtype=float)    
    f.close()
    return radius, nhalo #Return the virial radius for all the satellites in the file, and the number of halos


#the function goes from 0,5 and N(x) with a real/linear (non-log) axis already looks poissonian so I will take linear bins (also I don't want log because we start at 0)
#we have more than 20 000 000 haloes in the files, 1/20000000 ~ 5 * 10**(-8)
#so, we want to take the bins until N(x) becomes ~10**(-8), looking at the ylog scale of N(x)
#and at when the histogram plots of the data cease to have counts in bins that is around x~2
#the maximal radius for the first data set might expand a bit beyond 2 but I dont want too many zeros in the other data sets
#for the number of bins I inspect the histograms to see when they approximately look like a poisson
#for 25 bins between 0 and 2 I find a good balance between enough bins so that the ones with many halos look good, but the last one doesn't look too bad either.

bins = np.linspace(0,2,25)

#get the middle of each bin
xs = bins[1:] - (bins[3] - bins[2])*0.5

#get all functions from working classes
def rowswap(M, i, j):
	"""swap row i and j of matrix M"""
	B = np.copy(M).astype("float64")

	row = B[i, :].copy()
	B[i,:] = B[j, :]
	B[j, :] = row
	return B

def rowswapvec(M, i, j):
	"""swap indices i and j of a vector M"""
	B = np.copy(M).astype("float64")

	row = B[i].copy()
	B[i] = B[j]
	B[j] = row
	return B
    

def rowadd(M, i, j, scale):
	"""add row i to row j of matrix M scale times"""
	B = np.copy(M).astype("float64")

	row = B[j, :].copy()
	B[j, :] = row+ scale*B[i,:]
	return B


def rowscale(M, i, scale):
	"""scale row i of matrix M scale times"""
	B = np.copy(M).astype("float64")

	row = B[i, :].copy()
	B[i,:] = row*scale
	return B

def LUdecomp(M, x):
	"""LU decomposition given a matrix M and x, returns x_end satisfying M*x_end = x"""
	#make copies so we dont mess with any global variables accidentally
	M = M.copy()
	x = x.copy()
	x_new = x.copy()
	rows = M.shape[0]
	columns = M.shape[1]

	pivot_ind = np.zeros(columns).astype("int")
	index = np.linspace(0,rows-1, rows).astype("int")
	parit = 1

	#check if the matrix is singular
	for j in range(columns):
		if np.all(M[:,j] == 0):
			print("Matrix is singular")
			return M, np.zeros(rows), index

    	#get pivots by choosing the maximum and put the pivots on the diagonal
	for i in range(columns):
		maxim = np.abs(M[i,i])
		piv = i
		for k in range(i+1, rows):
			xij = np.abs(M[k,i])
			if xij > maxim:
				maxim = xij
				piv = k
                
		pivot_ind[i] = piv
		if piv != i:
			M = rowswap(M,i,piv)
			#keep track of what we swap
			index = rowswapvec(index,i,piv).astype("int")
			parit *= -1
            
		xii = M[i,i]
		#get the LU matrix
		for n in range(i+1, rows):
			LUni = M[n,i] / xii
			M[n,i] = LUni
			for m in range(i+1, rows):
				M[n,m] -= LUni * M[i,m]
        
        
	#get the solution
	x_end = np.zeros(rows)
	y_end = np.zeros(rows)

	#forward substitution
	for p in range(rows):
		ay = 0
		for l in range(0,p):
			ay += M[p,l]*y_end[l]
		y_end[p] = x_new[index[p]] -ay

	#backward substitution
	for w in range(rows):
		back = rows-(w+1)
		ax = 0
		for q in range(back,rows):
			ax += M[back,q]*x_end[q]
		x_end[back] = 1/M[back,back] * (y_end[back] -ax)
        
	
	#return the index vector so we know what we swapped
	return M, x_end, index

#we get the mean and by integrating N(x)dx = 4pi*n(x)*x^2 dx from xi to xi+1 so use Romberg

#simple trapezoid integration
def trapezoid(N, x0, xmax, func, p):
	"""Trapezoid integration with N steps, integrating func from x0 to xmax"""
	#step size is range divided by number of steps
	h = (xmax-x0)/N
	xes = np.linspace(x0,xmax,N)
	#trapezoid integration formula
	integr = h*(func(xes[0],p)*0.5 + np.sum(func(xes[1:N-1],p)) + func(xes[N-1],p)*0.5)

	return integr

#Romberg integration
def Romberg(N, m, x0, xmax, func, p):
	"""Romberg integration with N steps, an order of m, integrating func from x0 to xmax"""
	#step size is range divided by number of steps
	h = (xmax-x0)/N
	#r has the size of the order
	r = np.zeros(m)
	#first estimate is simply the trapezoid integration
	r[0] = trapezoid(N, x0, xmax, func, p)
	
	Np = N
	for i in range(1,m):
		#get different estimates with different step sizes
		r[i] = 0
		diff = h
		h *= 0.5
		x = x0+h

		for k in range(Np):
			r[i] += func(x,p)
			x += diff

		
		r_i = r[i].copy()

		r[i] = 0.5*(r[i-1]+diff*r_i)
		Np *= 2



	Np = 1
	for i in range(1,m):
		#iteratively combine the initial estimates to improve the result
		Np *= 4

		for j in range(0,m-i):
			r[j] = (Np*r[j+1] - r[j])/(Np-1)
	#return final Romberg integration value	    
	return r[0]

#functions for Levenberg-Marquardt

def chi2(x, y, func, p, sig2):
	"""Returns the chi squared given data points x,y, the model func, the model parameters p, and the variance sig2"""
	return np.sum((y - func(x, p))**2 /sig2)

def beta(x, y, func, p, sig2, deriv):
	"""Returns beta for the Levenberg-Marquardt routine given data points x,y, the model func, the model parameters p, the variance sig2, and derivatives of the model wrt p deriv."""
	return np.sum(((y - func(x, p)) /sig2) * deriv(x,p), axis=1)


def alpha(x, p, sig2, deriv, lam):
    """Returns alpha and the for the Levenberg-Marquardt routine given data points x, the model parameters p, the variance sig2, the derivatives of the model wrt p deriv, and a lambda value lam."""
    allderivs = deriv(x,p)
    a = np.zeros((len(allderivs), len(allderivs)))
    acc = np.zeros((len(allderivs), len(allderivs)))

#get the alpha and alpha accent which is modivied by lambda
    
    for i in range(len(allderivs)):
        for j in range(len(allderivs)):
            a[i,j] = np.sum((1/sig2) * allderivs[i,:] * allderivs[j,:])
            
            if i == j:
                acc[i,j] = (1+lam)*np.sum((1/sig2) * allderivs[i,:] * allderivs[j,:])
            else:
		#the off-diagonal elements are the same for alpha and alpha accent
                acc[i,j] = np.sum((1/sig2) * allderivs[i,:] * allderivs[j,:])
    
    return a, acc
        

def LM(xi, yi, func, sig2, p0, derivjes, maxiter, i=0, lambd=10**(-3), w=10):
    """Levenberg-Marquardt routine given data xi,yi, model func, variance sig2, first guess for the parameters p0, derivatives of the model wrt p derivjes, maximum iterations maxiter, iteration number i, starting lambda lambd, and w with which lambda will be multiplied or divided."""
    #get the initial chi squared value
    chi0 = chi2(xi, yi, func, p0, sig2)

    #get the beta values
    beta0  = beta(xi, yi, func, p0, sig2, derivjes)

    #get the alpha and alpha accent
    alpha0, alphacc = alpha(xi, p0, sig2, derivjes, lambd)
    #solve matrix calculation with alpha accent *dp = beta to get dp
    alphanew, dp, ind = LUdecomp(alphacc, beta0)
    
    #get the new p by adding dp
    pnew = p0 + dp
    #print("new p", pnew)
    
    #get the new chisquared with the new p
    chinew = chi2(xi, yi, func, pnew, sig2)
    #print("new chi", chinew)



#if the chi squared is worse, go back to the previous p estimate and make lambda bigger, otherwise, take the new p and make lambda smaller
    if chinew >= chi0:
        lambd *= w
        pnew = p0
    else:
        lambd = lambd/w

#calculate new A given with the new p
        global Aintgr
        A_inv = Romberg(50,6,0,5,n2, pnew)
        Aintgr = 1/A_inv
        
#if chi doesnt really change anymore, return
        if np.abs(chinew - chi0) < 10**(-3):
            return pnew, chinew

#iteration number rises and either return or do another iteration
    i+=1
    
    if i >= maxiter:
        #print("iter return")
        return pnew, chinew
    else:
        return LM(xi, yi, func, var, pnew, derivjes, maxiter, i, lambd, w)


#function for 1c
def QuasiNewt(func, x, maxiter, accur, grad, H_0= np.array([[1,0,0],[0,1,0],[0,0,1]]), i=0, xall=None):
    """Quasi Newton method to find the minimum x of a multidimensional function func using the gradient grad and either iterating maxiter times or until the size of the gradient is smaller than accur. Returns the minimum and all the steps it took to get there."""
    #matrix vector multiplication
    n_init = -np.sum(H_0 * grad(x), axis=1)
    

    #function to find the best lambda
    def minim(lambd):
        return func(x + lambd*n_init)
    
    #lams = np.linspace(-15,15,100)
    #ytjes = np.zeros(100)
    #for m in range(100):
        #ytjes[m] = minim(lams[m])
        
    #print(ytjes)
    
#if the function does not exist or returns NaNs for small numbers, search for a lambda in a small range, if it does exist somewhere, search in a bigger range
    if np.isnan(minim(-1)) or np.isnan(minim(1)):
        #print("1 nan")
        lam_i = Golden(minim, -0.1,0.001,0.1,50, 10**(-20))
    if np.isnan(minim(-0.1)) or np.isnan(minim(0.1)):
        #print("0.1 nan")
        lam_i = Golden(minim, -0.01,0.00001,0.01,50, 10**(-20))
    else:
        #print("take 15")
        lam_i = Golden(minim, -15,0.1,15,50, 10**(-20))
    #print("lam", lam_i)
    
    #plt.plot(lams, ytjes)
    #plt.scatter(lam_i, minim(lam_i))
    #plt.show()
    
    #we dont want to take a stepsize of zero
    if lam_i == 0:
        lam_i = 10**(-3)
    
    delta = lam_i *n_init
    #print("delta", delta)
    
    #we dont want to take a too small step
    while np.abs(delta)[0] < 10**(-10) and np.abs(delta)[1] < 10**(-10):
        delta *= 10
    
    #print("delta new", delta)
    #calculate the new x by adding delta
    x_new = x + delta
    
    #save all the steps we take
    if i == 0:
        xall = x_new.copy()
    else:
        xall = np.vstack((xall, x_new)).copy()
    
    #print("xnew", x_new)
    

    #get the function values of the old and new x
    f0, f_new = func(x), func(x_new)
    
    #if the difference is smaller than our accuracy, return
    if np.abs(f_new - f0)/(0.5*np.abs(f_new - f0)) < accur:
        #print("accur return")
        return x_new, xall
    
    #Calculate D
    D_i = grad(x_new) - grad(x)
    #print("D", D_i)
    
    #if the gradient converges, return
    if np.abs(np.amax(grad(x_new), axis=0)) < accur:
        print("grad conv")
        return x_new, xall
    
    #H times D
    HD =np.sum(H_0*D_i, axis=1)
    
    u = delta/np.sum(delta*D_i) - HD/np.sum(D_i * HD)
    #print("u", u)
    #calculate the new H
    H_i = H_0 + np.outer(delta, delta)/np.sum(delta*D_i) - np.outer(HD, HD)/np.sum(D_i * HD) + np.sum(D_i * HD)*np.outer(u,u)
    #print("H", H_i)
    
    #iteration goes up 1, so either return or do another iteration
    i+= 1
    
    if i >= maxiter:
        print("iter return")
        return x_new, xall
    else:
        #print("x now", x_new)
        return QuasiNewt(func, x_new, maxiter, accur, grad, H_i, i, xall)


#we make a new function to fit with one vector p for all of the parameters we already know Nsat and A depends on abc so we divide those out when fitting
# p[0] is a, p[1] is b, p[2] is c
def Nfit(x,p):
	"""Returns N(x) with parameters p"""
	#print("A, Nsat", Aintgr, Nsats)
	return 4*np.pi*Aintgr*Nsats* (x**(p[0]-1)/(p[1]**(p[0]-3)))*np.exp(-(x/p[1])**p[2])


def derivs(x, p):
	"""Returns the derivatives on Nfit to p = a, b, and c"""
	deriva = 4*np.pi*Aintgr*Nsats *(x**(p[0]-1)/(p[1]**(p[0]-3)))*np.exp(-(x/p[1])**p[2]) * np.log(x/p[1])
	derivb = -4*np.pi*Aintgr*Nsats* (x**(p[0]-1)/(p[1]**(p[0]-3)))*np.exp(-(x/p[1])**p[2]) * (-p[2]*(x/p[1])**(p[2]) + p[0] -3)
	derivc = -4*np.pi*Aintgr*Nsats* (x**(p[0]-1)/(p[1]**(p[0]-3)))*np.exp(-(x/p[1])**p[2]) * np.log(x/p[1]) * (x/p[1])**(p[2])
	return np.array([deriva,derivb,derivc])


#get the function from handin 2 to calculate A with
def n2(x,p):
	"""Returns n(x)*x^2 /(A*Nsat)"""
	return  4*np.pi *(x**(p[0]-1)/(p[1]**(p[0]-3)))*np.exp(-(x/p[1])**p[2])


	


for i in range(1,6):
#load in data
	r1, nhalo1 = readfile('satgals_m1'+str(i)+'.txt')

#To get Nsat (the mean number of satellites in each halo, we need the number of satellites and the number of haloes, which is the length of the r set and nhalo respectively

	Nsats = len(r1)/nhalo1
	print("Good Nsats", Nsats)
#get the number of haloes in each bin and then divide by nhalo and the binwidth to normalize properly

	Ni1, bins = np.histogram(r1, bins=bins)

	Nmean1 = Ni1/(nhalo1*(bins[3]-bins[2]))

	#take as p0 the given a,b,c but then tweaked a little to fit better
	p0 = [2.2, 0.5, 1.6]

#calculate the initial A given p0 

	A_inv = Romberg(50,6,0,5,n2, p0)
	Aintgr = 1/A_inv
	#print("initial A", Aintgr)

#calculate the initial variances
	var = np.zeros(len(bins)-1)
	for j in range(len(bins)-1):
		var[j] = Romberg(50,6,bins[j],bins[j+1],Nfit, p0)
	#print("initial vars", var)

#do a chi squared fit!
	pend, chiend = LM(xs, Nmean1, Nfit, var, p0, derivs, maxiter=100)
	print("i, p, chi", i, pend, chiend)

	#calculate the final A with the fit

	A_inv = Romberg(50,6,0,5,n2, pend)
	Aintgr = 1/A_inv
	
	Nfitchi = Nfit(xs,pend)
#scale so that the sum of Nmean and the fit are the same
	Nfitchi2 = Nfitchi*np.sum(Nmean1)/np.sum(Nfitchi)
	

	plt.bar(xs, Nmean1, bins[3]-bins[2], label="data")
	plt.plot(xs, Nfitchi2, color='crimson', label=r"Nsat = " +str(np.round(Nsats,2))+" [a,b,c] = "+str(np.round(pend,3))+r" $\chi^2$ ="+str(np.round(chiend,3))+" fit.")
	plt.xlabel("radius")
	plt.ylabel("Mean number of satellites per halo")
	plt.yscale('log')
	plt.xscale('log')
	plt.title("Dataset "+str(i))
	plt.legend(loc='lower left')
	plt.savefig("NUR3Q1plotchi2"+str(i)+".pdf")
	plt.close()
	
	#1c
	#for a Poisson likelihood we want to minimize minus ln Likelihood, and to use the QuasiNewton method we also need the derivative of the lnLikelihood

	def LnLike(p, func=Nfit, x=xs, y=Nmean1):
		"""Returns the -ln of the poissonian likelood of a model func with parameters p given the data x,y"""
		#calculate A, doesnt need to be too accurate
		global Aintgr
		A_inv = Romberg(25,4,0,5,n2, p)
		Aintgr = 1/A_inv

		#var = np.zeros(len(bins)-1)
		#for j in range(len(bins)-1):
			#var[j] = Romberg(25,3,bins[j],bins[j+1],Nfit, p)
		return -np.sum(y*np.log(func(x,p)) - func(x,p))

	def dLnLike(p, func=Nfit, x=xs, y=Nmean1, deriv=derivs):
		"""Returns the derivative wrt to parameters p of the ln of the poissonian likelood of a model func, 	the model gradient wrt to p deriv, given the data x,y""" 
		#calculate A
		global Aintgr
		A_inv = Romberg(25,4,0,5,n2, p)
		Aintgr = 1/A_inv

		#var = np.zeros(len(bins)-1)
		#for j in range(len(bins)-1):
			#var[j] = Romberg(25,3,bins[j],bins[j+1],Nfit, p)
		return np.sum((y/func(x,p) -1)* deriv(x,p), axis=1)
	
	p0 = [1.99,0.5,1.6]

#calculate the initial A given p0 

	A_inv = Romberg(50,6,0,5,n2, p0)
	Aintgr = 1/A_inv
	#print("initial A", Aintgr)

#calculate the initial variances
	var = np.zeros(len(bins)-1)
	for j in range(len(bins)-1):
		var[j] = Romberg(50,6,bins[j],bins[j+1],Nfit, p0)
	
	ppois, pall = QuasiNewt(LnLike, p0, maxiter=70, accur=10**(-8), grad=dLnLike)
	
	#calculate the final A with the fit accurately

	A_inv = Romberg(50,6,0,5,n2, ppois)
	Aintgr = 1/A_inv
	print("Pois A", Aintgr)


	Nfitpoiss = Nfit(xs,ppois)
#scale again so the total counts are the same
	Nfitpois = Nfitpoiss*np.sum(Nmean1)/np.sum(Nfitpoiss)
	Lnpois = LnLike(ppois)
	
	plt.bar(xs, Nmean1, bins[3]-bins[2], label="data")
	plt.plot(xs, Nfitpois, color='crimson', label=r"Nsat = " +str(np.round(Nsats,2))+" [a,b,c] = "+str(np.round(ppois,3))+r" $-lnL$ ="+str(np.round(Lnpois,3))+" Poisson fit.")
	plt.xlabel("radius")
	plt.ylabel("Mean number of satellites per halo")
	plt.title("Dataset "+str(i))
	plt.yscale('log')
	plt.xscale('log')
	plt.legend(loc='lower left')
	plt.savefig("NUR3Q1plotpois"+str(i)+".pdf")
	plt.close()
	
	
#1d
#get the expected values of the model by unnormalizing them so we can compare to the observed integer counts

	Nifitchi2 = Nfitchi2*(nhalo1*(bins[3]-bins[2]))
	Nifitpois = Nfitpois*(nhalo1*(bins[3]-bins[2]))
	
#G test
	Gchi2 = 0
	Gpois = 0
	
	for k in range(len(Ni1)):
		if Ni1[k] == 0:
			pass
		else:
			print("Ns",Ni1[k], Nifitchi2[k], Nifitpois[k])
			#print("ln Ns",np.log(Ni1[k]/Nifitchi2[k]), np.log(Ni1[k]/Nifitpois[k]))
			Gchi2 += 2*Ni1[k]*np.log(Ni1[k]/Nifitchi2[k])
			Gpois += 2*Ni1[k]*np.log(Ni1[k]/Nifitpois[k])
			
	print("G's", Gchi2, Gpois)
	print(np.sum(Ni1), np.sum(Nifitchi2), np.sum(Nifitpois))
#calculate Q with the UPPER incomplete gamma function (not the lower)

	def Q(Qx,Qk):
		#print("incgam, gam",sc.gammainc(Qk*0.5,Qx*0.5), sc.gamma(Qk*0.5))
		return 1-(sc.gammaincc(Qk*0.5,Qx*0.5)/sc.gamma(Qk*0.5))

#we have that x=G, and k is the number of data points minus the number of constraints, so data points is number of bins (25), number of constraints is 4 (number of fit params +1)

	ka = len(Ni1)-4
	Qchi2 = Q(Gchi2,ka)
	Qpois = Q(Gpois,ka)
	
	print("Q's", Qchi2, Qpois)
	
	np.savetxt("NUR3Q1chi2"+str(i)+".txt", [Nsats, pend[0], pend[1], pend[2], chiend])
	np.savetxt("NUR3Q1pois"+str(i)+".txt", [ppois[0], ppois[1], ppois[2], Lnpois])
	np.savetxt("NUR3Q1GQ"+str(i)+".txt", [Gchi2, Gpois, Qchi2, Qpois])
	
	
