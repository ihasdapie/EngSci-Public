#!/usr/bin/python
import numpy as np 
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# load data
f = open('cleandata.csv') # your data file name
# your file should be in this format:
# time, amplitude
# I used ", " as a delimiter, if you used spaces or anything
# just change it in the .split() on line 15 to whatever you used
# amp. doesn't have to be in radians but should cross the x-axis

# edit this error to whatever you have...

amp_err = 0.0117453 # 1 degree
time_err = 0.03222 # 2 frames 

b = f.readlines()
f.close()

b = [i.strip().split(', ') for i in b]

# yes, we are comparing floats
# don't really need to be that precise though: data isn't perfect anyways
b = [(float(i[0]), float(i[1])) for i in b]

nb = np.array([(i[0], i[1]) for i in b])
nbt = nb.transpose()

# find positive peaks(pp), negative peaks
pp = find_peaks(nbt[1], prominence=0.3)[0]
pn = find_peaks(nbt[1]*-1, prominence=0.3)[0] # you may have to change the prominence values depending on your setup. .3 should work for most
peaks = np.concatenate((pp, pn), axis=0)


# plot maxima on top of previous graph, just for show
plt.plot(nbt[0], nbt[1], 'g.', label="all data")
plt.plot(nbt[0][peaks], nbt[1][peaks], 'bo',label= "maxima")

plt.xlabel("Time, s")
plt.ylabel("Amplitude, θ")
plt.title("Amplitude vs Time")
plt.legend(loc='upper right')
plt.show()

# extract amplitude & time data








## ----------------------------
## if you have just amplitude vs period data in a file
## comment out whatever code is above of this line

#f = open('cleandata.csv') # your data file name
## formatted as 
##<period> <amp>
##<period> <amp>
##<period> <amp>

#amp_err = 0.0117453 # 1 degree
#time_err = 0.03222 # 2 frames 

#b = f.readlines()
#f.close()

#b = [i.strip().split(', ') for i in b]

## yes, we are comparing floats
## don't really need to be that precise though: data isn't perfect anyways
#b = [(float(i[0]), float(i[1])) for i in b]

#nb = np.array([(i[0], i[1]) for i in b])
#nbt = nb.transpose()


## for maxima
#amp = nbt[1]
#period = nbt[0]
#time_err = np.array([time_err for i in range(len(p_amp))]) # this is ugly
#amp_err = np.array([amp_err for i in range(len(p_period))])


# # plot period vs amp for maxima
# plt.errorbar(amp, period, yerr=  time_err, xerr = amp_err, ecolor='cyan', fmt='.')
# plt.xlabel("Amplitude, θ")
# plt.ylabel("Period, T")
# plt.title("Period vs Amplitude")
# plt.show()


# modify below functions to plot the variables from this part (e.g. changing n_amp to amp), etc


##--------------------------------



#--------------- comment this out if you are using amp vs T time only
# for maxima
p_amp = nbt[1][pp][:len(pp)-1]
p_period = nbt[0][pp]
p_period = np.array([p_period[i]-p_period[i-1] for i in range(1, len(p_period))]) 
p_time_err = np.array([time_err for i in range(len(p_amp))]) # this is ugly
p_amp_err = np.array([amp_err for i in range(len(p_period))])


# for minima
n_amp = nbt[1][pn][:len(pn)-1]
n_period = nbt[0][pn]
n_period = np.array([n_period[i]-n_period[i-1] for i in range(1, len(n_period))])
n_time_err = np.array([time_err for i in range(len(n_amp))]) # this is ugly but I forgot the proper numpy syntax...
n_amp_err = np.array([amp_err for i in range(len(n_period))])

# ------------------------- End of comment-out


# plot period vs amp for minima 
plt.errorbar(n_amp, n_period, yerr=  n_time_err, xerr = n_amp_err, ecolor='cyan', fmt='.')
plt.xlabel("Amplitude, θ")
plt.ylabel("Period, T")
plt.title("Period vs Amplitude")
plt.show()


# plot period vs amp for maxima
plt.errorbar(p_amp, p_period, yerr=  p_time_err, xerr = p_amp_err, ecolor='cyan', fmt='.')
plt.xlabel("Amplitude, θ")
plt.ylabel("Period, T")
plt.title("Period vs Amplitude")
plt.show()






# --------------------------
# make polyfit
# this one is for n-amplitudes, change it around to p_amp, etc for postive ones

# change that number for the nth power series that you want to fit to...
# This will GROSSLY overfit for n>~4ish because of all the rounding errors
# in the data and the narrow spectrum it covers
# You will only need n=1 for (b, a) or n=2 for (c, b, a)

# fits to y = Ax^n + Bx^n-1... + chr(65+n) (which is just the letter of the term w/o an x)

pfit, perr = np.polyfit(n_amp, n_period, 1, cov=True) #<- this number is n_terms + 1!

# takes sqrt of diagonal for error
for i in range(len(perr)):
    print(chr(65+i) + ": ", str(pfit[i]) + " +/- " + str(perr[i,i]**(0.5)))

pfunc = np.poly1d(pfit)
x = np.arange(min(n_amp), max(n_amp), ((max(n_amp)-min(n_amp))/1000))
plt.xlabel("Amplitude, θ")
plt.errorbar(n_amp, n_period, yerr=  n_time_err, xerr = n_amp_err,ecolor='cyan', fmt='.')
plt.plot(x, pfunc(x))
plt.ylabel("Period, T")
plt.title("Period vs Amplitude w/ power series regression line")
plt.show()




# plot residuals of T vs Amplitude
# This can be modified to plot things for postive amplitudes by changing n_amp, etc to p_ equivialent

r = pfunc(n_amp) - n_period
plt.xlabel("Amplitude, θ")
plt.errorbar(n_amp, r, yerr=  n_time_err, xerr = n_amp_err,ecolor='cyan', fmt='.')
plt.ylabel("Period, T")
plt.title("Residuals of Period vs Amplitude w/ power series regression line")
plt.show()






















# uncomment in order to get your data in csv format

# f=open("n_data.csv", "w+")
# for i, j in enumerate(n_period):
#     f.write(str(j)+", "+str(n_amp[i])+"\n")
# f.close()

# f=open("p_data.csv", "w+")
# for i, j in enumerate(p_period):
#     f.write(str(j)+", "+str(p_amp[i])+"\n")
# f.close()

# Note that if you used tracker data and had time in (1/60)s like me, 
# your graph period v. amp will have distinct horz. bars of data
# This is because of rounding error and you should see that
# whatever varience in there is experimentally zero.


