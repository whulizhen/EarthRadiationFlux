import pdb
import numpy as np
import os
import sys
from numpy import math
from numpy.random import uniform,seed
import scipy.interpolate as sci_interp

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pyplot  import plot,savefig
from matplotlib import cm
import matplotlib.colors as mcolors

from pylab import *



from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import AxesGrid

import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy import config
import cartopy.feature as cfeature


#import data files
#import mypolygon
# import the triangle flux data
#import myflux
#import the visiable triangles
#import vt




myfont = {'family' :'serif',
			  'color'  :'darkred',
			  'weight' :'normal',
			  'size'   : 20,
		 }

# the normolize of the color map

# this is used for the abolute value plotting
cm_norm = mpl.colors.Normalize(vmin=0, vmax=400)


## definition of a class for the point data structure
class POINT:
	def __init__(self,lon, lat, hgt):
		self.lon = 0.0;
		self.lat = 0.0;
		self.hgt = 0.0;

##definition of a function for drawing a line between two points
def LinePlotting( myglobe,pointA,  pointB, linecolor ,linewidth,pointsize):
	lon = [0.0,0.0];
	lat = [0.0,0.0];
	lon[0] = pointA[0];
	lon[1] = pointB[0];
	lat[0] = pointA[1];
	lat[1] = pointB[1];
	plt.scatter(lon,lat,color ='blue',s=pointsize,transform=ccrs.Geodetic(globe=myglobe));
	plt.plot(lon,lat,color=linecolor,LineWidth=linewidth,transform=ccrs.Geodetic(globe=myglobe));


#definition of a function for converting month names
def Month2String( month ):
	if month == '01':
		return 'JAN'
	elif month == '02':
		return 'FEB'
	elif month == '03':
		return 'MAR'
	elif month == '04':
		return 'APR'
	elif month == '05':
		return 'MAY'
	elif month == '06':
		return 'JUN'
	elif month == '07':
		return 'JUL'
	elif month == '08':
		return 'AUG'
	elif month == '09':
		return 'SEP'
	elif month == '10':
		return 'OCT'
	elif month == '11':
		return 'NOV'
	elif month == '12':
		return 'DEC'

#read in the triangle net data
def parseData(datafileName):
	dataList=[];
	for line in open(datafileName):
		test = line.split();
		linelength = len(test);
		if linelength==14:  # the last level
			d=dict();
			d['name']=test[0];
			d['fname']=test[1];
			d['cname1']='';
			d['cname2']='';
			d['cname3']='';
			d['cname4']='';
			d['x1']=float(test[2]);
			d['y1']=float(test[3]);
			d['z1']=float(test[4]);
			d['x2']=float(test[5]);
			d['y2']=float(test[6]);
			d['z2']=float(test[7]);
			d['x3']=float(test[8]);
			d['y3']=float(test[9]);
			d['z3']=float(test[10]);
			d['area']=float(test[11]);
			d['longwave']=float(test[12]);
			d['shortwave']=float(test[13]);
			dataList.append(d);
		elif linelength==18: # the normal level
			d=dict();
			d['name']=test[0];
			d['fname']=test[1];
			d['cname1']=test[2];
			d['cname2']=test[3];
			d['cname3']=test[4];
			d['cname4']=test[5];
			d['x1']=float(test[6]);
			d['y1']=float(test[7]);
			d['z1']=float(test[8]);
			d['x2']=float(test[9]);
			d['y2']=float(test[10]);
			d['z2']=float(test[11]);
			d['x3']=float(test[12]);
			d['y3']=float(test[13]);
			d['z3']=float(test[14]);
			d['area']=float(test[15]);
			d['longwave']=float(test[16]);
			d['shortwave']=float(test[17]);
			dataList.append(d);
		else:
			a=0;
			print('data format error');

	return dataList;

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


#draw the backgroup map with PlateCarree projection
def drawBackgroundMap_carree():

	myglobe =  ccrs.Globe(datum='WGS84',ellipse='sphere') #,semimajor_axis=6371000,semiminor_axis=6371000,flattening=0.0);

	myprojection = ccrs.PlateCarree(central_longitude = 180);
	mycrs   = ccrs.PlateCarree();
	mytransform = ccrs.PlateCarree();

	myax = plt.subplot(1,1,1,projection=myprojection,transform=mytransform );
	myax.set_global();

	#add grid lines
	#myax.gridlines();

	myax.coastlines();
	#myax.add_feature(cfeature.LAND)
	#myax.add_feature(cfeature.OCEAN)
	#myax.add_feature(cfeature.COASTLINE)

	myax.set_xticks(np.linspace(-180, 180, 5), crs=myprojection)
	myax.set_yticks(np.linspace(-90, 90, 5), crs=myprojection)
	lon_formatter = LongitudeFormatter(zero_direction_label=True)
	lat_formatter = LatitudeFormatter()
	myax.xaxis.set_major_formatter(lon_formatter)
	myax.yaxis.set_major_formatter(lat_formatter)

	# set the font size of the labels
	plt.tick_params(labelsize=myfont['size'])
	labels = myax.get_xticklabels() + myax.get_yticklabels()
	[label.set_fontname('Times New Roman') for label in labels]

	return ( myglobe,myprojection,mycrs,mytransform,myax);

# resample the triangular flux into regular grid
# datatype can only be "longwave" or "shortwave"
def triflux_resampling(mydata,datatype):

	mylength = len(mydata);
	ilon,ilat = np.mgrid[0.5:359.5:360j, -89.5:89.5:180j];
	lon_lat_c = np.ndarray(shape=(mylength,2), dtype=float)
	fluxdata = np.ndarray(shape=(mylength))
	for index in range(mylength):
		xyzA = [mydata[index]['x1'],mydata[index]['y1'],mydata[index]['z1']];
		xyzB = [mydata[index]['x2'],mydata[index]['y2'],mydata[index]['z2']];
		xyzC = [mydata[index]['x3'],mydata[index]['y3'],mydata[index]['z3']];
		central = getCentral(xyzA,xyzB,xyzC);
		LLt =  XYZ2BL(central[0],central[1],central[2]);
		#get the central of the triangle
		lon_lat_c[index][0] = LLt[1];
		lon_lat_c[index][1] = LLt[0];
		# get the flux data
		fluxdata[index] = mydata[index][datatype];

	# doing the interpolation using scipy
	ifluxdata= sci_interp.griddata( lon_lat_c, fluxdata, (ilon,ilat),method='nearest');
	return ilat,ilon,ifluxdata


#get the difference between triangular flux and grid flux
def diff_grid_tri(myfigure,myax,mytransform,gridflux,triflux):

	aLat_g, aLon_g  = shape(gridflux);
	diff_flux = triflux - gridflux;

	print "average:", diff_flux.mean()
	print "std:", diff_flux.std()

	#showGridFlux(myfigure,myax,mytransform,diff_flux );

	lons = np.linspace(0.5,359.5,360);
	lats = np.linspace(-89.5,89.5,180);

	lons, lats = np.meshgrid(lons,lats);
	gridfluxdata = map(list, zip(*diff_flux));

	mynorm = mpl.colors.Normalize(vmin=-100, vmax=100)

	cs = myax.contourf(lons,lats,gridfluxdata,cmap=plt.get_cmap("rainbow"),norm=mynorm,transform = mytransform); #note that maybe use myfluxdata.A

	cbar = myfigure.colorbar(cs, ax=myax,orientation='vertical',spacing='uniform',fraction=0.1, pad =0.02,shrink=0.5);

	cbar.set_label('$\mathrm{ W\cdot m^{-2}}$',fontdict=myfont)
	for t in cbar.ax.get_xticklabels():
		t.set_fontsize(myfont['size']);
	for t in cbar.ax.get_yticklabels():
		t.set_fontsize(myfont['size']);

	return diff_flux;

	#print shape(triflux)
	#print shape(gridflux)

# show the original grid flux file
def showGridFlux(myfigure,myax,mytransform,gridfluxdata):

	lons = np.linspace(0.5,359.5,360);
	lats = np.linspace(-89.5,89.5,180);

	lons, lats = np.meshgrid(lons,lats);
	gridfluxdata = map(list, zip(*gridfluxdata));

	cs = myax.contourf(lons,lats,gridfluxdata,cmap=plt.get_cmap("rainbow"),norm=cm_norm,transform = mytransform); #note that maybe use myfluxdata.A

	cbar = myfigure.colorbar(cs, ax=myax,orientation='vertical',spacing='uniform',fraction=0.1, pad =0.02,shrink=0.5);

	cbar.set_label('$\mathrm{ W\cdot m^{-2}}$',fontdict=myfont)
	for t in cbar.ax.get_xticklabels():
		t.set_fontsize(myfont['size']);
	for t in cbar.ax.get_yticklabels():
		t.set_fontsize(myfont['size']);


def XYZ2BL(x,y,z):
	blh = zeros(2);
	t = math.sqrt( x*x + y*y );
	eps=1.0E-10;
	if math.fabs(t) < eps:
		if z > 0 :
			blh[0]=90.0;
			blh[1]=0.0;
		elif z < 0:
			blh[0]=-90;
			blh[1]=0.0;
	else:
		blh[0] = math.atan( z/t)*180.0/math.pi; # -90 to 90
		# 0-360
		blh[1] = math.atan2(y,x)*180.0/math.pi;
		if blh[1]< 0.0:
			blh[1] = blh[1] + 360.0;

	return blh;

def BL2XYZ(lat, lon):
	R = 6371000.0; #6371km
	eps=1.0E-10;
	xyz = zeros(3);
	if math.fabs(lat-90)<eps:
		xyz[0]=0;
		xyz[1]=0;
		xyz[2]=R;
	elif math.fabs(lat+90)<eps:
		xyz[0]=0;
		xyz[1]=0;
		xyz[2]=-R;
	else:
		xyz[0] = math.cos(lat*math.pi/180.0)*math.cos(lon*math.pi/180.0)*R;
		xyz[1] = math.cos(lat*math.pi/180.0)*math.sin(lon*math.pi/180.0)*R;
		xyz[2] = math.sin(lat*math.pi/180.0)*R;
	return xyz;


# get the central of the spherical triangle
def getCentral(xyzA,xyzB,xyzC):
	Radius = 6371000;
	central = blh = zeros(3);
	central[0] = (xyzA[0] + xyzB[0] + xyzC[0])/3.0;
	central[1] = (xyzA[1] + xyzB[1] + xyzC[1])/3.0;
	central[2] = (xyzA[2] + xyzB[2] + xyzC[2])/3.0;

	tlen = math.sqrt(central[0]*central[0]+central[1]*central[1]+central[2]*central[2]);
	central[0] = central[0]/tlen*Radius;
	central[1] = central[1]/tlen*Radius;
	central[2] = central[2]/tlen*Radius;

	return central

## draw the original grid net
#plot the longitude and latitude grids
def showGridNet(myax,mycrs):

	interval_lat = 10;
	interval_lon = 10;
	gl = myax.gridlines(crs=mycrs, draw_labels=False,
                  linewidth=2, color='red', alpha=0.5, linestyle='-')
	gl.xlabels_top = False
	gl.ylabels_left = False
	#gl.xlines = False
	#gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
	gl.xlocator = mticker.IndexLocator(interval_lon, -180);
	gl.ylocator = mticker.IndexLocator(interval_lat, -90)



def showTriNet( mytransform,myax,mydata,withcentral=False ):

	mylength = len(mydata);
	linewidth = 0.02;
	pointsize  = 0.05;
	triangleColor = 'blue';
	clon = np.zeros(mylength);
	clat = np.zeros(mylength);

	ilon = np.linspace(0.5,359.5, 360 ); #360
	ilat = np.linspace(-89.5,89.5,180 );  #180
	pointA= zeros(2);
	pointB= zeros(2);
	pointC= zeros(2);
	Lat = zeros(3);
	Lon = zeros(3);

	for index in range(mylength):
		#print mydata[index]['x1'];
		xyzA = [mydata[index]['x1'],mydata[index]['y1'],mydata[index]['z1'] ];
		xyzB = [mydata[index]['x2'],mydata[index]['y2'],mydata[index]['z2'] ];
		xyzC = [mydata[index]['x3'],mydata[index]['y3'],mydata[index]['z3'] ];

		#print 'xyzA',xyzA[0]*5.0;

		blA= XYZ2BL(xyzA[0],xyzA[1],xyzA[2]);
		blB= XYZ2BL(xyzB[0],xyzB[1],xyzB[2]);
		blC= XYZ2BL(xyzC[0],xyzC[1],xyzC[2]);

		if withcentral == True:
			central = getCentral(xyzA,xyzB,xyzC);
			LLt =  XYZ2BL(central[0],central[1],central[2]);
			#draw the central point of the triangles
			myax.scatter(LLt[1],LLt[0],color ='red',s=10,transform=ccrs.Geodetic());

		#print 'A:' ,blA[1], blA[0], 'B:', blB[1], blB[0], 'C:', blC[1],blC[0]

		Lat  = [blA[0],blB[0],blC[0]];
		Lon  = [blA[1],blB[1],blC[1]];

		poly = Polygon(zip(Lon,Lat),closed=True,edgecolor='blue',facecolor='none',transform=ccrs.Geodetic() );

		plt.gca().add_patch(poly);
		plt.gca().plot();



def fillPolygon(picdir,myax,myglobe,points):
	print 'plot and fill the polygon'

	pointA = zeros(2);
	pointB = zeros(2);
	pointC = zeros(2);

	triangleColor ='purple';
	linewidth = 0.1;
	pointsize = 0.5;

	num_of_polygon =  len(points);

	for index in range( num_of_polygon ):
		#print index
		# for the other polygon, need to redefine the Lon and Lat
		len1 = len(points[index]);
		Lon = zeros(len1);
		Lat = zeros(len1);

		#print len1

		for myindex in range(len1):
			Lat[myindex] = points[index][myindex][1];
			Lon[myindex] = points[index][myindex][0];

		poly = Polygon(zip(Lon,Lat),closed=True,fill=True,alpha=0.2,linewidth=linewidth,facecolor=triangleColor,transform=mytransform );
		# here gca means get current axis
		plt.gca().add_patch(poly);
		# here, we must call plt.plot() to plot to the current axis
		myax.plot();
		plt.gca().plot();
		title =  " the visible area of BeiDou C08 at UTC 2014-04-16-20:59:44 "  ;# + Month2String(imonth) ;
		plt.title( title ,fontsize=16);




###
## the main script for executing
def main():
	# get the path of the current python script
	script_path = os.path.split(os.path.realpath(__file__))[0];
	datadir = "/Users/lizhen/projects/EarthRadiationModel/data/"
	picdir =  script_path + "/output/";
	fluxdir = datadir + "flux/"
	picPath = "";
	tri_flux_datafile = fluxdir + "01/level6.txt"
	grid_flux_datafile = fluxdir + "01/longwave01.grid"

	# a parameter used to distinguish different tests
	varName = sys.argv[1] ;
	print varName

	print "start the plotting job, it can be time consuming! \n"

	myfigure = plt.figure( num= 1,figsize=(10,8),dpi = 127,tight_layout=True,frameon=False);
	
	tri_flux_data = parseData(tri_flux_datafile);
	grid_flux_data = np.genfromtxt(grid_flux_datafile);

	(myglobe,myprojection,mycrs,mytransform,myax) = drawBackgroundMap_carree();

	# extract the triangular data
	ilat,ilon,tri_grid_fluxdata = triflux_resampling(tri_flux_data,"longwave");
	#showGridFlux(myfigure,myax,mytransform,tri_grid_fluxdata );
	#showTriNet(mytransform,myax,tri_flux_data,False);

	#showGridFlux(myfigure,myax,mytransform,grid_flux_data );
	#showGridNet(myax,mycrs);

	# difference between triangular and grid
	diff_flux = diff_grid_tri(myfigure,myax,mytransform,grid_flux_data,tri_grid_fluxdata);
	print diff_flux.max()
	print diff_flux.min()

	#cm_norm = mpl.colors.Normalize(vmin=-30, vmax=30)
	#showGridFlux(myfigure,myax,mytransform,diff_flux );

	#fillPolygon(picdir,myax,myglobe,vt.Points);

	picName =  varName + ".pdf";
	picPath = picdir +  picName;

	plt.savefig(picPath);

	print "finished all the plotting job!"


if __name__ == '__main__':
    main()
