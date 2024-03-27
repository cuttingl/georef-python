from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import time


# Sheet bounding box from Gauss-Krüger sheet ID
def boundsFromGKSheet(s):
    i1 = ord(s[0]) - ord('A')
    i2 = int(s[2:4])
    i3 = int(s[5:8])
    i4 = s[9]
    i5 = s[11]
    n0 = i1 * 4
    e0 = i2 * 6 - 186
    n1 = (11 - (i3 - 1) // 12) / 3
    e1 = ((i3 - 1) % 12) / 2
    n2 = 1 / 6 if i4 in 'AB' else 0
    e2 = 1 / 4 if i4 in 'BD' else 0
    n3 = 1 / 12 if i5 in 'ab' else 0
    e3 = 1 / 8 if i5 in 'bd' else 0
    W = e0 + e1 + e2 + e3
    S = n0 + n1 + n2 + n3
    E = W + 1 / 8
    N = S + 1 / 12
    return (W, S, E, N)


t0 = time.time()

kras = osr.SpatialReference()
kras.ImportFromProj4("+proj=longlat +ellps=krass +towgs84=17.20,-84.03,-60.97,1.085,0.682,-0.473,-3.185"
                     " +axis=neu +no_defs")  # s-42 pulkovo földrajzi
bessel = osr.SpatialReference()
bessel.ImportFromProj4("+proj=longlat +ellps=bessel +towgs84=566.54,108.52,487.93,2.2867,2.6409,-1.5194,-0.7365"
                       " +axis=neu +no_defs")  # elcseszett magyar Bessel

targ33 = osr.SpatialReference()
targ34 = osr.SpatialReference()
targ34k = osr.SpatialReference()
targ33.ImportFromProj4("+proj=tmerc +lat_0=0 +lon_0=15 +k=1 +x_0=3500000 +y_0=0 +ellps=bessel"
                       " +towgs84=566.54,108.52,487.93,2.2867,2.6409,-1.5194,-0.7365 +units=m +axis=neu +no_defs")
# Magyar Bessel Gauss-Krüger zone 3
targ34.ImportFromProj4("+proj=tmerc +lat_0=0 +lon_0=21 +k=1 +x_0=4500000 +y_0=0 +ellps=bessel"
                       " +towgs84=566.54,108.52,487.93,2.2867,2.6409,-1.5194,-0.7365 +units=m +axis=neu +no_defs")
# Magyar Bessel Gauss-Krüger zone 4
targ34k.ImportFromProj4("proj=tmerc +lat_0=0 +lon_0=21 +k=1 +x_0=4500000 +y_0=0 +ellps=krass"
                        " +towgs84=17.20,-84.03,-60.97,1.085,0.682,-0.473,-3.185 +units=m +axis=neu +no_defs")
# s-42 pulkovo Gauss-Krüger zone 4

w84=osr.SpatialReference()
w84.ImportFromEPSG(4326)

transform33 = osr.CoordinateTransformation(bessel, targ33)
transform34 = osr.CoordinateTransformation(bessel, targ34)
transform34k = osr.CoordinateTransformation(kras, targ34k)

imgpath = './maps_test/'
outpath = 'kesz_hi_w84/'
corners = 'corners_v3_jav.csv'
with open(corners, 'r') as f:
    sheets = f.read().split('\n')

n = 0
for s in sheets:
    s = s.split(',')
    if len(s) < 2:
        continue
    (W, S, E, N) = boundsFromGKSheet(s[0][4:-4])
    T = transform33 if W < 18 else transform34 if W < 20.5 or S < 47 else transform34k
    targ = targ33 if W < 18 else targ34 if W < 20.5 or S < 47 else targ34k
    src = bessel if W < 20.5 or S < 47 else kras
    G = []
    p = ogr.CreateGeometryFromWkt("POINT (%f %f)" % (N, W))
    p.Transform(T)
    G.append(gdal.GCP(p.GetY(), p.GetX(), 0, int(s[1]), int(s[2])))
    p = ogr.CreateGeometryFromWkt("POINT (%f %f)" % (N, E))
    p.Transform(T)
    G.append(gdal.GCP(p.GetY(), p.GetX(), 0, int(s[3]), int(s[4])))
    p = ogr.CreateGeometryFromWkt("POINT (%f %f)" % (S, W))
    p.Transform(T)
    G.append(gdal.GCP(p.GetY(), p.GetX(), 0, int(s[5]), int(s[6])))
    p = ogr.CreateGeometryFromWkt("POINT (%f %f)" % (S, E))
    p.Transform(T)
    G.append(gdal.GCP(p.GetY(), p.GetX(), 0, int(s[7]), int(s[8])))

    opts = gdal.TranslateOptions(GCPs=G, format="GTiff", outputSRS=targ)
    gdal.Translate('g.tif', imgpath + s[0], options=opts)
    Wcrop=20.49895 if W==20.5 and S>=47 else W
    Scrop=46.9997 if W>=20.5 and S==47 else S
    #gdal.Warp('g2.tif', 'g.tif', dstSRS=src, outputBounds=(Wcrop, S, E, N))
    gdal.Warp(outpath + s[0][0:-4] + '.tif', 'g.tif', dstSRS=w84, outputBoundsSRS=src, outputBounds=(Wcrop, Scrop, E, N), creationOptions=["COMPRESS=JPEG"])
    n += 1
    print(n, s[0])
#    if n >= 10:
#        break

t1 = time.time()
print("Elapsed time: %f s" % (t1 - t0))
