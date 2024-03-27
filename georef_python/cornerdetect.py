import cv2
import numpy as np
import math
from math import *
import pytesseract
import re
import os
import time

# max angle difference from horizontal/vertical
deltaMax=4

# path of images to process
imgpath="./maps_test/"
#imgpath='d:/geodata/gk_gyorshelyesbites/'

maxfiles=50

# output file name
outfile='vacakcorners_v3.csv'       #'all_corners2.csv'

# erosion kernel
ek=np.ones((5,5),np.uint8)

# outer corner refinement kernels
Ktl = np.ones((7, 7)) * -1
Ktl[3:7, 3:7] = 1
Ktl[3, 3] = 0
Ktl/=15
Ktr = np.ones((7, 7)) * -1
Ktr[3:7, 0:4] = 1
Ktr[3, 3] = 0
Ktr/=15
Kbl = np.ones((7, 7)) * -1
Kbl[0:4, 3:7] = 1
Kbl[3, 3] = 0
Kbl/=15
Kbr = np.ones((7, 7)) * -1
Kbr[0:4, 0:4] = 1
Kbr[3, 3] = 0
Kbr/=15


# calculates rho from line endpoints
def rho(x1,y1,x2,y2):
    l=math.sqrt((y2-y1)**2+(x2-x1)**2)
    rho=abs(x2*y1-y2*x1)/l
    return rho

# calculates the intersection of two lines
def intersection(la,lb):
    (x1a,y1a,x2a,y2a)=la
    (x1b,y1b,x2b,y2b)=lb
    # line steepness
    ma=(y2a-y1a)/(x2a-x1a) if x1a!=x2a else None
    mb=(y2b-y1b)/(x2b-x1b) if x2b!=x1b else None
    if (ma==mb):
        # lines are parallel
        return None
    if (x2a==x1a):
        # "la" vertical
        x=x1a
    elif (x2b==x1b):
        # "lb" vertical
        x=x1b
    else:
        x=(y1b-mb*x1b-y1a+ma*x1a)/(ma-mb)
    y=y1a+ma*(x-x1a) if ma is not None else y1b+mb*(x-x1b)
    return (int(x),int(y))

# pick horizontal and vertical ones from a list of lines
def getHorizVert(lines,margin=0,debug=False):
    horiz=[]
    vert=[]
    hr=[]
    vr=[]
    if not lines is None:
        for l in lines:
            (x1, y1, x2, y2)=l[0]
            alfa=math.atan2(abs(x1-x2),abs(y1-y2))/math.pi*180
            if debug:
                print(l,alfa)
            if alfa<deltaMax:
                r=rho(x1,y1,x2,y2)
                rr=(x1+x2)/2
                if r>w*margin and r<w*(1-margin):
                    vert.append(l[0])
                    vr.append(rr)
            if alfa>90-deltaMax:
                r=rho(x1,y1,x2,y2)
                rr=(y1+y2)/2
                if r > h * margin and r < h * (1-margin):
                    horiz.append(l[0])
                    hr.append(rr)
    return horiz,hr,vert,vr

# refine corner position in image by local corner/intersection search
def refineCorner(img,p,size,outer=False,top=True,left=True):
    (x,y)=p
    x0=x-int(size/2)
    y0=y-int(size/2)
    subimg=img[y0:y0+int(size),x0:x0+int(size)].copy()
    if outer:
        subimg=cv2.erode(subimg,ek)
        if top:
            K=Ktl if left else Ktr
        else:
            K = Kbl if left else Kbr
        M=cv2.filter2D(subimg,-1,K).flatten()
        if not top:
            M=M[::-1]
        am=M.argmax()
        if not top:
            am=len(M)-am
        amy=am//int(size)
        amx=am%int(size)
        """cv2.circle(subimg,(amx,amy),5,255,2)
        cv2.imshow("S",subimg)
        cv2.waitKey()"""
        return (x0+amx,y0+amy)

    # find lines
    lines=cv2.HoughLinesP(subimg,
                          rho=1, #2 if outer else 1,
                          theta=np.pi / 180/3 if outer else np.pi/180*2,
                          minLineLength=0.3*size if outer else 0.5*size,
                          maxLineGap=2 if outer else 1,
                          threshold=int(0.2*size) if outer else int(0.4*size)
                          )
    horiz, hr, vert, vr = getHorizVert(lines)
    """if outer:
        print(lines)
        if not lines is None:
            print("lines found: %d. horizontal: %d, vertical: %d"%(len(lines),len(horiz),len(vert)))
        cv2.imshow('er',subimg)
        cv2.waitKey()"""
    if len(horiz)>0 and len(vert)>0:
        hr=np.array(hr)
        vr=np.array(vr)
        #print(horiz)
        #print(vert)
        hi=hr.argmin() if top else hr.argmax()
        vi=vr.argmin() if left else vr.argmax()
        #print(hi)
        #print(vi)
        (x,y)=intersection(horiz[hi],vert[vi])
        print('ref: %f %f : %f %f'%(p+(x,y)))
        x+=x0
        y+=y0
    """else:
        getHorizVert(lines,debug=True)
        cv2.imshow("S",subimg)
        cv2.waitKey()"""
    return (x,y)

# Sheet bounding box from Gauss-Krüger sheet ID
def boundsFromGKSheet(s):
    i1=ord(s[0])-ord('A')
    i2=int(s[2:4])
    i3=int(s[5:8])
    i4=s[9]
    i5=s[11]
    n0=i1*4
    e0=i2*6-186
    n1=(11-(i3-1)//12)/3
    e1=((i3-1)%12)/2
    n2=1/6 if i4 in 'AB' else 0
    e2=1/4 if i4 in 'BD' else 0
    n3=1/12 if i5 in 'ab' else 0
    e3=1/8 if i5 in 'bd' else 0
    W=e0+e1+e2+e3
    S=n0+n1+n2+n3
    E=W+1/8
    N=S+1/12
    return (W,S,E,N)

# output text
outtext=''
# write result to file
of=open(outfile,'w')

# iterating over all images
t0=time.time()
n=0
for f in os.scandir(imgpath):
    print(f.name)
    # load image
    im=cv2.imread(imgpath+f.name)
    if im is None:
        continue # not an image file, skip
    (W, S, E, N) = boundsFromGKSheet(f.name[4:-4]) # bounds based on filename
    (h,w)=im.shape[:2]
    backup=im.copy()
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    ch=cv2.split(im)
    d1=abs(cv2.absdiff(ch[0],ch[1]))
    d2=abs(cv2.absdiff(ch[0],ch[2]))
    m1=cv2.inRange(ch[0],0,180)
    m4=cv2.inRange(ch[1],0,180)
    m5=cv2.inRange(ch[2],0,180)
    m2=cv2.inRange(d1,0,70)
    m3=cv2.inRange(d2,0,70)
    mask=cv2.bitwise_and(cv2.bitwise_and(cv2.bitwise_and(m1,m2),cv2.bitwise_and(m3,m4)),m5)
    mmm=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    cv2.imwrite('res_prob/mask_'+f.name,mmm)
    #cv2.imshow('m',cv2.resize(mask,(int(w/2),int(h/2))))
    #cv2.waitKey()

    lines=cv2.HoughLinesP(mask,
                          rho=1,
                          theta=np.pi / 180/3,
                          minLineLength=w/4,
                          maxLineGap=10,
                          threshold=int(w/8)
                          )
    #print(len(lines))

    horiz,hr,vert,vr=getHorizVert(lines,0.01)

    for (x1,y1,x2,y2) in horiz:
        cv2.line(mmm, (x1, y1), (x2, y2), (0, 0, 255), 4)
    for (x1,y1,x2,y2) in vert:
        cv2.line(mmm, (x1, y1), (x2, y2), (255, 0, 255), 4)
    #cv2.imwrite(imgpath + 'res/hwlines_' + f.name,mmm)
    cv2.imshow('m',cv2.resize(mmm,(int(w/3),int(h/3))))
    cv2.waitKey()

    vr=np.array(vr)
    leftmost=vr.argmin()
    rightmost=vr.argmax()
    #print('left: '+str(vert[leftmost]))
    (x1,y1,x2,y2)=vert[leftmost]
    cv2.line(im, (x1, y1), (x2, y2), (0, 255, 0), 5)
    (x1,y1,x2,y2)=vert[rightmost]
    #print('right: '+str(vert[rightmost]))
    cv2.line(im, (x1, y1), (x2, y2), (0, 255, 0), 5)

    hr=np.array(hr)
    topmost=hr.argmin()
    (x1,y1,x2,y2)=horiz[topmost]
    #print('top: '+str(horiz[topmost]))
    cv2.line(im, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # top left corner
    tl=intersection(vert[leftmost],horiz[topmost])
    cv2.circle(im,tl,7,(0,0,255),3)
    # top right corner
    tr=intersection(vert[rightmost],horiz[topmost])
    cv2.circle(im,tr,7,(0,0,255),3)

    # drop bottom lines that are surely outside the frame
    # gross map width (with frame)
    gmw=vr[rightmost]-vr[leftmost]
    #print('gross width: '+str(gmw))
    # estimated max rho of bottom frame

    brho=hr[topmost]+gmw*2/3/cos(radians(N))*1.006 # 1.0 for Hungary, to be calculated exactly :)
    cv2.line(im, (0, int(brho)), (w, int(brho)), (0, 255, 255), 5)

    # DEBUG
    bottommost=hr.argmax()
    (x1,y1,x2,y2)=horiz[bottommost]
    cv2.line(im, (x1, y1), (x2, y2), (255, 255, 0), 5)
    # DEBUG

    hr[hr>brho]=0 # drop rhos where line is clearly below the map
    bottommost=hr.argmax()

    (x1,y1,x2,y2)=horiz[bottommost]
    #print('bottom: '+str(horiz[bottommost]))
    cv2.line(im, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # bottom left corner
    bl=intersection(vert[leftmost],horiz[bottommost])
    cv2.circle(im,bl,7,(0,0,255),3)
    # bottom right corner
    br=intersection(vert[rightmost],horiz[bottommost])
    cv2.circle(im,br,7,(0,0,255),3)
    cv2.imshow('m',cv2.resize(im,(int(w/3),int(h/3))))
    cv2.waitKey()

    #cv2.imwrite(imgpath + 'res/outercorners_' + f.name, im)

    # estimating the positions of the inner corners
    innerH=372 # inner height of a map sheet (mm)
    frameW=10 # frame width (mm)
    outerH=innerH+2*frameW # outer height (mm)
    oH=cv2.norm((bl[0]-tl[0],bl[1]-tl[1])) # observed outer height with frame (pixels)
    fW=frameW*oH/outerH # estimated frame width in pixels
    print("estimated frame width in pixels: %f"%fW)

    # put four corners together
    rot = atan2(tr[1] - tl[1], tr[0] - tl[0])
    itlx=int(tl[0]+fW*(cos(rot)-sin(rot)))
    itly=int(tl[1]+fW*(cos(rot)+sin(rot)))
    itrx=int(tr[0]-fW*(cos(rot)+sin(rot)))
    itry=int(tr[1]+fW*(cos(rot)-sin(rot)))
    iblx=int(bl[0]+fW*(cos(rot)+sin(rot)))
    ibly=int(bl[1]-fW*(cos(rot)-sin(rot)))
    ibrx=int(br[0]-fW*(cos(rot)-sin(rot)))
    ibry=int(br[1]-fW*(cos(rot)+sin(rot)))
    fWg=int(fW*1.5)
    cimg=np.zeros((2*fWg,8*fWg,3),dtype='uint8')
    cimg[0:2*fWg,0:2*fWg]=im[itly-fWg:itly+fWg,itlx-fWg:itlx+fWg]
    cimg[0:2*fWg,2*fWg:4*fWg]=im[itry-fWg:itry+fWg,itrx-fWg:itrx+fWg]
    cimg[0:2*fWg,4*fWg:6*fWg]=im[ibly-fWg:ibly+fWg,iblx-fWg:iblx+fWg]
    cimg[0:2*fWg,6*fWg:8*fWg]=im[ibry-fWg:ibry+fWg,ibrx-fWg:ibrx+fWg]
    #cv2.imshow('cr',cimg)
    #cv2.imwrite(imgpath + 'res/cornersonly1_' + f.name, cimg)

    # refine outer corners
    #print('tl')
    tl=refineCorner(mask,tl,fW*.9,True,True,True)
    #print('tr')
    tr=refineCorner(mask,tr,fW*.9,True,True,False)
    #print('bl')
    bl=refineCorner(mask,bl,fW*.9,True,False,True)
    #print('br')
    br=refineCorner(mask,br,fW*.9,True,False,False)
    cv2.circle(im,tl,7,(255,0,0),3)
    cv2.circle(im,tr,7,(255,0,0),3)
    cv2.circle(im,bl,7,(255,0,0),3)
    cv2.circle(im,br,7,(255,0,0),3)
    #cv2.imwrite(imgpath + 'res/outercornersref_' + f.name, im)
    # put four corners together
    fWg=int(fW*1.5)
    cimg=np.zeros((2*fWg,8*fWg,3),dtype='uint8')
    cimg[0:2*fWg,0:2*fWg]=im[itly-fWg:itly+fWg,itlx-fWg:itlx+fWg]
    cimg[0:2*fWg,2*fWg:4*fWg]=im[itry-fWg:itry+fWg,itrx-fWg:itrx+fWg]
    cimg[0:2*fWg,4*fWg:6*fWg]=im[ibly-fWg:ibly+fWg,iblx-fWg:iblx+fWg]
    cimg[0:2*fWg,6*fWg:8*fWg]=im[ibry-fWg:ibry+fWg,ibrx-fWg:ibrx+fWg]
    #cv2.imshow('cr',cimg)
    #cv2.waitKey()
    #cv2.imwrite(imgpath + 'res/cornersonly2_' + f.name, cimg)

    # calculate frame rotation angle
    #print(tr)
    #print(tl)
    #print((tr[1]-tl[1],tr[0]-tl[0]))
    rot=atan2(tr[1]-tl[1],tr[0]-tl[0])
    print("rotation: %f°"%degrees(rot))

    fWl=fW*1.32 if W==20.5 and 47<=S<=48.42 else fW
    fWb=fW*1.12 if S==47 and W<=20.5 else fW

    itlx=int(tl[0]+fWl*(cos(rot)-sin(rot)))
    itly=int(tl[1]+fW*(cos(rot)+sin(rot)))
    itrx=int(tr[0]-fW*(cos(rot)+sin(rot)))
    itry=int(tr[1]+fW*(cos(rot)-sin(rot)))
    iblx=int(bl[0]+fWl*(cos(rot)+sin(rot)))
    ibly=int(bl[1]-fWb*(cos(rot)-sin(rot)))
    ibrx=int(br[0]-fW*(cos(rot)-sin(rot)))
    ibry=int(br[1]-fWb*(cos(rot)+sin(rot)))

    cv2.circle(im,(itlx,itly),7,(255,0,255),3)
    cv2.circle(im,(itrx,itry),7,(255,0,255),3)
    cv2.circle(im,(iblx,ibly),7,(255,0,255),3)
    cv2.circle(im,(ibrx,ibry),7,(255,0,255),3)
    #cv2.imwrite(imgpath + 'res/innercorners_' + f.name, im)

    print(((itlx,itly),(itrx,itry),(iblx,ibly),(ibrx,ibry)))
    # put four corners together
    fWg=int(fW*1.5)
    cimg=np.zeros((2*fWg,8*fWg,3),dtype='uint8')
    cimg[0:2*fWg,0:2*fWg]=im[itly-fWg:itly+fWg,itlx-fWg:itlx+fWg]
    cimg[0:2*fWg,2*fWg:4*fWg]=im[itry-fWg:itry+fWg,itrx-fWg:itrx+fWg]
    cimg[0:2*fWg,4*fWg:6*fWg]=im[ibly-fWg:ibly+fWg,iblx-fWg:iblx+fWg]
    cimg[0:2*fWg,6*fWg:8*fWg]=im[ibry-fWg:ibry+fWg,ibrx-fWg:ibrx+fWg]
    #cv2.imshow('cr',cimg)
    #cv2.imwrite(imgpath + 'res/cornersonly3_' + f.name, cimg)

    # refine corner positions
    print('refining...')
    (itlx,itly)=refineCorner(mask,(itlx,itly),fW*.5,False,True,True)
    (itrx,itry)=refineCorner(mask,(itrx,itry),fW*.5,False,True,False)
    (iblx,ibly)=refineCorner(mask,(iblx,ibly),fW*.5,False,False,True)
    (ibrx,ibry)=refineCorner(mask,(ibrx,ibry),fW*.5,False,False,False)

    cv2.circle(im,(itlx,itly),7,(0,128,255),3)
    cv2.circle(im,(itrx,itry),7,(0,128,255),3)
    cv2.circle(im,(iblx,ibly),7,(0,128,255),3)
    cv2.circle(im,(ibrx,ibry),7,(0,128,255),3)
    print(((itlx,itly),(itrx,itry),(iblx,ibly),(ibrx,ibry)))
    #cv2.imwrite(imgpath + 'res_prob/innrecornersref_' + f.name, im)

    # crop roi for sheet id
    titlebottom=int(max(tl[1],tr[1]))
    titletop=int(titlebottom-fW*2)
    titlefield=backup[titletop:titlebottom,int(w/3):int(2*w/3)] # magyar gk25k
    #titlefield = backup[titletop:titlebottom, 0:int(w / 5)] # zöldfoki utm 25k
    #titlefield = backup[0:titlebottom, int(0.7*w):w]

    # run OCR on title field
    txt=pytesseract.image_to_string(titlefield)
    print(txt)
    #cv2.imshow('tf',titlefield)
    #cv2.waitKey()
    m=re.search('([A-Z])-([0-9]+)-([0-9]+)-([A-D])-([a-d])',txt) # Gauss-Krüger sheet ID pattern
    if m is None:
        m=re.search('([\w])-([0-9]+)-([0-9]+)-([\w])-([\w¢])',txt)
    # m=re.search('([A-Z])-([\|1IlVX]+)-([NS])([WE]) ([1-4zZ])',txt) # zöldfoki utm
    #m = re.search('([\|1Il]*)-([0-9\|]+)-([0-9\|]+)-([0-9\|])',txt)
    print(m)
    sheetId=m.group(0) if m is not None else "error!"
    sheetId=sheetId.replace('¢','c')
    print(sheetId)
    cv2.putText(im,sheetId,(int(w*0.4),100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),10)

    #cv2.imshow('l',cv2.resize(im,(int(w/6),int(h/6))))
    #cv2.imshow('l',im[0:1000,0:1000,:])

    # put four corners together
    fWg=int(fW*1.5)
    cimg=np.zeros((2*fWg,8*fWg,3),dtype='uint8')
    cimg[0:2*fWg,0:2*fWg]=im[itly-fWg:itly+fWg,itlx-fWg:itlx+fWg]
    cimg[0:2*fWg,2*fWg:4*fWg]=im[itry-fWg:itry+fWg,itrx-fWg:itrx+fWg]
    cimg[0:2*fWg,4*fWg:6*fWg]=im[ibly-fWg:ibly+fWg,iblx-fWg:iblx+fWg]
    cimg[0:2*fWg,6*fWg:8*fWg]=im[ibry-fWg:ibry+fWg,ibrx-fWg:ibrx+fWg]
    cv2.imshow('cr',cimg)
    #cv2.imwrite(imgpath + 'res_prob/cornersonly4_' + f.name, cimg)

    cv2.waitKey()
    #cv2.imwrite('na.jpg',im)

    # append image info to output
    outtext=f.name+','+str(itlx)+','+str(itly)+','+str(itrx)+','+str(itry)+','+str(iblx)+','+str(ibly)+','+str(ibrx)+','+str(ibry)+','+sheetId+'\n'
    of.write(outtext)

    # DEBUG ONLY: process only "maxfiles" files
    n += 1
    print(str(n)+' file processed.')
    #if n >= maxfiles:
    #    break

# close file
of.close()

t1=time.time()
print('Elapsed time: '+str(t1-t0))
exit(0)
