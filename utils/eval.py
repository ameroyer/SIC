#!/usr/bin/python
#coding: utf-8

# Author: vincent.claveau@irisa.fr
#
# Purpose: Clustering evaluation
#
# Comment:

# Code:

##########################################################################
#                            INITIALIZATION                              #
##########################################################################
from __future__ import division

if __name__ == '__main__':
    import time, os, subprocess, codecs, sys, glob, re, operator
    try: import cPickle as pickle
    except: import pickle
    from math import *

    reload(sys)
    sys.setdefaultencoding('utf-8')


    from collections import defaultdict

    prg = sys.argv[0]
    def P(output=''): raw_input(output+"\nDebug point; Press ENTER to continue")
    def Info(output=''): sys.stderr.write(output+'\n')
    def Warn(output=''): sys.stderr.write('warning:'+output+'\n')

    #######################################
    # special imports

    import numpy
    from numpy import *
    from scipy import *
    from scipy.stats import mode
    from scipy.misc.common import factorial



    #######################################
    # files
    #######################################
    # variables

    #########################################
    # USAGE

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-in", "--in", dest="file_input",
                      help="file containing the txt to be analyzed", metavar="STR")
    parser.add_argument("-ref", "--ref", dest="file_ref",
                      help="file containing the txt to be analyzed", metavar="STR")

    parser.add_argument("-m", "--missing",
                      action="store_false", dest="cluster_missing", default=True,
                      help="consider ref obj missing in input as one additional cluster")

    parser.add_argument("-o", "--out", dest="prefix",
                      help="PREFIX for output files", metavar="STR")

    parser.add_argument("-v", "--verbose",
                      action="store_false", dest="verbose", default=True,
                      help="print status messages to stdout")

    args = parser.parse_args()



################################################################################
################################################################################
##                                                                            ##
##                                 FUNCTIONS                                  ##
##                                                                            ##
################################################################################
################################################################################








def combination(a,k):
    result=1
    for i in range(k):
        result=result*((a-i)/(k-i))
    return result

def P(predicted,labels):
    K=unique(predicted)
    p=0
    for cls in K:
        cls_members=nonzero(predicted==cls)[0]
        if cls_members.shape[0]<=1:
            continue
        real_label=mode(labels[cls_members])[0][0]
        correctCount=nonzero(labels[cls_members]==real_label)[0].shape[0]
        p+=double(correctCount)/cls_members.shape[0]
    return p/K.shape[0]


def R(predicted,labels):
    K=unique(predicted)
    ccount=0
    for cls in K:
        cls_members=nonzero(predicted==cls)[0]
        real_label=mode(labels[cls_members])[0][0]
        ccount+=nonzero(labels[cls_members]==real_label)[0].shape[0]
    return double(ccount)/predicted.shape[0]


def F(predicted,labels):
    p=P(predicted,labels)
    r=R(predicted,labels)
    return 2*p*r/(p+r),p,r

#Adjusted purity
#http://acl.ldc.upenn.edu/acl2002/MAIN/pdfs/Main303.pdf
def APP(predicted,labels):
    K=unique(predicted)
    app=0.0
    for cls in K:
        cls_members=nonzero(predicted==cls)[0]
        if cls_members.shape[0]<=1:
            continue
        real_labels=labels[cls_members]
        correct_pairs=0
        for i in range(real_labels.shape[0]):
            for i2 in range(i+1):
                if real_labels[i]==real_labels[i2]:
                        correct_pairs+=2
        total=cls_members.shape[0]
        total_pairs=total+1
        app+=double(correct_pairs)/(total_pairs*total)
    return double(app)/K.shape[0]


#Mutual information
def mutual_info(x,y):
    N=double(x.size)
    I=0.0
    eps = numpy.finfo(float).eps
    for l1 in unique(x):
        for l2 in unique(y):
            #Find the intersections
            l1_ids=nonzero(x==l1)[0]
            l2_ids=nonzero(y==l2)[0]
            pxy=(double(intersect1d(l1_ids,l2_ids).size)/N)+eps
            I+=pxy*log2(pxy/((l1_ids.size/N)*(l2_ids.size/N)))
    return I


#Normalized mutual information
def nmi(x,y):
    N=x.size
    I=mutual_info(x,y)
    Hx=0
    for l1 in unique(x):
        l1_count=nonzero(x==l1)[0].size
        Hx+=-(double(l1_count)/N)*log2(double(l1_count)/N)
    Hy=0
    for l2 in unique(y):
        l2_count=nonzero(y==l2)[0].size
        Hy+=-(double(l2_count)/N)*log2(double(l2_count)/N)
    return I/((Hx+Hy)/2)

def grp2idx(labels):
    inds=dict()
    for label in labels:
        if label not in inds:
            inds[label]=len(inds)
    return array([inds[label] for label in labels])

#Vmeasure
def V(predicted,labels):
    predicted=grp2idx(predicted)
    labels=grp2idx(labels)

    a=zeros((unique(labels).size,unique(predicted).size))
    for i in range(a.shape[0]):
        for i2 in range(a.shape[1]):
            a[i,i2]=intersect1d(nonzero(labels==i)[0],nonzero(predicted==i2)[0]).size
    N=labels.size
    n=a.shape[0]
    a=double(a)
    Hck=0
    Hc=0
    Hkc=0
    Hk=0
    for i in range(a.shape[0]):
        for i2 in range(a.shape[1]):
            if a[i,i2]>0:
                Hkc+=(a[i,i2]/N)*log(a[i,i2]/sum(a[i,:]))
                Hck+=(a[i,i2]/N)*log(a[i,i2]/sum(a[:,i2]))
        Hc+=(sum(a[i,:])/N)*log(sum(a[i,:])/N)
    Hck=-Hck
    Hkc=-Hkc
    Hc=-Hc
    for i in range(a.shape[1]):
        ak=sum(a[:,i])
        Hk+=(ak/n)*log(ak/N)
    Hk=-Hk

    h=1-(Hck/Hc)
    c=1-(Hkc/Hc)
    vmeasure=(2*h*c)/(h+c)
    return vmeasure

#Mutal information
#http://acl.ldc.upenn.edu/acl2002/MAIN/pdfs/Main303.pdf
def mi(predicted,labels):
    predicted=grp2idx(predicted)
    labels=grp2idx(labels)
    a=zeros((unique(labels).size,unique(predicted).size))
    for i in range(a.shape[0]):
        for i2 in range(a.shape[1]):
            a[i,i2]=intersect1d(nonzero(labels==i)[0],nonzero(predicted==i2)[0]).size

    a=double(a)
    n=labels.size
    mi=0
    for i in range(a.shape[0]):
        for i2 in range(a.shape[1]):
            if a[i,i2]>0:
                mi+=a[i,i2]*log((a[i,i2]*n)/(sum(a[i,:])*sum(a[:,i2])))
    mi=mi/log(unique(labels).size*unique(predicted).size)
    mi=mi/n
    return mi

#Adjusted rand index
#http://acl.ldc.upenn.edu/eacl2003/papers/main/p39.pdf
def rand(predicted,labels):
    predicted=grp2idx(predicted)
    labels=grp2idx(labels)
    a=zeros((unique(labels).size,unique(predicted).size))
    for i in range(a.shape[0]):
        for i2 in range(a.shape[1]):
            a[i,i2]=intersect1d(nonzero(labels==i)[0],nonzero(predicted==i2)[0]).size
    cij=0
    a=double(a)
    for i in range(a.shape[0]):
        for i2 in range(a.shape[1]):
            if a[i,i2]>1:
                cij+=combination(a[i,i2],2)
    ci=0
    for i in range(a.shape[0]):
        if sum(a[i,:])>1:
            ci+=combination(sum(a[i,:]),2)
    cj=0
    for i in range(a.shape[1]):
        if sum(a[:,i])>1:
            cj+=combination(sum(a[:,i]),2)
    cn=combination(double(labels.size),2)
    nominator=cij-((ci*cj)/cn)
    denominator=0.5*(ci+cj)-(ci*cj/cn)
    #print cij, ci, cj, cn
    return nominator/denominator




################################################################################
################################################################################
##                                                                            ##
##                                   CODE                                     ##
##                                                                            ##
################################################################################
################################################################################




#Example from http://nlp.stanford.edu/IR-book
#/html/htmledition/evaluation-of-clustering-1.html
#    print completeness(array([1,2,3]),array([1,2,2]))
#
#print rand(array([1,1,6,1,1,1,2,2,2,2,2,2,3,3,3,3,3])
#           ,array([2,5,1,1,3,1,2,2,2,2,2,2,3,3,3,3,3]))





def Read_file (file_cl):
    # un cluster = une ligne d'onbjets speares par des tab
    h_cl,num_cl = {}, 1

    for line in codecs.open(file_cl,'r','utf-8'):
        for o in line.rstrip('\r\n').split('\t'):
            h_cl[o]=num_cl
        num_cl += 1
    num_cl -= 1
    return h_cl,num_cl



if __name__ == '__main__':
    h_in,num_cl_in = Read_file(args.file_input)
    print 'read',num_cl_in,'clusters in file',args.file_input
    h_ref,num_cl_ref = Read_file(args.file_ref)
    print 'read',num_cl_ref,'clusters in file',args.file_ref
    # on construit le dict inverse pour les ref
    h_cl2obj_ref = defaultdict(lambda : [])
    for o in h_ref:
        h_cl2obj_ref[h_ref[o]].append(o)



    #################

    t_intersect = sorted([o for o in h_ref if o in h_in])
    t_not_in_input = [o for o in h_ref if o not in h_in]
    t_not_in_ref = [o for o in h_in if o not in h_ref]

    mylog = codecs.open(args.prefix+'.log','w','utf-8')
    from output_format import Tee
    sys.stdout = Tee(sys.stdout, mylog)
    mylog.write('\n\nin input but not in ref:\n\t'+'\n\t'.join(t_not_in_ref)+'\n\n')
    mylog.write('\n\nin ref but not in input:\n\t'+'\n\t'.join(t_not_in_input)+'\n\n')

    print len(t_not_in_ref),'obj in input not in ref, ', len(t_not_in_input),'obj of ref not in input'
    print t_intersect[:10]
    print t_not_in_input[:10]
    print t_not_in_ref[:10]

    if args.cluster_missing:
        print 'adding missing in input as a new cluster'
        t_intersect += t_not_in_input
        for o in t_not_in_input: h_in[o]=0


    ########

    print 'remaing of analysis carried on the',len(t_intersect),'wide intersection'

    if len(t_intersect)<2: sys.exit()

    # tranformer le format 1 cluster par ligne en foramt 1obj -> 1 num de cluster
    array_in = array([h_in[o] for o in t_intersect])
    array_ref = array([h_ref[o] for o in t_intersect])



    print 'V measure = ',V(array_in,array_ref)
    print 'mutual info = ',mi(array_in,array_ref)
    print 'normalized mutual info = ',nmi(array_in,array_ref)
    print 'adjusted purity = ',APP(array_in,array_ref)
    print 'adjusted rand index =',rand(array_in,array_ref)

    h_cross = defaultdict(lambda: defaultdict(lambda:0))
    for i in xrange(len(array_in)):
        h_cross[array_in[i]][array_ref[i]]+=1 #matrice de confusion

    print 'reference:\t','\t\t'.join([h_cl2obj_ref[cl][0] for cl in xrange(1,num_cl_ref+1)])
    print '-'*142
    for cl1 in xrange(1,num_cl_in+1):
        print 'cluster',cl1,'\t','\t\t'.join([str(h_cross[cl1][cl2]) for cl2 in xrange(1,num_cl_ref+1)])

    mylog.close()
