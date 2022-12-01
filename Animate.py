import fbx 
import FbxCommon
from fbx import *
import model as sas
import sys
#from lib.serialization import load_model
import numpy as np
from numpy import linalg
import math
import transforms3d
from chumpy.ch import MatVecMult
import chumpy as chump
from scipy.sparse import csr_matrix
from serialization import load_model





def Pose(frameNumber, animMat, m):

    myBetas = np.zeros((10))
   
    myBetas = np.random.rand(m.betas.size) * .03
    m.pose[:] = animMat['pose'][:,frameNumber]
    m.betas[:] = myBetas


    vertices = sas.getVertices()



    LocalPosition = []

    LocalPosition = calcLocalPosition(m, vertices)



    jointRotation = np.zeros((24,3))

    for i in range(0,len(m.pose)):
        jointRotation[int(i)/3,int(i)%3] = m.pose[i]

 
    for i in range(0,24):
        identity = []
        skew = []
        vector = []
        matrix = []

        vector = jointRotation[i,:]/linalg.norm(jointRotation[i,:])

        identity.append([1.0,0.0,0.0])
        identity.append([0.0,1.0,0.0])
        identity.append([0.0,0.0,1.0])

        skew.append([0.0, -vector[2], vector[1]])
        skew.append([vector[2], 0.0, -vector[0]])
        skew.append([-vector[1], vector[0], 0.0])
        skew = np.array(skew)

        matrix = identity + skew*math.sin(linalg.norm(jointRotation[i,:]))+np.matmul(skew,skew)*(1.0-math.cos(linalg.norm(jointRotation[i,:])))


        jointRotation[i,:] = np.array(transforms3d.euler.mat2euler(matrix))*180.0/math.pi

    
    trans1 = animMat['joints3D'][:,0,frameNumber]*int(sys.argv[4])



    return jointRotation, LocalPosition, trans1



def calcLocalPosition(m, vertices):

    myLocalPosition = []


    regressor = m.J_regressor.todense()


    J_tmpx = np.matmul(regressor,vertices[:,0])
    J_tmpy = np.matmul(regressor,vertices[:,1])
    J_tmpz = np.matmul(regressor,vertices[:,2])
    #J = chump.vstack((J_tmpx, J_tmpy, J_tmpz)).T


    J_tmpx = np.squeeze(np.asarray(J_tmpx))
    J_tmpy = np.squeeze(np.asarray(J_tmpy))
    J_tmpz = np.squeeze(np.asarray(J_tmpz))

    '''J_tmpx = J[:,0]        
    J_tmpy = J[:,1]        
    J_tmpz = J[:,2]'''




    order = [0,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]


    for i in range(0,24):
        myLocalPosition.append([0.0,0.0,0.0])  

        if i == 0:
            myLocalPosition[0] = [J_tmpx[0], J_tmpy[0], J_tmpz[0]] 

        if i != 0:
            #myLocalPosition[i] = [J_tmpx[i], J_tmpy[i], J_tmpz[i]]
            myLocalPosition[i] = [J_tmpx[i] - J_tmpx[order[i]], J_tmpy[i] - J_tmpy[order[i]], J_tmpz[i] - J_tmpz[order[i]]]


    return myLocalPosition



#impostare input initialframe e finalframe
def Animate(myModel,Nodi, pAnimLayer, pSdkManager,pScene, Rotation, myMat, initFrame, finalFrame, translation, m):

    #myModel, Nodi = CreateHumanModel(pSdkManager, "Pyramid", pScene)
    #m = load_model( './models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')

    myRoot = pScene.GetRootNode()

    root = myRoot.GetChild(0)



    lCurve = FbxAnimCurve.Create(pScene, "curve1");
    lTime = FbxTime()



    if lCurve:


        seconds = 0.0
        count = 0.0

        toolbar_width = 19


        #sys.stdout.write("[%s]" % (" " * toolbar_width))
        sys.stdout.flush()
        #sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
      
        for ind in range(0,24):

            sys.stdout.write("joint " + str(ind+1))
            sys.stdout.flush()


            if Rotation:
                lCurveX = Nodi[ind].LclRotation.GetCurve(pAnimLayer, "X", True)
                lCurveY = Nodi[ind].LclRotation.GetCurve(pAnimLayer, "Y", True)
                lCurveZ = Nodi[ind].LclRotation.GetCurve(pAnimLayer, "Z", True)
                lCurveTransX = Nodi[ind].LclTranslation.GetCurve(pAnimLayer, "X", True)
                lCurveTransY = Nodi[ind].LclTranslation.GetCurve(pAnimLayer, "Y", True)
                lCurveTransZ = Nodi[ind].LclTranslation.GetCurve(pAnimLayer, "Z", True)
            else:
                lCurveX = Nodi[ind].LclTranslation.GetCurve(pAnimLayer, "X", True)
                lCurveY = Nodi[ind].LclTranslation.GetCurve(pAnimLayer, "Y", True)
                lCurveZ = Nodi[ind].LclTranslation.GetCurve(pAnimLayer, "Z", True)

            lCurveX.KeyModifyBegin()
            lCurveY.KeyModifyBegin()
            lCurveZ.KeyModifyBegin()
            lCurveTransX.KeyModifyBegin()
            lCurveTransY.KeyModifyBegin()
            lCurveTransZ.KeyModifyBegin()

            count = 0.0

            # video1  --------------  137, 150



        


            for frame in range(initFrame+1,finalFrame):


                jointRotation, LocalPosition, trans = Pose(frame, myMat, m)

                lTime.SetSecondDouble(seconds+count)

                lKeyIndexX = lCurveX.KeyAdd(lTime)[0]
                lKeyIndexY = lCurveY.KeyAdd(lTime)[0]
                lKeyIndexZ = lCurveZ.KeyAdd(lTime)[0]
                

                if ind==0:

                    lKeyIndexX = lCurveTransX.KeyAdd(lTime)[0]
                    lKeyIndexY = lCurveTransY.KeyAdd(lTime)[0]
                    lKeyIndexZ = lCurveTransZ.KeyAdd(lTime)[0]
                   
                    lCurveX.KeySetValue(lKeyIndexX, jointRotation[ind][0])
                    if translation:
                        lCurveTransX.KeySetValue(lKeyIndexX, LocalPosition[ind][0] - trans[2])
                    else:
                        lCurveTransX.KeySetValue(lKeyIndexX, LocalPosition[ind][0])

                    lCurveX.KeySetInterpolation(lKeyIndexX, FbxAnimCurveDef.eInterpolationLinear)

                    
                    lCurveY.KeySetValue(lKeyIndexY, jointRotation[ind][1])                        
                    
                    if translation:
                        lCurveTransY.KeySetValue(lKeyIndexY, LocalPosition[ind][1] + trans[0])
                    else:
                        lCurveTransY.KeySetValue(lKeyIndexY, LocalPosition[ind][1])

                    lCurveY.KeySetInterpolation(lKeyIndexY, FbxAnimCurveDef.eInterpolationLinear)

                
                    lCurveZ.KeySetValue(lKeyIndexZ, jointRotation[ind][2])
                    if translation:
                        lCurveTransZ.KeySetValue(lKeyIndexZ, LocalPosition[ind][2] - trans[1])
                    else:
                        lCurveTransZ.KeySetValue(lKeyIndexZ, LocalPosition[ind][2])

                    lCurveZ.KeySetInterpolation(lKeyIndexZ, FbxAnimCurveDef.eInterpolationLinear)



                else:
                    if Rotation:
                        lCurveX.KeySetValue(lKeyIndexX, jointRotation[ind][0])
                    else:
                        lCurveX.KeySetValue(lKeyIndexX, LocalPosition[ind][0])
                    lCurveX.KeySetInterpolation(lKeyIndexX, FbxAnimCurveDef.eInterpolationCubic)

                    if Rotation:
                        lCurveY.KeySetValue(lKeyIndexY, jointRotation[ind][1])                        
                    else:
                        lCurveY.KeySetValue(lKeyIndexY, -LocalPosition[ind][1])
                    lCurveY.KeySetInterpolation(lKeyIndexY, FbxAnimCurveDef.eInterpolationCubic)

                    if Rotation:
                        lCurveZ.KeySetValue(lKeyIndexZ, jointRotation[ind][2])
                    else:
                        lCurveZ.KeySetValue(lKeyIndexZ, LocalPosition[ind][2])
                    lCurveZ.KeySetInterpolation(lKeyIndexZ, FbxAnimCurveDef.eInterpolationCubic)

                count += 0.02
                # DA MODIFICARE: in base agli fps del video, settare il passo in modo da rendere l'animazione sincronizzata col video
                

            lCurveX.KeyModifyEnd()
            lCurveY.KeyModifyEnd()
            lCurveZ.KeyModifyEnd()
            lCurveTransX.KeyModifyEnd()
            lCurveTransY.KeyModifyEnd()
            lCurveTransZ.KeyModifyEnd()

            sys.stdout.write("\n")
