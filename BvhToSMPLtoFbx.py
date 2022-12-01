import sys
import fbx 
import FbxCommon
from fbx import *
from lib.serialization import load_model
from lib.serialization import save_model
import numpy as np
import pprint
import pickle as _pickle
import math
from chumpy.ch import MatVecMult
import scipy.io as sio



from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D




gExportVertexCacheMCFormat = True
gCacheType = 0
    



def CreateScene(pSdkManager, pScene, pSampleFileName):


    MeshNode,Nodi = CreateHumanModel(pSdkManager, "human", pScene, 1)  

    lAnimStack = FbxAnimStack.Create(pScene, "Take001")

    # The animation nodes can only exist on AnimLayers therefore it is mandatory to
    # add at least one AnimLayer to the AnimStack. And for the purpose of this example,
    # one layer is all we need.
    lAnimLayer = FbxAnimLayer.Create(pScene, "Base Layer")
    lAnimStack.AddMember(lAnimLayer)
    
    Animate(MeshNode, Nodi, lAnimLayer, pSdkManager,pScene, 100) 
        
    
    return True



def GetNodeHierarchy(myScene):

    myRoot = myScene.GetRootNode()

    root = fbx.FbxNode.Create(manager, "character")

    myMeshNode = fbx.FbxNode.Create(manager, 'myMesh')

    names = [ "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", 
    "R_Collar", "Head", "L_Shoulder", "R_Sholder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand", "Root"]


    Nodi = []

    for i in range(0,25):
        Nodi.append(fbx.FbxNode.Create(manager, names[i]))
        Nodi[i].SetNodeAttribute(FbxSkeleton.Create(manager, "mySkeleton"+str(i)))
        Nodi[i].GetNodeAttribute().SetSkeletonType(FbxSkeleton.eLimbNode)
    
    #Nodi[0].GetPropertyCount()

    '''for j in range(0,25):
        Nodi[j].SetNodeAttribute(FbxSkeleton.Create(manager, "mySkeleton"+str(j)))
        Nodi[j].GetNodeAttribute().SetSkeletonType(FbxSkeleton.eLimbNode)'''

    order = [24,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]

    myRoot.AddChild(root)


    root.AddChild(Nodi[24])    #root
    root.AddChild(myMeshNode)

    Nodi[24].AddChild(Nodi[0]) #pelvis

    for i in range(0, 24):
        Nodi[order[i]].AddChild(Nodi[i])


   

    '''Nodi[24].AddChild(Nodi[0])
    Nodi[0].AddChild(Nodi[1])   
    Nodi[0].AddChild(Nodi[2])   
    Nodi[0].AddChild(Nodi[3]) 

    Nodi[1].AddChild(Nodi[4])
    Nodi[2].AddChild(Nodi[5])

    Nodi[3].AddChild(Nodi[6])

    Nodi[4].AddChild(Nodi[7])
    Nodi[5].AddChild(Nodi[8])
    Nodi[6].AddChild(Nodi[9])


    Nodi[7].AddChild(Nodi[10])

    Nodi[8].AddChild(Nodi[11])


    Nodi[9].AddChild(Nodi[12])
    Nodi[9].AddChild(Nodi[13])
    Nodi[9].AddChild(Nodi[14])


    Nodi[12].AddChild(Nodi[15])

    Nodi[13].AddChild(Nodi[16])
    Nodi[14].AddChild(Nodi[17])


    Nodi[16].AddChild(Nodi[18])
    Nodi[17].AddChild(Nodi[19])
    Nodi[18].AddChild(Nodi[20])

    Nodi[19].AddChild(Nodi[21])

    Nodi[20].AddChild(Nodi[22])


    Nodi[21].AddChild(Nodi[23])'''


    return Nodi, myMeshNode


def CreateSkeleton(pSdkManager, pScene,m, myMesh):

    myRoot = myScene.GetRootNode()

    '''
    LimbTransl = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.073, -0.090, -0.008], [0.033, -0.381, -0.006], [-0.012, -0.404, -0.045], [0.024, -0.057, 0.116], [-0.071, -0.089, -0.004], [-0.040, -0.388, -0.008],
    [0.015, -0.405, -0.043], [-0.022, -0.030, 0.120], [-0.002, 0.103, -0.022], [0.003, 0.123, -0.001], [0.002, 0.048, 0.026], [0.077, 0.119, -0.036], [0.088, 0.033, -0.007], [0.261, -0.014, -0.026],
    [0.244, 0.010, -0.000], [0.079, -0.010, -0.013], [-0.079, 0.117, -0.041], [-0.090, 0.036, -0.010], [-0.255, -0.016, -0.020], [-0.252, 0.009, -0.004], [-0.080, -0.007, -0.010], [-0.003, 0.214, -0.050], 
    [0.004, 0.065, 0.050]]

    '''


    Weights = []

    Weights = m.weights

    mySkinDeformer = FbxSkin.Create(manager, 'skin')
    myMesh.AddDeformer(mySkinDeformer)


    Nodi, myMeshNode = GetNodeHierarchy(pScene)  



    count = 0
    jointRotation = np.zeros((24,3))

    for i in range(0,len(m.pose)):
        jointRotation[int(i)/3,int(i)%3] = m.pose[i]


    

    for i in range(0,24):
        jointRotation[i,0] = m.J[i,0]*180.0/math.pi
        jointRotation[i,1] = -m.J[i,1]*180.0/math.pi
        jointRotation[i,2] = m.J[i,2]*180.0/math.pi

    




    '''#plot the points
    fig = pyplot.figure()
    ax = Axes3D(fig)
    
    for j in range(0,len(J_tmpx)):
        ax.scatter(J_tmpx[j], J_tmpy[j], J_tmpz[j])

    pyplot.show()'''




    LocalPosition = []

    J_tmpx = m.J_transformed[:,0]        
    J_tmpy = m.J_transformed[:,1]        
    J_tmpz = m.J_transformed[:,2]
    

    order = [0,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]


    for i in range(0,24):
        LocalPosition.append([0.0,0.0,0.0])  

        if i == 0:
            LocalPosition[0] = [J_tmpx[0], J_tmpy[0], J_tmpz[0]] 

        if i != 0:
            LocalPosition[i] = [J_tmpx[i] - J_tmpx[order[i]], J_tmpy[i] - J_tmpy[order[i]], J_tmpz[i] - J_tmpz[order[i]]]
       

    

    Clusters = []

    skinDeformer = myMesh.GetDeformer(0, FbxDeformer.eSkin)

    meshMatrix = myMeshNode.EvaluateGlobalTransform()

    for j in range(0,24):
        Nodi[j].LclTranslation.Set(FbxDouble3(LocalPosition[j][0], -LocalPosition[j][1], LocalPosition[j][2]))
        Nodi[j].LclRotation.Set(FbxDouble3(jointRotation[j,0],jointRotation[j,1], jointRotation[j,2]))
        Clusters.append(FbxCluster.Create(pScene, "cluster"+str(j)))

        for vert in range(0, len(Weights)):
            if float(Weights[vert][j]) != 0.0:
                #Clusters[j].AddControlPointIndex(vert , float(Weights[vert][j]))
                Clusters[j].AddControlPointIndex(vert , float(Weights[vert][j]))
        print Clusters[j].GetControlPointWeights()
        print len(Clusters[j].GetControlPointWeights())
                

        Clusters[j].SetLink(Nodi[j])
        Clusters[j].SetLinkMode(0)
        Clusters[j].SetTransformMatrix(meshMatrix)
        Clusters[j].SetTransformLinkMatrix(Nodi[j].EvaluateGlobalTransform())
        skinDeformer.AddCluster(Clusters[j])



    BindPose = FbxPose.Create(pSdkManager, 'bindPose')
    BindPose.SetIsBindPose(True)


    #Da implementare: Configurare la bind pose
    '''for i in range(0,24):
        BindPose.Add(Nodi[i], Nodi[i].EvaluateGlobalTransform())'''
            
    pScene.AddPose(BindPose)
    #Clusters.append(FbxCluster.Create(pScene, "cluster"+str(24)))
    #Clusters[j].SetLink(Nodi[24])
    #skinDeformer.AddCluster(Clusters[24])


    return myMeshNode, Nodi, myMesh


def CreateHumanModel(pSdkManager, pName, myScene, frameNumber):
    myMesh = FbxMesh.Create(pSdkManager, pName)

    '''if (frameNumber < 10):
        myFile = open('../data/frames/0000'+str(frameNumber)+'_image.jpg_body.pkl', 'rb')
    elif(frameNumber < 100):
        myFile = open('../data/frames/000'+str(frameNumber)+'_image.jpg_body.pkl', 'rb')
    elif(frameNumber < 1000):
        myFile = open('../data/frames/00'+str(frameNumber)+'_image.jpg_body.pkl', 'rb')


    ComputedModel = _pickle.load(myFile)'''


    jointRotation, m = ExtractModel(frameNumber)



    '''m = load_model( '../data/model/basicModel_f_lbs_10_207_0_v1.0.0.pkl')

    #print(ComputedModel['pose'])
    m.pose[:] = ComputedModel['pose']
    m.betas[:] = ComputedModel['betas']'''

    #path = '.modello.txt'

    controlPoints = []

    vertices = m.r
    faces = m.f




    for v in vertices:
        controlPoints.append(FbxVector4(v[0], -v[1], v[2]))


    myFaces = []

    for j in range(len(faces)):
        myFaces = np.concatenate((myFaces, faces[j,:]), axis = 0)


    myMesh.InitControlPoints(len(myFaces))


    for i in range(len(controlPoints)):
        myMesh.SetControlPointAt(controlPoints[i], i)


    for i in range(len(faces)):
        myMesh.BeginPolygon() # Material index.


        myMesh.AddPolygon(faces[i,0])
        myMesh.AddPolygon(faces[i,1])
        myMesh.AddPolygon(faces[i,2])

        myMesh.EndPolygon()

    myMesh.RemoveBadPolygons()
    #print(myMesh.GetPolygonCount())

    


    #print myMesh.GetDeformer(0, FbxDeformer.eSkin)

    myNode, Nodi, myMesh = CreateSkeleton(pSdkManager, myScene, m, myMesh)
    myNode.SetNodeAttribute(myMesh)
    CreateSkinning(myNode, Nodi)



    #CreateMaterials(pSdkManager, lMesh)

    return myNode, Nodi






# Create materials for pyramid.
'''def CreateMaterials(pSdkManager, pMesh):
    colors = (FbxDouble3(0.0, 0.0, 0.0), FbxDouble3(0.0, 1.0, 1.0), FbxDouble3(0.0, 1.0, 0.0), FbxDouble3(1.0, 1.0, 1.0), FbxDouble3(1.0, 0.0, 0.0))
    
    for i in range(5):
        lMaterialName = "material"
        lShadingName = "Phong"
        lMaterialName += str(i)
        lBlack = FbxDouble3(0.0, 0.0, 0.0)
        lRed = FbxDouble3(1.0, 0.0, 0.0)
        lMaterial = FbxSurfacePhong.Create(pSdkManager, lMaterialName)

        # Generate primary and secondary colors.
        lMaterial.Emissive.Set(lBlack)
        lMaterial.Ambient.Set(lRed)
            
        lMaterial.Diffuse.Set(colors[i])
        lMaterial.TransparencyFactor.Set(0.0)
        lMaterial.ShadingModel.Set(lShadingName)
        lMaterial.Shininess.Set(0.5)

        #get the node of mesh, add material for it.
        lNode = pMesh.GetNode()
        if lNode:
            lNode.AddMaterial(lMaterial)'''





# Pyramid is translated to the right.
'''def SetNodesProperties(pNodes):
    pPyramid.LclTranslation.Set(FbxDouble3(75.0, -50.0, 0.0))
    pPyramid.LclRotation.Set(FbxDouble3(0.0, 0.0, 0.0))
    pPyramid.LclScaling.Set(FbxDouble3(1.0, 1.0, 1.0))'''

def CreateSkinning(MeshNode, Nodes):
    print 'Hello'



    
def ExtractModel(frameNumber):

    


  
    myMat = sio.loadmat('03_01_c0001_info.mat')



    m = load_model( './models/basicModel_f_lbs_10_207_0_v1.0.0.pkl')

    #print(ComputedModel['pose'])

    #m.pose[:] = ComputedModel['pose']
    poses = myMat['pose']
    m.betas[:] = np.random.rand(m.betas.size) * .03

    print poses[70][90]


    LocalPosition = []

    J_tmpx = m.J_transformed[:,0]        
    J_tmpy = m.J_transformed[:,1]        
    J_tmpz = m.J_transformed[:,2]
    

    order = [0,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]


    for i in range(0,24):
        LocalPosition.append([0.0,0.0,0.0])  

        if i == 0:
            LocalPosition[0] = [J_tmpx[0], J_tmpy[0], J_tmpz[0]] 

        if i != 0:
            LocalPosition[i] = [J_tmpx[i] - J_tmpx[order[i]], J_tmpy[i] - J_tmpy[order[i]], J_tmpz[i] - J_tmpz[order[i]]]

    jointRotation = np.zeros((24,3))

    for i in range(0,len(poses)):
        jointRotation[i/3,i%3] = poses[i][frameNumber]
       

    for i in range(0,24):
        jointRotation[i,0] = m.J[i,0]*180.0/math.pi
        jointRotation[i,1] = -m.J[i,1]*180.0/math.pi
        jointRotation[i,2] = m.J[i,2]*180.0/math.pi



    return jointRotation, m


def Animate(myModel,Nodi, pAnimLayer, pSdkManager,pScene, frameNumber):

    #myModel, Nodi = CreateHumanModel(pSdkManager, "Pyramid", pScene)

    myRoot = pScene.GetRootNode()

    root = myRoot.GetChild(0)





    p = FbxProperty.Create(pScene, FbxDouble3DT, "Vector3Property");
    p.Set(FbxDouble3(1.1, 2.2, 3.3));
    lCurveNode = FbxAnimCurveNode.CreateTypedCurveNode(p, pScene);

    pAnimLayer.AddMember(lCurveNode);

    p.ConnectSrcObject(lCurveNode);

    lCurve = FbxAnimCurve.Create(pScene, "curve1");


    lTime = FbxTime()

    '''MyAnimCurveNode = root.LclRotation.GetCurveNode(pAnimLayer, True)
    xlCurve = root.LclRotation.GetCurve(pAnimLayer, "X", True)
    ylCurve = root.LclRotation.GetCurve(pAnimLayer, "Y", True)
    zlCurve = root.LclRotation.GetCurve(pAnimLayer, "Z", True)'''

   
    key = FbxAnimCurveKey()

    if lCurve:

        for numFrame in range(0, frameNumber):
            jointRotation, m = ExtractModel(numFrame)
            lTime.SetSecondDouble(numFrame * 0.1)

            for ind in range(0,24):




                lCurve.KeyModifyBegin()
     

                key.Set(lTime, jointRotation[ind,0])

                lCurve.KeyAdd(lTime, key)

                lCurve.KeyModifyEnd()

                lCurveNode.ConnectToChannel(lCurve, 'X');

                lCurve.KeyModifyBegin()
                



                key.Set(lTime, jointRotation[ind,1])

                lCurve.KeyAdd(lTime, key)

                lCurve.KeyModifyEnd()

                lCurveNode.ConnectToChannel(lCurve, 'Y');

                lCurve.KeyModifyBegin()
                



                key.Set(lTime, jointRotation[ind,2])

                lCurve.KeyAdd(lTime, key)

                lCurve.KeyModifyEnd()

                lCurveNode.ConnectToChannel(lCurve, 'Z');


        print 'fino qui'



        





        '''lTime.SetSecondDouble(0.0)
            xlKeyIndex = xlCurve.KeyAdd(lTime)[0]
            ylKeyIndex = ylCurve.KeyAdd(lTime)[0]
            zlKeyIndex = zlCurve.KeyAdd(lTime)[0]

            #thisTranslation = thisNode.LclTranslation.Get()

            #print(thisTranslation[0])
            MyAnimCurveNode = Nodi[ind].LclRotation.GetCurveNode(pAnimLayer, True)
            xlCurve = Nodi[ind].LclRotation.GetCurve(pAnimLayer, "X", True)
            ylCurve = Nodi[ind].LclRotation.GetCurve(pAnimLayer, "Y", True)
            zlCurve = Nodi[ind].LclRotation.GetCurve(pAnimLayer, "Z", True)

            xlCurve.KeyModifyBegin()
            xlCurve.KeySetValue(xlKeyIndex, jointRotation[ind,0])
            xlCurve.KeySetInterpolation(xlKeyIndex, FbxAnimCurveDef.eInterpolationCubic)

          

            ylCurve.KeyModifyBegin()
            ylCurve.KeySetValue(ylKeyIndex, jointRotation[ind,1])
            ylCurve.KeySetInterpolation(ylKeyIndex, FbxAnimCurveDef.eInterpolationCubic)

            zlCurve.KeyModifyBegin()
            zlCurve.KeySetValue(zlKeyIndex, jointRotation[ind,2])
            zlCurve.KeySetInterpolation(zlKeyIndex, FbxAnimCurveDef.eInterpolationCubic)

            xlCurve.KeyModifyEnd()
            ylCurve.KeyModifyEnd()
            zlCurve.KeyModifyEnd()

      
            



        jointRotation = ExtractModel(8)

        for ind in range(0,24):

            lTime.SetSecondDouble(1.0)
            xlKeyIndex = xlCurve.KeyAdd(lTime)[0]
            ylKeyIndex = ylCurve.KeyAdd(lTime)[0]
            zlKeyIndex = zlCurve.KeyAdd(lTime)[0]

            #thisTranslation = thisNode.LclTranslation.Get()

            #print(thisTranslation[0])
            MyAnimCurveNode = Nodi[ind].LclRotation.GetCurveNode(pAnimLayer, True)
            xlCurve = Nodi[ind].LclRotation.GetCurve(pAnimLayer, "X", True)
            ylCurve = Nodi[ind].LclRotation.GetCurve(pAnimLayer, "Y", True)
            zlCurve = Nodi[ind].LclRotation.GetCurve(pAnimLayer, "Z", True)

            xlCurve.KeyModifyBegin()
            xlCurve.KeySetValue(xlKeyIndex, jointRotation[ind,0])
            xlCurve.KeySetInterpolation(xlKeyIndex, FbxAnimCurveDef.eInterpolationCubic)

          

            ylCurve.KeyModifyBegin()
            ylCurve.KeySetValue(ylKeyIndex, jointRotation[ind,1])
            ylCurve.KeySetInterpolation(ylKeyIndex, FbxAnimCurveDef.eInterpolationCubic)

            zlCurve.KeyModifyBegin()
            zlCurve.KeySetValue(zlKeyIndex, jointRotation[ind,2])
            zlCurve.KeySetInterpolation(zlKeyIndex, FbxAnimCurveDef.eInterpolationCubic)

            xlCurve.KeyModifyEnd()
            ylCurve.KeyModifyEnd()
            zlCurve.KeyModifyEnd()


        jointRotation = ExtractModel(9)

        for ind in range(0,24):

            lTime.SetSecondDouble(2.0)
            xlKeyIndex = xlCurve.KeyAdd(lTime)[0]
            ylKeyIndex = ylCurve.KeyAdd(lTime)[0]
            zlKeyIndex = zlCurve.KeyAdd(lTime)[0]

            #thisTranslation = thisNode.LclTranslation.Get()

            #print(thisTranslation[0])
            MyAnimCurveNode = Nodi[ind].LclRotation.GetCurveNode(pAnimLayer, True)
            xlCurve = Nodi[ind].LclRotation.GetCurve(pAnimLayer, "X", True)
            ylCurve = Nodi[ind].LclRotation.GetCurve(pAnimLayer, "Y", True)
            zlCurve = Nodi[ind].LclRotation.GetCurve(pAnimLayer, "Z", True)
            
            xlCurve.KeyModifyBegin()
            xlCurve.KeySetValue(xlKeyIndex, jointRotation[ind,0])
            xlCurve.KeySetInterpolation(xlKeyIndex, FbxAnimCurveDef.eInterpolationCubic)

          

            ylCurve.KeyModifyBegin()
            ylCurve.KeySetValue(ylKeyIndex, jointRotation[ind,1])
            ylCurve.KeySetInterpolation(ylKeyIndex, FbxAnimCurveDef.eInterpolationCubic)

            zlCurve.KeyModifyBegin()
            zlCurve.KeySetValue(zlKeyIndex, jointRotation[ind,2])
            zlCurve.KeySetInterpolation(zlKeyIndex, FbxAnimCurveDef.eInterpolationCubic)

            xlCurve.KeyModifyEnd()
            ylCurve.KeyModifyEnd()
            zlCurve.KeyModifyEnd()

       

        jointRotation = ExtractModel(10)

        for ind in range(0,24):

            lTime.SetSecondDouble(3.0)
            xlKeyIndex = xlCurve.KeyAdd(lTime)[0]
            ylKeyIndex = ylCurve.KeyAdd(lTime)[0]
            zlKeyIndex = zlCurve.KeyAdd(lTime)[0]

            #thisTranslation = thisNode.LclTranslation.Get()

            #print(thisTranslation[0])
            MyAnimCurveNode = Nodi[ind].LclRotation.GetCurveNode(pAnimLayer, True)
            xlCurve = Nodi[ind].LclRotation.GetCurve(pAnimLayer, "X", True)
            ylCurve = Nodi[ind].LclRotation.GetCurve(pAnimLayer, "Y", True)
            zlCurve = Nodi[ind].LclRotation.GetCurve(pAnimLayer, "Z", True)
            
            xlCurve.KeyModifyBegin()
            xlCurve.KeySetValue(xlKeyIndex, jointRotation[ind,0])
            xlCurve.KeySetInterpolation(xlKeyIndex, FbxAnimCurveDef.eInterpolationCubic)

          

            ylCurve.KeyModifyBegin()
            ylCurve.KeySetValue(ylKeyIndex, jointRotation[ind,1])
            ylCurve.KeySetInterpolation(ylKeyIndex, FbxAnimCurveDef.eInterpolationCubic)

            zlCurve.KeyModifyBegin()
            zlCurve.KeySetValue(zlKeyIndex, jointRotation[ind,2])
            zlCurve.KeySetInterpolation(zlKeyIndex, FbxAnimCurveDef.eInterpolationCubic)

            xlCurve.KeyModifyEnd()
            ylCurve.KeyModifyEnd()
            zlCurve.KeyModifyEnd()

      

        jointRotation = ExtractModel(11)

        for ind in range(0,24):

            lTime.SetSecondDouble(4.0)
            xlKeyIndex = xlCurve.KeyAdd(lTime)[0]
            ylKeyIndex = ylCurve.KeyAdd(lTime)[0]
            zlKeyIndex = zlCurve.KeyAdd(lTime)[0]

            #thisTranslation = thisNode.LclTranslation.Get()

            #print(thisTranslation[0])
            MyAnimCurveNode = Nodi[ind].LclRotation.GetCurveNode(pAnimLayer, True)
            xlCurve = Nodi[ind].LclRotation.GetCurve(pAnimLayer, "X", True)
            ylCurve = Nodi[ind].LclRotation.GetCurve(pAnimLayer, "Y", True)
            zlCurve = Nodi[ind].LclRotation.GetCurve(pAnimLayer, "Z", True)
            
            xlCurve.KeyModifyBegin()
            xlCurve.KeySetValue(xlKeyIndex, jointRotation[ind,0])
            xlCurve.KeySetInterpolation(xlKeyIndex, FbxAnimCurveDef.eInterpolationCubic)

          

            ylCurve.KeyModifyBegin()
            ylCurve.KeySetValue(ylKeyIndex, jointRotation[ind,1])
            ylCurve.KeySetInterpolation(ylKeyIndex, FbxAnimCurveDef.eInterpolationCubic)

            zlCurve.KeyModifyBegin()
            zlCurve.KeySetValue(zlKeyIndex, jointRotation[ind,2])
            zlCurve.KeySetInterpolation(zlKeyIndex, FbxAnimCurveDef.eInterpolationCubic)

            xlCurve.KeyModifyEnd()
            ylCurve.KeyModifyEnd()
            zlCurve.KeyModifyEnd()'''

        












        

    '''lCurve = pNode.LclRotation.GetCurve(pAnimLayer, "X", True)
    if lCurve:
        lTime.SetSecondDouble(2.0)
        lKeyIndex = lCurve.KeyAdd(lTime)[0]

        lCurve.KeyModifyBegin()
        lCurve.KeySetValue(lKeyIndex, 0.0)
        lCurve.KeySetInterpolation(lKeyIndex, FbxAnimCurveDef.eInterpolationCubic)

        lTime.SetSecondDouble(2.5)
        lKeyIndex = lCurve.KeyAdd(lTime)[0]
        lCurve.KeySetValue(lKeyIndex, 90.0)
        lCurve.KeySetInterpolation(lKeyIndex, FbxAnimCurveDef.eInterpolationCubic)

        lTime.SetSecondDouble(3.5)
        lKeyIndex = lCurve.KeyAdd(lTime)[0]
        lCurve.KeySetValue(lKeyIndex, -90.0)
        lCurve.KeySetInterpolation(lKeyIndex, FbxAnimCurveDef.eInterpolationCubic)

        lTime.SetSecondDouble(4.0)
        lKeyIndex = lCurve.KeyAdd(lTime)[0]
        lCurve.KeySetValue(lKeyIndex, 0.0)
        lCurve.KeySetInterpolation(lKeyIndex, FbxAnimCurveDef.eInterpolationCubic)
        lCurve.KeyModifyEnd()'''

    # The upside down shape is at index 0 because it is the only one.
    # The cube has no shape so the function returns NULL is this case.
    '''lGeometry = pNode.GetNodeAttribute()
    lCurve = lGeometry.GetShapeChannel(0, 0, pAnimLayer, True)
    if lCurve:
        lTime.SetSecondDouble(0.0)
        lKeyIndex = lCurve.KeyAdd(lTime)[0]

        lCurve.KeyModifyBegin()
        lCurve.KeySetValue(lKeyIndex, 0.0)
        lCurve.KeySetInterpolation(lKeyIndex, FbxAnimCurveDef.eInterpolationCubic)

        lTime.SetSecondDouble(2.0)
        lKeyIndex = lCurve.KeyAdd(lTime)[0]
        lCurve.KeySetValue(lKeyIndex, 100.0)
        lCurve.KeySetInterpolation(lKeyIndex, FbxAnimCurveDef.eInterpolationCubic)

        lTime.SetSecondDouble(4.0)
        lKeyIndex = lCurve.KeyAdd(lTime)[0]
        lCurve.KeySetValue(lKeyIndex, 0.0)
        lCurve.KeySetInterpolation(lKeyIndex, FbxAnimCurveDef.eInterpolationCubic)
        lCurve.KeyModifyEnd()'''


#def AnimateVertexCacheOnTriangleDoubleVertex(pTriangle, pFrameRate):

#def AnimateVertexCacheOnTriangleInt32(pTriangle, pFrameRate):



if __name__ == "__main__":
    

    # Prepare the FBX SDK.
    #(manager, myScene) = FbxCommon.InitializeSdkObjects()

    manager = fbx.FbxManager.Create()

    IOsettings = FbxIOSettings.Create(manager, IOSROOT)

    manager.SetIOSettings(IOsettings)

    '''manager.GetIOSettings().SetBoolProp(fbx.EXP_MATERIAL, true)
    manager.GetIOSettings().SetBoolProp(EXP_TEXTURE, true)
    manager.GetIOSettings().SetBoolProp(EXP_SHAPE, true)
    manager.GetIOSettings().SetBoolProp(EXP_GOBO, true)
    manager.GetIOSettings().SetBoolProp(EXP_ANIMATION, true)
    manager.GetIOSettings().SetBoolProp(EXP_GLOBAL_SETTINGS, true)'''

    # Create a scene
    myScene = fbx.FbxScene.Create(manager, "")

    
    # Create the scene.
    lResult = CreateScene(manager, myScene, "d")

    

    # Save the scene.
    FbxCommon.SaveScene(manager, myScene, "prova")

    exporter = fbx.FbxExporter.Create(manager, "")

    save_path = "./prova.fbx"

# Specify the path and name of the file to be imported                                                                            
    exportstat = exporter.Initialize(save_path, -1)

    exportstat = exporter.Export(myScene)

    
    # Destroy all objects created by the FBX SDK.
    manager.Destroy()
   
    sys.exit(0)




