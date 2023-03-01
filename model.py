import sys
import fbx 
import FbxCommon
from fbx import *
#from lib.serialization import load_model
#from lib.serialization import save_model
from serialization import save_model
from serialization import load_model
import numpy as np
import pprint
import pickle as _pickle
import math
from chumpy.ch import MatVecMult
import transforms3d
from sklearn import preprocessing
from numpy import linalg
import scipy.io as sio
import Animate as anim
from chumpy.ch import MatVecMult





from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D





# impostare input path
def CreateScene(pSdkManager, pScene, pSampleFileName):


    myMat = sio.loadmat(sys.argv[1])

    MeshNode,Nodi = CreateHumanModel(pSdkManager, "human", pScene, 6, myMat)  

    lAnimStack = FbxAnimStack.Create(pScene, "Take001")

    # The animation nodes can only exist on AnimLayers therefore it is mandatory to
    # add at least one AnimLayer to the AnimStack. And for the purpose of this example,
    # one layer is all we need.
    lAnimLayer = FbxAnimLayer.Create(pScene, "Base Layer")
    lAnimStack.AddMember(lAnimLayer)

    #print pScene.GetGlobalSettings().GetAxisSystem().GetUpVector()
    
    anim.Animate(MeshNode, Nodi, lAnimLayer, pSdkManager,pScene, True, myMat, initFrame, finalFrame, translation) 
        
    
    return True



def GetNodeHierarchy(myScene, manager):

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
    
 

    order = [24,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]

    myRoot.AddChild(root)


    root.AddChild(Nodi[24])    #root
    root.AddChild(myMeshNode)

    Nodi[24].AddChild(Nodi[0]) #pelvis

    for i in range(0, 24):
        Nodi[order[i]].AddChild(Nodi[i])


    #


    return Nodi, myMeshNode


def CreateSkeleton(pSdkManager, pScene,m, myMesh):

    myRoot = pScene.GetRootNode()

   


    Weights = []

    Weights = m.weights

    

    mySkinDeformer = FbxSkin.Create(pSdkManager, 'skin')
    myMesh.AddDeformer(mySkinDeformer)


    Nodi, myMeshNode = GetNodeHierarchy(pScene, pSdkManager)  



    count = 0
    jointRotation = np.zeros((24,3))

    pose = np.zeros((72,1))

    for i in range(0,len(m.pose)):
        jointRotation[int(i)/3,int(i)%3] = pose[i]

    

    vertices = getVertices()


    LocalPosition = []

    LocalPosition = anim.calcLocalPosition(m, vertices)
       

       

    Clusters = []

    skinDeformer = myMesh.GetDeformer(0, FbxDeformer.eSkin)

    meshMatrix = myMeshNode.EvaluateGlobalTransform()

    Clusters.append(FbxCluster.Create(pScene, "cluster"+str(0)))
    Clusters[0].SetLink(Nodi[24])
    Clusters[0].SetLinkMode(0)
    Clusters[0].SetTransformMatrix(meshMatrix)
    Clusters[0].SetTransformLinkMatrix(Nodi[24].EvaluateGlobalTransform())
    skinDeformer.AddCluster(Clusters[0])

    #myWeights = np.zeros((24,6890))
    outputFile = 'myWeights.npy'
    myWeights = np.load(outputFile)
    #print myWeights


    for j in range(0,24):
        Nodi[j].LclTranslation.Set(FbxDouble3(LocalPosition[j][0], LocalPosition[j][1], LocalPosition[j][2]))
        Nodi[j].LclRotation.Set(FbxDouble3(jointRotation[j][0], jointRotation[j][1], jointRotation[j][2]))
        Clusters.append(FbxCluster.Create(pScene, "cluster"+str(j+1)))

        for vert in range(0, 6890):
            if float(myWeights[j][vert]) != 0:
               
                #Clusters[j].AddControlPointIndex(vert , float(Weights[vert][j]))
                Clusters[j+1].AddControlPointIndex(vert , myWeights[j][vert])
                #myWeights[j][vert] = Weights[vert][j]

           

        Clusters[j+1].SetLink(Nodi[j])
        Clusters[j+1].SetLinkMode(0)
        Clusters[j+1].SetTransformMatrix(meshMatrix)
        Clusters[j+1].SetTransformLinkMatrix(Nodi[j].EvaluateGlobalTransform())
        skinDeformer.AddCluster(Clusters[j+1])

   



    BindPose = FbxPose.Create(pSdkManager, 'bindPose')
    BindPose.SetIsBindPose(True)


    #Da implementare: Configurare la bind pose
    BindPose.Add(Nodi[24], FbxMatrix(Nodi[24].EvaluateGlobalTransform()))
    for i in range(0,24):
        BindPose.Add(Nodi[i], FbxMatrix(Nodi[i].EvaluateGlobalTransform()))
            
    pScene.AddPose(BindPose)
 


    return myMeshNode, Nodi, myMesh

def getVertices():

    newGeometry = sio.loadmat(sys.argv[3])

    return newGeometry['VERT']


def CreateHumanModel(pSdkManager, pName, myScene, frameNumber, myMat, initFrame, meshMat):
    myMesh = FbxMesh.Create(pSdkManager, pName)




    jointRotation, LocalPosition, m, trans = ExtractModel(initFrame, myMat, initFrame, meshMat)





    controlPoints = []

    vertices = getVertices();
    faces = m.f




    for v in vertices:
        controlPoints.append(FbxVector4(v[0], v[1], v[2]))


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




def AttachTexture(mesh):

    normalPosZ = fbx.FbxVector4( 0, 0, 1 ) # positive-Z normal vector. 


    layer = mesh.GetLayer( 0 )
    if( not layer ):
        mesh.CreateLayer()
        layer = mesh.GetLayer( 0 )
    # Create a normal layer element.
    normalLayerElement = fbx.FbxLayerElementNormal.Create( mesh, 'normals' )
    
    # We want to have one normal for each vertex (or control point),    # so we set the mapping mode to eByControlPoint
    normalLayerElement.SetMappingMode( fbx.FbxLayerElement.eByControlPoint )
    
    # Set the normal values for every control point.
    normalLayerElement.SetReferenceMode( fbx.FbxLayerElement.eDirect )
    
    global normalPosZ # positive-Z normals.
    normalLayerElement.GetDirectArray().Add( normalPosZ )
    normalLayerElement.GetDirectArray().Add( normalPosZ )
    normalLayerElement.GetDirectArray().Add( normalPosZ )
    normalLayerElement.GetDirectArray().Add( normalPosZ )
    
    # Assign the normal layer element to the mesh's layer 0.
    layer.SetNormals( normalLayerElement )



def CreateSkinning(MeshNode, Nodes):
    #print 'Hello'
    Nodes = Nodes



    
def ExtractModel(frameNumber, myMat, initFrame, meshMat):

    
    m = load_model( './models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')




    m.pose[:] = np.zeros((72))




    




    LocalPosition = []

    LocalPosition = anim.calcLocalPosition(m, getVertices())




    


    jointRotation = np.zeros((24,3))

    for i in range(0,len(m.pose)):
        jointRotation[int(i)/3,int(i)%3] = m.pose[i]

    if frameNumber!=initFrame:
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

    
    trans1 = np.zeros((3,1))


    return jointRotation, LocalPosition, m, trans1




