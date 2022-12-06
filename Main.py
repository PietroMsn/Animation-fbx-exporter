import fbx 
import FbxCommon
from fbx import *
import model as sas
import sys
import scipy.io as sio
import Animate as anim
#from lib.serialization import load_model
from serialization import load_model




gExportVertexCacheMCFormat = True
gCacheType = 0



    
initFrame = 0
finalFrame = len(sio.loadmat(sys.argv[1])['pose'][1])


videoStr = 'video8/frames/'
translation = False

# Select the SMPL model 
#gender = 'Model_f'
gender = 'model_m'


'''
Create a 3d scene ready to be exported in fbx file
'''
def CreateScene(pSdkManager, pScene, pSampleFileName):

    m = load_model( './models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')

    print(m)
    #Loads the .mat from SURREAL for the pose data, passed as first argument in commandline
    myMat = sio.loadmat(sys.argv[1])

    meshMat = sio.loadmat(sys.argv[3])


    # Creates the 3d model ready to be animated
    MeshNode,Nodi = sas.CreateHumanModel(pSdkManager, "human", pScene, 6, myMat, initFrame, meshMat)  


    lAnimStack = FbxAnimStack.Create(pScene, "Take001")

    # The animation nodes can only exist on AnimLayers therefore it is mandatory to
    # add at least one AnimLayer to the AnimStack. And for the purpose of this example,
    # one layer is all we need.
    lAnimLayer = FbxAnimLayer.Create(pScene, "Base Layer")
    lAnimStack.AddMember(lAnimLayer)

    #print pScene.GetGlobalSettings().GetAxisSystem().GetUpVector()
    
    # Animates the model using the animation data in 'myMat' 
    anim.Animate(MeshNode, Nodi, lAnimLayer, pSdkManager,pScene, True, myMat, initFrame, finalFrame, translation, m) 
        
    
    return True



if __name__ == "__main__":
    

    # Prepare the FBX SDK.
    #(manager, myScene) = FbxCommon.InitializeSdkObjects()



    manager = FbxManager.Create()

    IOsettings = FbxIOSettings.Create(manager, IOSROOT)

    manager.SetIOSettings(IOsettings)

    IOsettings.SetBoolProp(EXP_FBX_MATERIAL, True)
    IOsettings.SetBoolProp(EXP_FBX_TEXTURE, True)
    IOsettings.SetBoolProp(EXP_FBX_EMBEDDED, True)  # or False if you want ASCII

    # Create an empty scene
    myScene = FbxScene.Create(manager, "")

    
    # Create the complete scene with all the animation elements
    lResult = CreateScene(manager, myScene, "d")

    

    # Saves the scene 
    FbxCommon.SaveScene(manager, myScene, "prova")
    # creates the exporter
    exporter = FbxExporter.Create(manager, "")


    # exports the scene in fbx format at the path:
    save_path = "./resFbx/"
    filename = sys.argv[2] + ".fbx"

    # Specify the path and name of the file to be imported                                                                            
    exportstat = exporter.Initialize(save_path + filename, -1)

    exportstat = exporter.Export(myScene)

    
    # Destroy all objects created by the FBX SDK.
    manager.Destroy()
   
    sys.exit(0)



