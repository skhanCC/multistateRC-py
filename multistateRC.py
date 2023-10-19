import numpy as np
import numpy.linalg as la
import scipy as sp

# Visualization
import matplotlib.pyplot as plt

# Function to return trained weights for the classification of an arbitrary number of states
def calcMultiStateWO(dataMat, NTrain, rP=0.0):
    
    # Extract data properties from data matrix
    C     = np.shape(dataMat)[0]
    NTraj = np.shape(dataMat)[1]
    D     = np.shape(dataMat)[2]
    NT    = np.shape(dataMat)[3]
    
    
################################# Compiled Matrices ################################
    
    # Construct matrix of readout features
    for q in range(C):
        for d in range(D):
            if d == 0:
                QM = np.hstack( (dataMat[q,:,d,:],np.ones((NTraj,1))) )
            else:
                QM = np.hstack( (QM,np.hstack( (dataMat[q,:,d,:],np.ones((NTraj,1))) )) )

        if q == 0:
            features_train = QM
        else:
            features_train = np.vstack( (features_train,QM) )

    # Feature matrix in original basis
    X = np.real( features_train ) 
    
    # Populate target matrix
    Y = np.zeros((C*NTraj,C))
    for q in range(C):
        cV = np.zeros(C)
        cV[q] = 1.0
        for m in range(NTraj):
            Y[q*NTraj+m,:] = cV
         
        
###################### Extraction of training and testing sets #####################

                  
    # Initialize extraction vectors
    eTr = np.zeros(np.shape(X)[0],dtype=int)
#     eTe = np.zeros(np.shape(X)[0],dtype=int)
            
    # Extraction vectors
    for q in range(C):
        eTr[q*NTraj:q*NTraj+NTrain] = int(1)
#     for q in range(C):
#         eTe[q*NTraj+NTrain:q*NTraj+NTrain+NTest] = int(1)

    # Extract training and testing datasets
    XTr = X[eTr>0,:]
    YTr = Y[eTr>0,:]

#     XTe = X[eTe>0,:]
#     YTe = Y[eTe>0,:]
            
            
############################# Solve optimization problem ############################

            
    # Learned weights by training
    if rP == 0:
        WOb = np.linalg.pinv(XTr)@YTr
    else:
        WOb = ( np.linalg.inv( (XTr.T)@XTr + rP*np.eye(np.shape((XTr.T)@XTr)[0]) )@(XTr.T) )@YTr
            
                
    return WOb



# Function to return semi-analytic matched filters in the Gaussian white noise limit
# for the classification of an arbitrary number of states
def calcMultiStateMF(dataMat):
    
    # Extract data properties from data matrix
    C     = np.shape(dataMat)[0]
    NTraj = np.shape(dataMat)[1]
    D     = np.shape(dataMat)[2]
    NT    = np.shape(dataMat)[3]
    
    
    ################################# Statistics of measurement data ################################
    
    
    # Calculate mean trajectories for all variables
    meanVec = np.zeros((C,NT,D))
    for q in range(C):
        for d in range(D):
            meanVec[q,:,d:(d+1)] = np.mean(dataMat[q,:,d:(d+1),:],axis=0).T
    
    # Assume time-independent variances for each quadrature: calculate variances using first time point; first data variable
    Xi2 = 0
    for q in range(C):
        Xi2 = Xi2 + np.var(dataMat[q,:,0,0])
        
        
    ################################# Computation of matrices ################################
    
    
    # Calculate state overlap matrix
    M = np.zeros((C,C))
    for q1 in range(C):
        for q2 in range(C):
            # Loop over distinct measured features
            for jj in range(D):
                # Calculate overlap
                Ojj = 1 + (meanVec[q1,:,jj:(jj+1)].T)@(meanVec[q2,:,jj:(jj+1)])
                
                M[q1,q2] = M[q1,q2] + Ojj
                
            # Add variance contribution if q1 == q2
            if q1 == q2:
                M[q1,q2] = M[q1,q2] + Xi2
                
    
    # Generate list of C-1 pairs of states
    pL = []
    for p in range(C-1):
        pL.append([p,p+1])
    
    # Calculate system matrix
    S = np.zeros( (len(pL),len(pL)) )
    for p in range(len(pL)):
        for q2 in range(C-1):
            S[p,q2] = M[pL[p][0],q2]-M[pL[p][1],q2] - ( M[pL[p][0],-1]-M[pL[p][1],-1] )
            
    # Calculate inverse of system matrix
    IS = la.inv(S)
    
    # Calculate diagonal bias matrix
    V = np.zeros( (len(pL),len(pL)) )
    for p in range(len(pL)):
        V[p,p] = (1/D)*(M[pL[p][0],-1]-M[pL[p][1],-1])
                
    # Calculate pairwise filters 
    Sv = np.zeros( (D*(NT+1),len(pL)) )
    for p in range(len(pL)):
        for d in range(D):
            # Filter term
            Sv[d*(NT+1):(d+1)*(NT+1)-1,p:(p+1)] = meanVec[pL[p][0],:,d:(d+1)]-meanVec[pL[p][1],:,d:(d+1)]
            
    # Calculate inhomogeneous vector
    n = np.zeros( (D*(NT+1),1) )
    for d in range(D):
        n[(d+1)*(NT+1)-1,0] = 1
                     
           
    ################################# Learning optimal filters ################################
                  
        
    # Construct C learned optimal filters and bias weights
    Fv = np.zeros( (D*(NT+1),C) )
    bv = np.zeros( (D*(NT+1),C) )
    for q in range(C):
        # For independent filters and biases
        if q < C-1:
            for p in range(len(pL)):
                Fv[:,q] = Fv[:,q] + IS[q,p]*Sv[:,p]
                bv[:,q] = bv[:,q] - IS[q,p]*V[p,p]*n[:,0]
        else:
            # Final filter and bias weight using completeness relation
            fSum = np.zeros( (D*(NT+1),1) )
            bSum = np.zeros( (D*(NT+1),1) )
            for qp in range(C-1):
                fSum[:,0] = fSum[:,0] + Fv[:,qp]
                bSum[:,0] = bSum[:,0] + bv[:,qp]
            Fv[:,q] = - fSum[:,0]
            bv[:,q] = (1/D)*n[:,0] - bSum[:,0]
                         
    ################################# Trained weight matrix ################################    
            
        
    # Calculate trained matrix
    WO = np.zeros( (D*(NT+1),C) )
    for q in range(C):
        # Assign learned filters and biases
        WO[:,q] = Fv[:,q] + bv[:,q]
        
   
    ################################# Additional filter forms ################################   

    
    # Individual filters and their coefficients
    filterMat   = np.zeros((C,NT,D))
    filterCoeff = np.zeros((C,C))
    for q in range(C):
        for d in range(D):
            filterMat[q,:,d] = Fv[d*(NT+1):(d+1)*(NT+1)-1,q].T
            
        # Store filter coefficients
        if q < C-1:
            for p in range(C):           
                if p == 0:
                    filterCoeff[q,p] = +IS[q,p]
                elif p == C-1:
                    filterCoeff[q,p] = -IS[q,p-1]
                else:
                    filterCoeff[q,p] = +IS[q,p]-IS[q,p-1]
                    
        else:
            # Final filter coefficients using dependence constraint
            filterCoeff[q,:] = -1*( np.sum(filterCoeff[0:q,:],axis=0) )
                
    # Individual biases    
    
        
    return WO, meanVec, filterMat, filterCoeff


# Function to test trained weights for the classification of an arbitrary number of states
def testMultiStateWO(WOb, dataMat, NTrain, NTest):
    
    # Extract data properties from data matrix
    C     = np.shape(dataMat)[0]
    NTraj = np.shape(dataMat)[1]
    D     = np.shape(dataMat)[2]
    NT    = np.shape(dataMat)[3]
    
    
################################# Compiled Matrices ################################
    
    # Construct matrix of readout features
    for q in range(C):
        for d in range(D):
            if d == 0:
                QM = np.hstack( (dataMat[q,:,d,:],np.ones((NTraj,1))) )
            else:
                QM = np.hstack( (QM,np.hstack( (dataMat[q,:,d,:],np.ones((NTraj,1))) )) )

        if q == 0:
            features_train = QM
        else:
            features_train = np.vstack( (features_train,QM) )

    # Feature matrix in original basis
    X = np.real( features_train ) 
         
        
###################### Extraction of training and testing sets #####################

                  
    # Initialize extraction vectors
    eTr = np.zeros(np.shape(X)[0],dtype=int)
    eTe = np.zeros(np.shape(X)[0],dtype=int)
            
    # Extraction vectors
    for q in range(C):
        eTr[q*NTraj:q*NTraj+NTrain] = int(1)
    for q in range(C):
        eTe[q*NTraj+NTrain:q*NTraj+NTrain+NTest] = int(1)

    # Extract training and testing datasets
    XTr = X[eTr>0,:]
#     YTr = Y[eTr>0,:]

    XTe = X[eTe>0,:]
#     YTe = Y[eTe>0,:]
            
            
############################# Classification accuracy ############################
           
            
    # Calculate classification accuracy for training data
    classVecTR  = np.zeros( (C) )
    classVecTE  = np.zeros( (C) )

    for q in range(C):
        for m in range(NTrain):
            testY = np.real(XTr[q*NTrain+m,:]@WOb)

            # Classify
            classVecTR[q] = classVecTR[q] + ( np.array([(np.argmax(np.abs(testY[:]))-q)==0]) ).astype(int)[0]

    # Average over records
    classVecTR = classVecTR/(NTrain)

    # Calculate classification accuracy for testing data
    for q in range(C):
        for m in range(NTest):
            testY = np.real(XTe[q*NTest+m,:]@WOb)

            # Classify
            classVecTE[q] = classVecTE[q] + ( np.array([(np.argmax(np.abs(testY[:]))-q)==0]) ).astype(int)[0]

    # Average over records
    classVecTE = classVecTE/(NTest)

    # Matched filter training and testing accuracy
#     mfMatTR, mfMatTE, filterMat = classifyMF(dataMatQ[:,:,:,0:2,:], NTrain, trainI)

    # Store RC CA results
#     classVecTRP[nv,p] = np.mean(classVecTR)
#     classVecTEP[nv,p] = np.mean(classVecTE)
#     mfTRP[nv,p] = mfMatTR
#     mfTEP[nv,p] = mfMatTE
        
                
    return np.mean(classVecTR), np.mean(classVecTE)



# Function to test trained weights for the classification of an arbitrary number of states
def testMultiStateRC(dataMat, NTrain, NTest, rctype='gen'):
    
    # Extract data properties from data matrix
    C     = np.shape(dataMat)[0]
    NTraj = np.shape(dataMat)[1]
    D     = np.shape(dataMat)[2]
    NT    = np.shape(dataMat)[3]
    
    # Determine RC weights
    assert rctype == 'gen' or rctype == 'wn', f"rctype must be 'gen' or 'wn', got '{rctype}' instead!"
    if rctype == 'gen': # General case
        WOb = calcMultiStateWO(dataMat, NTrain, rP=0.0)
    if rctype == 'wn': # White noise case
        WOb, _, _, _ = calcMultiStateMF(dataMat)
    
    
################################# Compiled Matrices ################################
    
    # Construct matrix of readout features
    for q in range(C):
        for d in range(D):
            if d == 0:
                QM = np.hstack( (dataMat[q,:,d,:],np.ones((NTraj,1))) )
            else:
                QM = np.hstack( (QM,np.hstack( (dataMat[q,:,d,:],np.ones((NTraj,1))) )) )

        if q == 0:
            features_train = QM
        else:
            features_train = np.vstack( (features_train,QM) )

    # Feature matrix in original basis
    X = np.real( features_train ) 
         
        
###################### Extraction of training and testing sets #####################

                  
    # Initialize extraction vectors
    eTr = np.zeros(np.shape(X)[0],dtype=int)
    eTe = np.zeros(np.shape(X)[0],dtype=int)
            
    # Extraction vectors
    for q in range(C):
        eTr[q*NTraj:q*NTraj+NTrain] = int(1)
    for q in range(C):
        eTe[q*NTraj+NTrain:q*NTraj+NTrain+NTest] = int(1)

    # Extract training and testing datasets
    XTr = X[eTr>0,:]
#     YTr = Y[eTr>0,:]

    XTe = X[eTe>0,:]
#     YTe = Y[eTe>0,:]
            
            
############################# Classification accuracy ############################
           
            
    # Calculate classification accuracy for training data
    classVecTR  = np.zeros( (C) )
    classVecTE  = np.zeros( (C) )

    for q in range(C):
        for m in range(NTrain):
            testY = np.real(XTr[q*NTrain+m,:]@WOb)

            # Classify
            classVecTR[q] = classVecTR[q] + ( np.array([(np.argmax(np.abs(testY[:]))-q)==0]) ).astype(int)[0]

    # Average over records
    classVecTR = classVecTR/(NTrain)

    # Calculate classification accuracy for testing data
    for q in range(C):
        for m in range(NTest):
            testY = np.real(XTe[q*NTest+m,:]@WOb)

            # Classify
            classVecTE[q] = classVecTE[q] + ( np.array([(np.argmax(np.abs(testY[:]))-q)==0]) ).astype(int)[0]

    # Average over records
    classVecTE = classVecTE/(NTest)
                
    return np.mean(classVecTR), np.mean(classVecTE)



#####################################################################################
#################################### Visualization ##################################
#####################################################################################

# Function to compute RC weights as filters and visualize them
def visFilters(dataMat, NTrain, NTest, dt):
    
    # Extract data properties from data matrix
    C     = np.shape(dataMat)[0]
    NTraj = np.shape(dataMat)[1]
    M     = np.shape(dataMat)[2]
    NT    = np.shape(dataMat)[3]
    
    # Determine RC weights for both general case and white noise case
    # General case
    WOb = calcMultiStateWO(dataMat, NTrain, rP=0.0)
    # White noise case
    WOA, _, _, _ = calcMultiStateMF(dataMat)
    
    # Time vector using sampling time
    dt = 0.01
    T = np.arange(0,NT*dt,dt)

    fig, axs = plt.subplots(M, C, figsize=(4*C,4*M))

    # White noise filters
    for m in range(M):
        for k in range(C):
            ax = axs[m,k]
            ax.plot(T, WOA[m*(NT+1):(m+1)*(NT+1)-1,k]/la.norm(WOA[m*(NT+1):(m+1)*(NT+1)-1,k]), 'k')
            ax.plot(T, WOb[m*(NT+1):(m+1)*(NT+1)-1,k]/la.norm(WOb[m*(NT+1):(m+1)*(NT+1)-1,k]), 'gray', zorder=-3)
            ax.set_title('m = ' + str(m) + ', k = ' + str(k))
            
    plt.show()
    
    return
