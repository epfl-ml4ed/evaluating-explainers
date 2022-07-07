import numpy as np
def fillNaN(feature):
    shape = feature.shape
    feature_min = np.nanmin(feature.reshape(-1,shape[2]),axis=0)#min of that feature ober all weeks
    # feature_min = feature_min.reshape(shape[1],-1) 
    # feature_min = np.where(np.isnan(feature_min),np.concatenate((feature_min[1:,:],np.zeros((1,shape[2])))),feature_min)
    feature = feature.reshape(-1,shape[2])
    inds = np.where(np.isnan(feature))
    feature[inds] = np.take(feature_min.reshape(-1), inds[1])
    feature = feature.reshape(shape)
    return feature
def transform_x(x,num_feature_type,num_weeks,features_min,features_max):
    x = np.array(x)
    num_feature_type = np.array(num_feature_type)
    num_features = num_feature_type.sum()
    x = x.reshape((-1,num_weeks,num_features))
    shape = x.shape
    features_max = np.where(features_max==0,np.ones(features_max.shape),features_max)
    max_instance = 1.001*features_max
    feature_current = np.vstack([x,max_instance.reshape((1,)+max_instance.shape)])
    features_max = features_max.reshape(-1)
    feature_norm = (feature_current.reshape(shape[0]+1,-1)-features_min)/(1.001*features_max-features_min)
    # feature_current = feature_norm.reshape(-1,feature_current.shape[1],feature_current.shape[2] )
    x = feature_norm[:feature_norm.shape[0]-1,:]
    return x