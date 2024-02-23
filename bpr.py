import numpy as np
from keras.layers import Input, Embedding, Dot, Activation, Add, Subtract,Flatten,Reshape
from keras.regularizers import l2
from keras.models import Model
import pandas as pd

#define sampler for keras
def bpr_triplet_impression_sampler(df,user_id,pos_id,neg_id,price_diff_var=None,batch_size=1000,shuffle=True):
    '''
    This functions samples from a matrix of revealed-preference implied inequalities and feeds it to a keras-readable format
    where inequalities are of the form U(pos_id) > U(neg_id)
    df: a pandas dataframe that where each row indicates an inequality
    user_id: an integer (from 0 to num users) identifying each consumer
    pos_id: an integer (from 0 to num products) identifying each preferred product
    neg_id: an integer (from 0 to num products) identifying each less preferred product
    price_diff: a variable that has the difference in the price of pos_id and neg_id. included in sampler to allow for time-varying differences in prices within product (otherwise, these may be included in X during model creation)
    batch_size: how many samples to feed keras for each training step
    shuffle: indicator for whether to permute the inequality matrix after each training iteration
    '''
    while True:
        if shuffle:
            idxs = np.random.permutation(len(df))
        else:
            idxs = np.arange(len(df))
        for i in  range(0,len(df),batch_size):
            batch = df.iloc[idxs[i:(i+batch_size)],:].copy()
            if price_diff_var is not None:
                yield [ [np.array(batch[user_id]),np.array(batch[pos_id]),np.array(batch[neg_id]),np.array(batch[price_diff_var])],
                    np.ones_like(batch[user_id])]
            else:
                yield [[np.array(batch[user_id]), np.array(batch[pos_id]), np.array(batch[neg_id])],
                 np.ones_like(batch[user_id])]

def build_bpr_model(num_items,num_users,X=None,k=5,usebias=False,pen=0.,include_price=True,price_pen=0.):
    '''
    Builds our keras model of BPR
    num_items: number of unique products/items
    num_users: number of unique users
    k: embedding dimensionality (number of latent characteristics for each product)
    pen: regularization parameter on user embeddings/preferences
    X: an optional array of item observables, with num_items rows.
    usebias: bool to indicate inclusion of an item bias term (e.g. an item fixed effect)
    price_pen: a penalty regularization parameter to be used only on the price coefficients of consumers, potentially due to the differing scale of these characteristics.
      '''
    if X is None:
        num_obs = 0
    else:
        num_obs = X.shape[1]
    user_input = Input((1,),name='user')
    pos_item_input = Input((1,),name='pos_item')
    neg_item_input = Input((1,),name='neg_item')

    all_scores = []
    if k>0:
        item_latent_vecs = Embedding(input_dim=num_items,output_dim = k,input_length=1,name='item_latent_embeddings',embeddings_regularizer = l2(pen))
        user_latent_vecs = Embedding(input_dim=num_users,output_dim = k,input_length=1,name='user_latent_embeddings',embeddings_regularizer = l2(pen))(user_input)
        pos_latent_vecs = item_latent_vecs(pos_item_input)
        neg_latent_vecs = item_latent_vecs(neg_item_input)
        latent_vec_diff = Subtract(name='latent_item_diff')([pos_latent_vecs,neg_latent_vecs])
        latent_y = Dot(2,name='latent_score')([user_latent_vecs,latent_vec_diff])
        all_scores.append(latent_y)
    if num_obs > 0 :
        item_obs_vecs = Embedding(input_dim=num_items,output_dim = num_obs,input_length=1,name='item_observables')
        user_obs_vecs = Embedding(input_dim=num_users,output_dim = num_obs,input_length=1,name='user_obs_embeddings',embeddings_regularizer = l2(pen))(user_input)
        pos_obs_vecs = item_obs_vecs(pos_item_input)
        neg_obs_vecs = item_obs_vecs(neg_item_input)
        obs_vec_diff = Subtract(name='obs_item_diff')([pos_obs_vecs,neg_obs_vecs])
        obs_y = Dot(2,name='obs_score')([user_obs_vecs,obs_vec_diff])
        all_scores.append(obs_y)
    if include_price:
        d_price = Input((1,), name='price_diff')
        user_price_vecs = Embedding(input_dim = num_users,output_dim=1,input_length=1,name='price_coef',embeddings_regularizer = l2(price_pen))(user_input) #don't regularize price coef due to differing scale.
        price_y = Dot(1,name='price_score')([user_price_vecs,d_price])
        all_scores.append(price_y)
    if len(all_scores)>1:
        y = Add(name='score')(all_scores)
    else:
        y = all_scores[0]
    if usebias:
        item_bias = Embedding(input_dim=num_items,output_dim = 1 ,input_length=1,name='item_bias',embeddings_regularizer = l2(pen))
        pos_bias = item_bias(pos_item_input)
        neg_bias = item_bias(neg_item_input)
        pos_bias = Reshape((1,),name='pos_item_bias')(pos_bias)
        neg_bias = Reshape((1,),name='neg_item_bias')(neg_bias)
        bias_diff = Subtract(name='bias_diff')([pos_bias,neg_bias])
        y = Add(name='score_w_bias')([y,bias_diff])
    y = Activation('sigmoid',name='prob')(y)
    if len(y.shape)>2:
        y = Flatten()(y)
    if include_price:
        mf = Model([user_input,pos_item_input,neg_item_input,d_price],y)
    else:
        mf = Model([user_input, pos_item_input, neg_item_input], y)
    if num_obs > 0:
        mf.get_layer('item_observables').trainable=False
        mf.get_layer('item_observables').set_weights([X])
    return mf
def export_embeddings(model,product_dict,user_dict,observables=None,usebias=True,include_price=True):
    '''
    Helper function to export embeddings from BPR model to dataframe
    model: a BPR model initialized by build_bpr_model
    product dict: a dict mapping product integers used in keras to original identifiers of products
    user dict: a dict mapping users/consumers integers used in keras to original identifiers of users
    observables: an optional dataframe passing the original observables dataframe (if the model used observables). Column names should correspond to those used in estimation
    usebias: indicator for whether or not a bias term is included in keras model.
    include_price: indicator for whether or not prices were fed from sampler (otherwise, prices should be included in the observables DF)
    '''
    has_latent_chars = 'user_latent_embeddings' in [l.name for l in model.layers]
    has_observables = 'user_obs_embeddings' in [l.name for l in model.layers]
    #create original IDs.
    reverse_product_dict = {j: i for i, j in product_dict.items()}
    reverse_user_dict = {j: i for i, j in user_dict.items()}
    users = [reverse_user_dict[i] for i in range(len(user_dict))]
    products = [reverse_product_dict[i] for i in range(len(product_dict))]
    # extract user embeddings
    if has_latent_chars:
        product_lat_embeddings = model.get_layer('item_latent_embeddings').get_weights()[0]
        dim = product_lat_embeddings.shape[1]
        lat_colnames = ['attr_%g' % k for k in range(dim)]
        product_lat_embeddings = pd.DataFrame(product_lat_embeddings, index=products, columns=lat_colnames)
        user_lat_embeddings = model.get_layer('user_latent_embeddings').get_weights()[0]
        user_lat_embeddings = pd.DataFrame(user_lat_embeddings, index=users, columns=lat_colnames)
    if has_observables:
        product_obs_embeddings = pd.DataFrame(observables.values, index=products,columns=observables.columns)
        print(product_obs_embeddings)
        user_obs_embeddings = model.get_layer('user_obs_embeddings').get_weights()[0]
        user_obs_embeddings = pd.DataFrame(user_obs_embeddings, index=users, columns=observables.columns)
        # unstandardize observable embeddings so works with the magnitude of original observables
        user_obs_embeddings /= observables.std()
    if has_observables and has_latent_chars:
        product_embeddings = product_obs_embeddings.join(product_lat_embeddings)
        user_embeddings = user_lat_embeddings.join(user_obs_embeddings)
    elif has_observables:
        product_embeddings = product_obs_embeddings
        user_embeddings = user_obs_embeddings
    elif has_latent_chars:
        product_embeddings = product_lat_embeddings
        user_embeddings = user_lat_embeddings
    #extract price preferences
    if include_price:
        price_embeddings = model.get_layer('price_coef').get_weights()[0]
        user_embeddings['price'] = price_embeddings
    #extract bias (product fixed effect)
    if usebias:
        product_embeddings['bias'] = model.get_layer('item_bias').get_weights()[0]
    return(user_embeddings,product_embeddings)

