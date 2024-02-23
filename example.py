#estimate example of BPR embeddings using clothing data.
import sys
from bpr import *
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import keras.backend as K
np.random.seed(9)

###############
#clean clickstream data to create inequality matrix

#from https://archive.ics.uci.edu/dataset/553/clickstream+data+for+online+shopping
click_df = pd.read_csv('external/e-shop clothing 2008.csv',sep=';')
#click_df = click_df[click_df['session ID'] < 1000]
product_dict  = {i:j for j,i in enumerate(sorted(set(click_df['page 2 (clothing model)'])))}
consumer_dict = {i:j for j,i in enumerate(sorted(set(click_df['session ID'])))}
click_df['product_id'] = click_df['page 2 (clothing model)'].apply(lambda x: product_dict[x])
click_df['consumer_id'] = click_df['session ID'].apply(lambda x: consumer_dict[x])
click_df['price'] = click_df['price']
click_df['cat'] = click_df['page 1 (main category)']
click_df['1'] = 1 #for many-to-many merge in constructing inequalities

num_consumers = click_df['consumer_id'].max()+1
num_products = click_df['product_id'].max()+1

# create RHS df for inequalities
neg_df = click_df.groupby('product_id')['price'].mean().reset_index().rename({'product_id':'neg_product_id','price':'neg_price'},axis=1)
neg_df['1'] = 1 #for many-to-many merge

# construct the inequality matrix.
sessions_per_batch = 100
ineq_df = []
print(click_df['consumer_id'].max())
for b in range(0,click_df['consumer_id'].max(),sessions_per_batch):
    triplet_df = click_df.loc[(click_df['consumer_id']>=b) & (click_df['consumer_id']<b+sessions_per_batch) ,['consumer_id','product_id','price','1']].copy()
    triplet_df.rename({'product_id':'pos_product_id','price':'pos_price'},axis=1,inplace=True)
    #construct triplet df
    pos_hotels = {i:j for i,j in triplet_df.groupby('consumer_id')['pos_product_id'].apply(lambda x: set(x)).items()}
    triplet_df['pos_hotel_set'] = triplet_df['consumer_id'].apply(lambda x: pos_hotels[x])
    #flag one product to be included in validation_set
    triplet_df['val'] = triplet_df.groupby('consumer_id')['pos_product_id'].transform(lambda x: pd.Series.sample(x,n=1).values[0])
    triplet_df['val'] = triplet_df['val']==triplet_df['pos_product_id']
    triplet_df = triplet_df.merge(neg_df,on='1',how='inner')
    #throw out inequalities with searched products on LHS
    triplet_df = triplet_df[(triplet_df['pos_hotel_set'] & triplet_df['neg_product_id'].apply(lambda x: {x}))==False]
    triplet_df['d_price'] = triplet_df['pos_price'] - triplet_df['neg_price']
    triplet_df = triplet_df.drop(['pos_price','neg_price','pos_hotel_set','1'],axis=1)
    ineq_df.append(triplet_df)
    sys.stdout.write('\r %d' % b)
    sys.stdout.flush()
ineq_df = pd.concat(ineq_df)

# split sample and drop 1-search individuals from validation DF
train_df = ineq_df.loc[(ineq_df['val']==False),:]
train_users = set(ineq_df['consumer_id'])
val_df = ineq_df.loc[(ineq_df['val']==True) & ineq_df['consumer_id'].apply(lambda x: x in train_users),:]

# create stdized version of observables
observables = click_df.groupby('product_id')[['cat','colour']].first()
for c in observables.columns:
    dummies = pd.get_dummies(observables[c],drop_first=True)
    for d in dummies.columns:
        observables['%s_%s' % (c.replace(' ','_'),str(d))] = dummies[d]
    observables.drop(c,axis=1,inplace=True)
observables_stdized = observables/observables.std()

#set batch size
batch_size=10000
#define number of steps to take for each sampler, given batch size
trainsteps = np.ceil(len(train_df)/float(batch_size))*1
valsteps = np.ceil(len(val_df)/float(batch_size))*1
fullsteps = np.ceil(len(ineq_df)/float(batch_size))*1

#initialize the samplers
sampler = lambda df: bpr_triplet_impression_sampler(df,'consumer_id','pos_product_id','neg_product_id',
        'd_price',batch_size=batch_size,shuffle=True)
train_gen = sampler(train_df)
val_gen = sampler(val_df)
full_gen = sampler(ineq_df)

#flag to stop if does not improve in 3 iterations
earlyend = EarlyStopping(patience=3,monitor='val_loss',mode='min')

##################################
####estimate model without observables (but includes price)
dim = 5  # number of latent parameters (L in paper)
pen = 1e0  # penalty term (lambda_theta in paper)
pen_normed = pen/(num_consumers*(dim+1) + num_products*(dim+1))

# create model of unobservables
K.clear_session()
#tf.set_random_seed(9)
lat_model = build_bpr_model(num_items = num_products,num_users = num_consumers,
            k=dim, pen=pen_normed,  usebias=True)
print(lat_model.summary())
lat_model.compile('adam',loss='binary_crossentropy')
#train to determine num iterations
fitobj = lat_model.fit_generator(train_gen,epochs=1000,steps_per_epoch=trainsteps,verbose=1,callbacks=[earlyend],
            validation_data=val_gen,validation_steps=valsteps,use_multiprocessing=False,workers=1,max_queue_size=10)

#now run it on the full dataset
num_opt_epochs = np.argmin(fitobj.history['val_loss'])+1
K.clear_session()
lat_model = build_bpr_model(num_items = num_products,num_users = num_consumers,
    k=dim, pen=pen_normed,  usebias=True)
lat_model.compile('adam',loss='binary_crossentropy')
fitobj = lat_model.fit_generator(full_gen,epochs=num_opt_epochs,steps_per_epoch=fullsteps,verbose=2,use_multiprocessing=False,
    workers=1,max_queue_size=10)

# export learned embeddings
(user_lat_embeddings,clothing_lat_embeddings) = export_embeddings(lat_model,product_dict,consumer_dict,observables=None,usebias=True)
clothing_lat_embeddings.to_csv("external/clothing_latent_embeddings_5d.tsv" ,sep='\t')
user_lat_embeddings.to_csv("external/consumer_latent_embeddings_5d.tsv" ,sep='\t')


##################################
####estimate model with only observables
pen = 1e-1 #penalty term (lambda_theta in text)
pen_normed = pen/(num_consumers*(observables.shape[1]+1))
K.clear_session()
np.random.seed(9)
#create model
obs_model = build_bpr_model(num_items = num_products,num_users = num_consumers,
            k=0, pen=pen_normed,  usebias=True,X=observables_stdized)
print(obs_model.summary())
obs_model.compile('adam',loss='binary_crossentropy')
#train to determine num iterations
fitobj = obs_model.fit_generator(train_gen,epochs=1000,steps_per_epoch=trainsteps,verbose=1,callbacks=[earlyend],
            validation_data=val_gen,validation_steps=valsteps,use_multiprocessing=False,workers=1,max_queue_size=10)

#now run it on the full dataset
K.clear_session()
num_opt_epochs = np.argmin(fitobj.history['val_loss'])+1
obs_model = build_bpr_model(num_items = num_products,num_users = num_consumers,
                            k=0, pen=pen_normed, usebias=True, X=observables_stdized)
obs_model.compile('adam',loss='binary_crossentropy')
fitobj = obs_model.fit_generator(full_gen,epochs=num_opt_epochs,steps_per_epoch=fullsteps,verbose=2,use_multiprocessing=False,
    workers=1,max_queue_size=10)

# export learned embeddings
(user_obs_embeddings,clothing_obs_embeddings) = export_embeddings(obs_model,product_dict,consumer_dict,observables=observables,usebias=True)
user_obs_embeddings.to_csv("external/consumer_obs_embeddings.tsv" ,sep='\t')
#correlation structure of preferences
print(user_obs_embeddings.corr())