#!/usr/bin/env python
# coding: utf-8

# In[17]:


#from pycaret.datasets import get_data

import pandas as pd
dataset = pd.read_csv("loandata.csv")
dataset = dataset.drop(["Loan_ID"], axis =1)

data = dataset.sample(frac=0.75, random_state=786).reset_index(drop=True)
data_unseen = dataset.drop(data.index).reset_index(drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))


# In[18]:


from pycaret.classification import *
exp_clf101 = setup(data = data, target = 'Loan_Status', session_id=123) 


# In[19]:


compare_models()


# In[23]:


lgbm = create_model("lightgbm")


# In[21]:


xgb = create_model("xgboost")


# In[27]:


tune_lgbm = tune_model(lgbm)


# In[28]:


tune_xgb = tune_model(xgb)


# In[30]:


plot_model(tune_lgbm, plot="auc")


# In[31]:


plot_model(tune_xgb, plot="auc")


# In[32]:


plot_model(tune_lgbm, plot = 'pr')


# In[33]:


plot_model(tune_xgb, plot = 'pr')


# In[34]:


plot_model(tune_xgb, plot='feature')


# In[35]:


plot_model(tune_xgb, plot = 'confusion_matrix')


# In[36]:


evaluate_model(tune_xgb)


# In[37]:


predict_model(tune_xgb)


# In[39]:


final_xgb = finalize_model(tune_xgb)


# In[40]:


print(final_xgb)


# In[41]:


predict_model(final_xgb)


# In[42]:




unseen_predictions = predict_model(final_xgb, data=data_unseen)
unseen_predictions.head()


# In[44]:


save_model(final_xgb,'Final XGB Model A112020')


# In[ ]:




