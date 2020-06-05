#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


orders = pd.read_csv("D:\Market Basket Analysis\instacart-market-basket-analysis\orders.csv\orders.csv")


# In[ ]:


orders.head()


# In[ ]:


orders.shape


# In[ ]:


print('there are', len(orders[orders.eval_set == 'prior']), 'entries for prior')
print('there are', len(orders[orders.eval_set == 'train']), 'entries for train')
print('there are', len(orders[orders.eval_set == 'test']), 'entries for test')


# In[ ]:


print('there are', len(orders[orders.eval_set == 'prior'].user_id.unique()), 'unique customers in total')
print('there are', len(orders[orders.eval_set == 'train'].user_id.unique()), 'unique customers in train set')
print('there are', len(orders[orders.eval_set == 'test'].user_id.unique()), 'unique customers in test set')


# In[ ]:


orders_amount_for_customer = orders.groupby('user_id')['order_number'].count().value_counts()

plt.figure(figsize=(20,8))
sns.barplot(x=orders_amount_for_customer.index, y=orders_amount_for_customer.values, color='mediumseagreen')
plt.title('Amount of Orders Distribution', fontsize=16)
plt.ylabel('Number of Customers', fontsize=16)
plt.xlabel('Amount of Orders', fontsize=16)
plt.xticks(rotation='vertical');


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x=orders.order_dow, color='mediumseagreen')

plt.title("Order Amounts by Days", fontsize=16)
plt.xlabel('', fontsize=16)
plt.xticks(fontsize=15)
plt.ylabel('Order Counts', fontsize=16)
plt.yticks(fontsize=15)
("")


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x=orders.order_hour_of_day, color='mediumseagreen')

plt.title("Order Amounts by Hours", fontsize=16)
plt.xlabel('Hour of Day', fontsize=16)
plt.xticks(fontsize=15)
plt.ylabel('Order Counts', fontsize=16)
plt.yticks(fontsize=15)
("")


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x=orders.days_since_prior_order, color= 'mediumseagreen')

#plt.title("Number of Orders per Days Since Last Purchase", fontsize=16)
plt.xlabel('Days Since Last Purchase', fontsize=16)
plt.xticks(np.arange(31), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                          24, 25, 26, 27, 28, 29, 30],  fontsize=15)
plt.xticks(rotation='vertical')
plt.ylabel('Order Counts', fontsize=16)
plt.yticks(fontsize=15)
("")


# In[ ]:


order_products_prior = pd.read_csv("D:\Market Basket Analysis\instacart-market-basket-analysis\order_products__prior.csv\order_products__prior.csv")
order_products_prior.head()


# In[ ]:


order_products_train = pd.read_csv("D:\Market Basket Analysis\instacart-market-basket-analysis\order_products__train.csv\order_products__train.csv")
order_products_train.head()


# In[ ]:


# Concatenation of both tables.
order_products_total = pd.concat([order_products_prior, order_products_train]) 
# Create amounts of item per order.
frequency_per_number_of_order = order_products_total.groupby('order_id')['product_id'].count().value_counts()

plt.figure(figsize=(20,8))
sns.barplot(x=frequency_per_number_of_order.index, y=frequency_per_number_of_order.values, color='mediumseagreen')
plt.title('Amount of Items Per Order', fontsize=16)
plt.ylabel('Order Counts', fontsize=16)
plt.xlabel('Number of Items', fontsize=16)
plt.xticks(rotation='vertical');


# In[ ]:


print('there are', order_products_total.shape[0], 'grocery products ordered')
print('there are', len(order_products_total.order_id.unique()), 'order transactions')
print('there are', len(order_products_total.product_id.unique()), 'unique products')


# In[ ]:


# Find out how many products have been reordered before.
print(len(order_products_total[order_products_total.reordered == 1]), 'products have reordered before')
print(len(order_products_total[order_products_total.reordered == 0]), 'products haven\'t reordered before')
# Find out the ratio.
print(len(order_products_total[order_products_total.reordered == 1])/order_products_total.shape[0], 'have reordered before')
print(len(order_products_total[order_products_total.reordered == 0])/order_products_total.shape[0], 'haven\'t reordered before')


# In[ ]:


aisles = pd.read_csv("D:\Market Basket Analysis\instacart-market-basket-analysis/aisles.csv/aisles.csv")
aisles.head()


# In[ ]:


departments = pd.read_csv("D:\Market Basket Analysis\instacart-market-basket-analysis\departments.csv\departments.csv")
departments.head()


# In[ ]:


products = pd.read_csv("D:\Market Basket Analysis\instacart-market-basket-analysis\products.csv\products.csv")
products.head()


# In[ ]:


# Merging tables together.
products_departments = products.merge(departments, left_on='department_id', right_on='department_id', how='left')
products_departments.head()


# In[ ]:


products_departments_aisles = products_departments.merge(aisles, left_on='aisle_id', right_on='aisle_id', how='left')
products_departments_aisles.head()


# In[ ]:


# Counting how many items is in each product category. 
products_departments_aisles.groupby('department')['product_name'].count().reset_index().sort_values(by='product_name', ascending=False).head(10)


# In[ ]:


# Merging products_departments_aisles and order_products_total.
df = order_products_total.merge(products_departments_aisles, left_on='product_id', right_on='product_id', how='left')
df.head()


# In[ ]:


# Find out the top 15 items people purchased the most.
top_15_products = df.product_name.value_counts(ascending=False).reset_index().head(15)
top_15_products.columns = ['product_name', 'count']
top_15_products


# In[ ]:


# Finding top 15 aisles. 
top_15_aisles = df.aisle.value_counts(ascending=False).reset_index().head(15)
top_15_aisles.columns = ['aisle_name', 'count']
top_15_aisles


# In[ ]:


# Finding top 15 departments.
top_15_department = df.department.value_counts(ascending=False).reset_index().head(15)
top_15_department.columns = ['department_name', 'count']
top_15_department


# In[ ]:


# Find out reorder ratio per department.
reorder_ratio_per_dep = df.groupby('department')['reordered'].mean().reset_index()
reorder_ratio_per_dep.columns = ['department', 'reorder_ratio']
reorder_ratio_per_dep.sort_values(by='reorder_ratio', ascending=False)


# In[ ]:


# Find out reorder ration per aisle.
reorder_ratio_per_aisle = df.groupby('aisle')['reordered'].mean().reset_index()
reorder_ratio_per_aisle.columns = ['aisle', 'reorder_ratio']
reorder_ratio_per_aisle.sort_values(by='reorder_ratio', ascending=False)


# In[ ]:


# get the list of orders that have been reordered before
reorders = order_products_total[order_products_total['reordered'] == 1]
orders2 = orders[['order_id', 'user_id']]


# In[ ]:


# merge to get user_id and product_id
user_orders = reorders.merge(orders2, on='order_id')


# In[ ]:


# filtering out the high volumn products that user reordered more than once
user_orders['high_volume'] = (user_orders['product_id'].value_counts().sort_values(ascending=False)>1)
high_volume = user_orders[user_orders['high_volume'] == True]


# In[ ]:


# get a matrix of different high volume items that particular user purchased
high_volume_users = high_volume.groupby(['user_id','product_id']).size().sort_values(ascending=False).unstack().fillna(0)


# In[ ]:


# calculate similarity between each user
cosine_dists = pd.DataFrame(cosine_similarity(high_volume_users),index=high_volume_users.index, columns=high_volume_users.index)
cosine_dists.head()


# In[ ]:


def Recommender_System(user_id):
    
    '''
    enter user_id and return a list of 5 recommendations.
    '''
    
    u = high_volume.groupby(['user_id','product_id']).size().sort_values(ascending=False).unstack().fillna(0)
    u_sim = pd.DataFrame(cosine_similarity(u), index=u.index, columns=u.index)

    p = high_volume.groupby(['product_id','user_id']).size().sort_values(ascending=False).unstack().fillna(0)
    
    recommendations = pd.Series(np.dot(p.values,cosine_dists[user_id]), index=p.index)
    return recommendations.sort_values(ascending=False).head(10)


# In[ ]:


# recommendation for customer id 175965.
Recommender_System(175965)


# In[ ]:


# filter 1000 users for calculation
#  because the dataframe is too large 
users = high_volume.user_id.unique().tolist()
# calculate recall for the :1000 users
def how_match():
    res = []
    for user in sorted(users)[:1000]:
        recommendations = Recommender_System(user)
        top_20_itmes = _[_.user_id == user].product_name.value_counts().head(20)
    
        recommendations_list = recommendations.index.tolist()
        top_20_items_list = top_20_itmes.index.tolist()
    
        res.append((len(set(recommendations_list) & set(top_20_items_list)))/5)
    return np.mean(res)
# get metric for the :1000 users
how_match()
# calculate the mean of all the metric from all the Recommender notebooks.
print('The Final Score for Metric is', (0.531 + 0.522 + 0.519 + 0.530 + 0.523 + 0.519 + 0.526)/7)

