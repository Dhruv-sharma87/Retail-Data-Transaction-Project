#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


trxn= pd.read_csv('Retail_Data_Transactions.csv')


# In[3]:


trxn


# In[4]:


response= pd.read_csv('Retail_Data_Response.csv')


# In[5]:


response


# In[6]:


df= trxn.merge(response, on='customer_id', how='left')
df


# In[7]:


df.dtypes
df.shape
df.tail()


# In[8]:


df.describe()


# In[9]:


df.isnull().sum()


# In[10]:


df= df.dropna()
df


# In[11]:


df['trans_date'] = pd.to_datetime(df['trans_date'])
df['response']= df['response'].astype('int64')


# In[12]:


df


# In[13]:


set(df['response'])


# In[14]:


df.dtypes


# In[15]:


from scipy import stats
import numpy as np

#calculate z_score
z_scores= np.abs(stats.zscore(df['tran_amount']))

#set a threshols
threshold = 3 

outliers= z_scores>threshold 

print(df[outliers]) 


# In[16]:


from scipy import stats
import numpy as np

#calculate z_score
z_scores= np.abs(stats.zscore(df['response']))

#set a threshols
threshold = 3 

outliers= z_scores>threshold 

print(df[outliers]) 


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df['tran_amount'])
plt.show()


# In[18]:


df['month']= df['trans_date'].dt.month


# In[19]:


df


# In[20]:


monthly_sales= df.groupby('month')['tran_amount'].sum()
monthly_sales= monthly_sales.sort_values(ascending=False).reset_index().head(3)


# In[21]:


monthly_sales


# In[22]:


customer_counts= df['customer_id'].value_counts().reset_index()


# In[23]:


customer_counts.columns= ['customer_id','counts']
customer_counts


# In[24]:


top_5_cus= customer_counts.sort_values(by='counts',ascending=False).head(5) 


# In[25]:


top_5_cus


# In[26]:


sns.barplot(x='customer_id',y='counts',data=top_5_cus)


# In[27]:


customer_sales= df.groupby('customer_id')['tran_amount'].sum().reset_index()
customer_sales


top_5_sales= customer_sales.sort_values(by='tran_amount',ascending=False).head(5) 
top_5_sales


# In[28]:


sns.barplot(x='customer_id',y='tran_amount',data=top_5_sales)


# In[29]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

df['month_year'] = df['trans_date'].dt.to_period('M')
monthly_sales = df.groupby('month_year')['tran_amount'].sum()

monthly_sales.index = monthly_sales.index.to_timestamp()

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index, monthly_sales.values)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Corrected typo

plt.xlabel('Month-Year')
plt.ylabel('Sales')
plt.title('Monthly Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[30]:


#recency
recency= df.groupby('customer_id')['trans_date'].max()

#frequency
frequency= df.groupby('customer_id')['trans_date'].count()

#monetary
monetary= df.groupby('customer_id')['tran_amount'].sum()

#combine

rfm= pd.DataFrame({'recency':recency, 'frequency':frequency, 'monetary':monetary})


# In[31]:


rfm


# In[32]:


def segment_customer(row):
    if row['recency'].year>=2012 and row['frequency']>=15 and row['monetary']>=1000:
        return 'P0'
    elif (2011<=row['recency'].year<2012) and (10<row['frequency']<15) and (500<=row['monetary']<=1000):
        return 'P1'
    else:
        return 'P2'
    
rfm['Segment']= rfm.apply(segment_customer, axis=1)


# In[33]:


rfm


# In[34]:


# count the number of churned and active counts 
churn_counts= df['response'].value_counts()

# plot
churn_counts.plot(kind='bar')


# In[35]:


top_5_cus = monetary.sort_values(ascending=False).head(5).index

top_5_customers_df = df[df['customer_id'].isin(top_5_cus)]  # Correct variable name

top_customer_sales = top_5_customers_df.groupby(['customer_id', 'month_year'])['tran_amount'].sum().unstack(level=0)

top_customer_sales.plot(kind='line')


# In[36]:


import seaborn as sns

plt.figure(figsize=(10, 5))
sns.histplot(df["tran_amount"], bins=30, kde=True, color="blue")
plt.title("Distribution of Transaction Amounts", fontsize=14)
plt.xlabel("Transaction Amount", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True)
plt.show()


# In[37]:


# Count response values
response_counts = df["response"].value_counts()

# Plot pie chart
plt.figure(figsize=(6, 6))
plt.pie(response_counts, labels=["No Response", "Response"], autopct="%1.1f%%", colors=["red", "green"])
plt.title("Customer Response Distribution", fontsize=14)
plt.show()


# In[38]:


plt.figure(figsize=(8, 5))
sns.boxplot(x=df["response"], y=df["tran_amount"])
plt.title("Transaction Amounts by Response", fontsize=14)
plt.xlabel("Response (0 = No, 1 = Yes)", fontsize=12)
plt.ylabel("Transaction Amount", fontsize=12)
plt.grid(True)
plt.show()


# In[39]:


plt.figure(figsize=(10, 5))
sns.scatterplot(x=df["month"], y=df["tran_amount"], alpha=0.5)
plt.title("Transaction Amount by Month", fontsize=14)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Transaction Amount", fontsize=12)
plt.grid(True)
plt.show()


# In[40]:


df = pd.read_csv("MainData.csv")

# Convert transaction date column to datetime format (if applicable)
df["trans_date"] = pd.to_datetime(df["trans_date"], format="%Y-%m-%d")

# Calculate summary statistics for the 'tran_amount' column
Q1 = df["tran_amount"].quantile(0.25)  # First quartile (25th percentile)
Q2 = df["tran_amount"].median()        # Median (50th percentile)
Q3 = df["tran_amount"].quantile(0.75)  # Third quartile (75th percentile)
IQR = Q3 - Q1                          # Interquartile Range
minimum = df["tran_amount"].min()      # Minimum value
maximum = df["tran_amount"].max()      # Maximum value
mean_value = df["tran_amount"].mean()  # Mean value

# Print the results
print(f"Q1 (25th percentile): {Q1}")
print(f"Median (Q2, 50th percentile): {Q2}")
print(f"Q3 (75th percentile): {Q3}")
print(f"IQR (Interquartile Range): {IQR}")
print(f"Minimum Transaction Amount: {minimum}")
print(f"Maximum Transaction Amount: {maximum}")
print(f"Mean Transaction Amount: {mean_value}")


# In[ ]:





# In[ ]:





# In[ ]:




