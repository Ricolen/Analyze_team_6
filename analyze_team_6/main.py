# Import Libraries
import pandas as pd
import numpy as np

# Data Loading and Preprocessing
ebp_url = 'https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/electrification_by_province.csv'
ebp_df = pd.read_csv(ebp_url)

for col, row in ebp_df.iloc[:,1:].iteritems():
    ebp_df[col] = ebp_df[col].str.replace(',','').astype(int)

ebp_df.head()


# Twitter Data

twitter_url = 'https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/twitter_nov_2019.csv'
twitter_df = pd.read_csv(twitter_url)
twitter_df.head()


# Important Variables

# gauteng ebp data as a list
gauteng = ebp_df['Gauteng'].astype(float).to_list()

# dates for twitter tweets
dates = twitter_df['Date'].to_list()

# dictionary mapping official municipality twitter handles to the municipality name
mun_dict = {
    '@CityofCTAlerts' : 'Cape Town',
    '@CityPowerJhb' : 'Johannesburg',
    '@eThekwiniM' : 'eThekwini' ,
    '@EMMInfo' : 'Ekurhuleni',
    '@centlecutility' : 'Mangaung',
    '@NMBmunicipality' : 'Nelson Mandela Bay',
    '@CityTshwane' : 'Tshwane'
}

# dictionary of english stopwords
stop_words_dict = {
    'stopwords':[
        'where', 'done', 'if', 'before', 'll', 'very', 'keep', 'something', 'nothing', 'thereupon', 
        'may', 'why', 'â€™s', 'therefore', 'you', 'with', 'towards', 'make', 'really', 'few', 'former', 
        'during', 'mine', 'do', 'would', 'of', 'off', 'six', 'yourself', 'becoming', 'through', 
        'seeming', 'hence', 'us', 'anywhere', 'regarding', 'whole', 'down', 'seem', 'whereas', 'to', 
        'their', 'various', 'thereafter', 'â€˜d', 'above', 'put', 'sometime', 'moreover', 'whoever', 'although', 
        'at', 'four', 'each', 'among', 'whatever', 'any', 'anyhow', 'herein', 'become', 'last', 'between', 'still', 
        'was', 'almost', 'twelve', 'used', 'who', 'go', 'not', 'enough', 'well', 'â€™ve', 'might', 'see', 'whose', 
        'everywhere', 'yourselves', 'across', 'myself', 'further', 'did', 'then', 'is', 'except', 'up', 'take', 
        'became', 'however', 'many', 'thence', 'onto', 'â€˜m', 'my', 'own', 'must', 'wherein', 'elsewhere', 'behind', 
        'becomes', 'alone', 'due', 'being', 'neither', 'a', 'over', 'beside', 'fifteen', 'meanwhile', 'upon', 'next', 
        'forty', 'what', 'less', 'and', 'please', 'toward', 'about', 'below', 'hereafter', 'whether', 'yet', 'nor', 
        'against', 'whereupon', 'top', 'first', 'three', 'show', 'per', 'five', 'two', 'ourselves', 'whenever', 
        'get', 'thereby', 'noone', 'had', 'now', 'everyone', 'everything', 'nowhere', 'ca', 'though', 'least', 
        'so', 'both', 'otherwise', 'whereby', 'unless', 'somewhere', 'give', 'formerly', 'â€™d', 'under', 
        'while', 'empty', 'doing', 'besides', 'thus', 'this', 'anyone', 'its', 'after', 'bottom', 'call', 
        'nâ€™t', 'name', 'even', 'eleven', 'by', 'from', 'when', 'or', 'anyway', 'how', 'the', 'all', 
        'much', 'another', 'since', 'hundred', 'serious', 'â€˜ve', 'ever', 'out', 'full', 'themselves', 
        'been', 'in', "'d", 'wherever', 'part', 'someone', 'therein', 'can', 'seemed', 'hereby', 'others', 
        "'s", "'re", 'most', 'one', "n't", 'into', 'some', 'will', 'these', 'twenty', 'here', 'as', 'nobody', 
        'also', 'along', 'than', 'anything', 'he', 'there', 'does', 'we', 'â€™ll', 'latterly', 'are', 'ten', 
        'hers', 'should', 'they', 'â€˜s', 'either', 'am', 'be', 'perhaps', 'â€™re', 'only', 'namely', 'sixty', 
        'made', "'m", 'always', 'those', 'have', 'again', 'her', 'once', 'ours', 'herself', 'else', 'has', 'nine', 
        'more', 'sometimes', 'your', 'yours', 'that', 'around', 'his', 'indeed', 'mostly', 'cannot', 'â€˜ll', 'too', 
        'seems', 'â€™m', 'himself', 'latter', 'whither', 'amount', 'other', 'nevertheless', 'whom', 'for', 'somehow', 
        'beforehand', 'just', 'an', 'beyond', 'amongst', 'none', "'ve", 'say', 'via', 'but', 'often', 're', 'our', 
        'because', 'rather', 'using', 'without', 'throughout', 'on', 'she', 'never', 'eight', 'no', 'hereupon', 
        'them', 'whereafter', 'quite', 'which', 'move', 'thru', 'until', 'afterwards', 'fifty', 'i', 'itself', 'nâ€˜t',
        'him', 'could', 'front', 'within', 'â€˜re', 'back', 'such', 'already', 'several', 'side', 'whence', 'me', 
        'same', 'were', 'it', 'every', 'third', 'together'
    ]
}

#Function_1

### START FUNCTION 1
def dictionary_of_metrics(items):
    """A function that calculates the dictionary of metrics including mean,
    median, standard deviation, variance, minimum value, and a maximum value from a list

    parameter: 
        df: takes a list of float numbers as an input
    
    body:
        calculate the summary statistics (metrics) including mean, median, standard deviation, variance, maximum and minimum using numpy
    
    return:
        dictionary of metrics
    """
    
    mean = np.mean(items).round(2) # calculating mean
    median = np.median(items).round(2) #calculating median
    std = np.std(items, ddof=1).round(2) #calculating standard deviation
    var = np.var(items, ddof=1).round(2) #calculating variance
    min = np.min(items).round(2) # calculating minimum of value
    max = np.max(items).round(2) # calculating maximum value
 
    
    dict = {'mean':mean, 'median':median, 'std':std, 'var':var, 'min':min, 'max':max, } # dictionary of metrics
    
    
    return  dict # returning dictionary of metrics named dict

### END FUNCTION



#Function_2
### START FUNCTION
def five_num_summary(items):
    """
    The function should take a list as input.
    The function should return a dict with keys 'max', 'median', 'min', 'q1', and 'q3' corresponding to the maximum, median, minimum, first quartile and third quartile, respectively. You may use numpy functions to aid in your calculations.
    All numerical values should be rounded to two decimal places.
    """
    five_num_sum = np.percentile(items,[0, 25, 50, 75, 100])
    dict = {'min': five_num_sum[0],'Q1':five_num_sum[1],'Median':five_num_sum[2],'Q3':five_num_sum[3],'Max':five_num_sum[4]}
    
    return dict

### END FUNCTION


#Function_3
### START FUNCTION

"""This code serves to shorten input list which is given as 
    year-month-date:minutes and seconds to year-month-date
    parameter: 
        df: function takes list of strings as input.
    
    body:
        this function takes as input a list of these 
        datetime strings.
        
    
    return:
        returns a list of strings where each element in the 
        returned list contains only the date in the 'yyyy-mm-dd'
"""
def date_parser(dates):
    # your code here
    #This portion of the code slices the date to the required length.
    new_list = [magic[0:10] for magic in dates]
    # We return the list with sliced dates.
    return(new_list)

### END FUNCTION



#Function_4

### START FUNCTION
def extract_municipality_hashtags(df):
    # your code here
    
    return 

### END FUNCTION



#Function_5

### START FUNCTION
""""This code serves to calculate the number of tweets per day, 
    taking in a pandas dataframe as input and returning a new 
    dataframe grouped by day and number of tweets 

    parameter: 
        df: function takes a pandas dataframe as input.
    
    body:
        function calculates the number of tweets that were posted per day.
    
    return:
        function returns a new dataframe, grouped by day, with the number
        of tweets for that day..
"""
def number_of_tweets_per_day(df):
    # your code here
    # We slice the portion of the code needed.
    df['Date'] = [date[0:10] for date in df['Date']]
    # Drop the duplicated column.
    new_df = df.drop(columns=['Date'],axis=1)
    # Group the contents of the column by the sliced date.
    new_df = new_df.groupby(df['Date']).count() 
    return new_df

### END FUNCTION




#Function_6
### START FUNCTION
def word_splitter(df):
    # your code here
    """function that splits the sentences ina dataframe into a list
    of the separate words and returns a modified dataFrame

    parameter: 
        df: expected parameter is a pandas dataframe
    
    body:
        modifies the dataFrame into split words and assigns them to splits tweet
    
    return:
        modified dataframe with a new column of split tweets
    """
    new_df = pd.DataFrame(df) # creating a new dataframe from the old dataframe
    column1 = df['Tweets'].str.lower().values.tolist() #tweets column is extracted and coverted to a list 
    new_list = [i.split() for i in column1] # spliting the tweets
    new_df['Split Tweets'] = new_list # assigning new_list to split tweets column
    
    return new_df # returning the modified dataframe named new_df

### END FUNCTION



#Function_7
### START FUNCTION
def stop_words_remover(df):
    # your code here
    return

### END FUNCTION


if __name__ == "__main__":
    dictionary_of_metrics(gauteng)
    five_num_summary(gauteng)
    date_parser(dates[:3])
    extract_municipality_hashtags(twitter_df.copy())
    number_of_tweets_per_day(twitter_df.copy())
    word_splitter(twitter_df.copy())
    stop_words_remover(twitter_df.copy())


