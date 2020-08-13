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

    Parameter: 
        df: takes a list of float numbers as an input
    
    Body:
        calculate the summary statistics (metrics) including mean, median, standard deviation, variance, maximum and minimum using numpy
    
    Return:
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
    # your code here
    return

### END FUNCTION


#Function_3
### START FUNCTION
def date_parser(dates):
    # your code here
    return

### END FUNCTION



#Function_4

### START FUNCTION
def extract_municipality_hashtags(df):
    """ Function which takes in a pandas dataframe and returns a modified dataframe that includes 
    two new columns that contain information about the municipality and hashtag of the tweet 

    Param
        Pandas dataframe df

    Body
        Modifies the dataFrame to extract the municipality and hashtags. Assigns null value to missing data.
    
    Return:
        Modified dataframe with Municipality and Hashtags column
    """
    
    hashtags = []
    municipality = []

    tweets = [i.split(" ") for i in df['Tweets']]

    new_munic_list = []
    new_tag_list = []

    for tweet in tweets:
        municipality.append([mun_dict[word] for word in tweet if word in list(mun_dict.keys())])
        hashtags.append([tag.lower() for tag in tweet if tag.startswith('#')])

    for item in municipality:
        if item == []:
            item = np.nan  
        new_munic_list.append(item)

    for tag in hashtags:
        if tag == []:
            tag = np.nan
        new_tag_list.append(tag)
    
    df['municipality'] = new_munic_list
    df['hashtags'] = new_tag_list
  
    return df
     

### END FUNCTION



#Function_5

### START FUNCTION
def number_of_tweets_per_day(df):
    # your code here
    return

### END FUNCTION




#Function_6
### START FUNCTION
def word_splitter(df):
    # your code here
    """eter: 
        df: expected parameter is a pandas dataframe
    
    body:  function that splits the sentences ina dataframe into a list
    of the separate words and returns a modified dataFrame

    Param
        modifies the dataFrame into split words and assigns them to splits tweet
    
    Return:
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
   """Function that splits the sentences ina dataframe into a list
    of the separate words and removes all stop words 

     Param
        Pandas dataframe df
        
    Body
        Modifies the dataFrame into split words and removes the stop words from the tweet
    
    Return:
        Modified dataframe of split tweets without stop words
    """
    df['Without Stop Words'] = df['Tweets'].apply(lambda x: [item for item in str(x).lower().split() if item not in stop_words_dict['stopwords']])

    return df

### END FUNCTION


if __name__ == "__main__":
    dictionary_of_metrics(gauteng)
    five_num_summary(gauteng)
    date_parser(dates[:3])
    extract_municipality_hashtags(twitter_df.copy())
    number_of_tweets_per_day(twitter_df.copy())
    word_splitter(twitter_df.copy())
    stop_words_remover(twitter_df.copy())


