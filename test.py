# coding: utf-8
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re


messages = pd.read_json(r"result.json")


messages = messages[['id', 'date', 'from','text']]
messages['sender'] = messages['from'].apply(lambda x: 'Anna' if x == 'Анюта' else 'Pavel')
messages = messages[['id', 'date', 'sender','text']]



# Count

messages['day'] = messages['date'].apply(lambda x : x.date())

piv_count_messages_day = pd.pivot_table(messages, index = 'day',columns = 'sender', values = 'id',aggfunc='count')
piv_count_messages_day = piv_count_messages_day.reset_index()

anna_count = messages[messages.sender == 'Anna'].shape[0]
pavel_count = messages[messages.sender != 'Anna'].shape[0]
anna_mean = piv_count_messages_day['Anna'].mean()
pavel_mean = piv_count_messages_day['Pavel'].mean()


#Messages
messages_count = plt.figure();
ax = messages_count.add_axes([0,0,1,1])
ax.set_xticks(np.arange(0, 150, 10))
ax.plot(piv_count_messages_day['Anna'])
ax.plot(piv_count_messages_day['Pavel'])
ax.legend(['Anna','Pavel']); 


#Count
fig = plt.figure()
fig1 = plt.figure()
ax = fig.add_axes([1,1,0.3,1])
ax_mean = fig1.add_axes([1,1,0.3,1])
langs = ['Anna', 'Pavel']
students = [anna_count,pavel_count]
ax.bar('Anna',anna_count,color = 'red')
ax.bar('Pavel',pavel_count,color = 'blue')
ax_mean.bar('Anna',anna_mean,color = 'red')
ax_mean.bar('AnPavelna',pavel_mean,color = 'blue')
plt.show()



# love
tags = pd.read_excel(r'tags.xlsx')

love_tags = tags[tags.name == 'love']['tags'].reset_index(drop=True)[0]
love_tags = love_tags.split(';')
love_tags = '|'.join(love_tags)
love_compile = re.compile(love_tags)

ans_tags = tags[tags.name == 'answer']['tags'].reset_index(drop=True)[0]
ans_tags = ans_tags.split(';')
ans_tags = '|'.join(ans_tags)
ans_compile = re.compile(ans_tags)

serd_tags = tags[tags.name == 'serd']['tags'].reset_index(drop=True)[0]
serd_tags = serd_tags.split(';')
serd_tags = '|'.join(serd_tags)
serd_compile = re.compile(serd_tags)

anal_tags = tags[tags.name == 'anal']['tags'].reset_index(drop=True)[0]
anal_tags = anal_tags.split(';')
anal_tags = '|'.join(anal_tags)
anal_compile = re.compile(anal_tags)

tnx_tags = tags[tags.name == 'tnx']['tags'].reset_index(drop=True)[0]
tnx_tags = tnx_tags.split(';')
tnx_tags = '|'.join(tnx_tags)
tnx_compile = re.compile(tnx_tags)

smile_tags = tags[tags.name == 'smile']['tags'].reset_index(drop=True)[0]
smile_tags = smile_tags.split(';')
smile_tags = '|'.join(smile_tags)
smile_compile = re.compile(smile_tags)

youtube_tags = tags[tags.name == 'youtube']['tags'].reset_index(drop=True)[0]
youtube_tags = youtube_tags.split(';')
youtube_tags = '|'.join(youtube_tags)
youtube_compile = re.compile(youtube_tags)






messages['text'] = messages['text'].apply(str)
messages['love'] = messages['text'].apply(lambda x: bool(love_compile.search(x)))
messages['ans'] = messages['text'].apply(lambda x: bool(ans_compile.search(x)))
messages['serd'] = messages['text'].apply(lambda x: bool(serd_compile.search(x)))
messages['anal'] = messages['text'].apply(lambda x: bool(anal_compile.search(x)))
messages['tnx'] = messages['text'].apply(lambda x: bool(tnx_compile.search(x)))
messages['smile'] = messages['text'].apply(lambda x: bool(smile_compile.search(x)))
messages['youtube'] = messages['text'].apply(lambda x: bool(youtube_compile.search(x)))


messages[(messages.love) & (messages.sender == 'Pavel')].shape[0]
messages[(messages.love) & (messages.sender == 'Anna')].shape[0]
messages[(messages.ans) & (messages.sender == 'Pavel')].shape[0]
messages[(messages.ans) & (messages.sender == 'Anna')].shape[0]





piv_love = pd.pivot_table(messages, index = 'sender',values = ['love','ans',
                                                                'serd','anal','tnx',
                                                                'smile', 'youtube'],aggfunc='sum')

piv_love = piv_love.reset_index()








