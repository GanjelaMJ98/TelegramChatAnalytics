# coding: utf-8
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import string
import re


messages = pd.read_json(r"result.json")

messages = messages[['id', 'date', 'from','text']]
messages['sender'] = messages['from'].apply(lambda x: 'Anna' if x == 'Анюта' else 'Pavel')
messages = messages[['id', 'date', 'sender','text']]
print(f'''
MIN DATE all message: {messages.date.min()}
MAX DATE all message: {messages.date.max()}''')


# Count

messages['day'] = messages['date'].apply(lambda x : x.date())

piv_count_messages_day = pd.pivot_table(messages, index = 'day',columns = 'sender', values = 'id',aggfunc='count')
piv_count_messages_day = piv_count_messages_day.reset_index()
piv_count_messages_day['m'] = piv_count_messages_day['day'].apply(lambda x : str(x)[:7])

anna_count = messages[messages.sender == 'Anna'].shape[0]
pavel_count = messages[messages.sender != 'Anna'].shape[0]
anna_mean = int(piv_count_messages_day['Anna'].mean())
pavel_mean = int(piv_count_messages_day['Pavel'].mean())

print(f'''
Anna all message: {anna_count} {round(anna_count/(anna_count+pavel_count)*100,1)}%
Pasha all message: {pavel_count} {round(pavel_count/(anna_count+pavel_count)*100,1)}%
Anna mean message: {anna_mean} {round(anna_mean/(anna_mean+pavel_mean)*100,1)}%
Pasha mean message: {pavel_mean} {round(pavel_mean/(anna_mean+pavel_mean)*100,1)}%''')


#Messages
messages_count = plt.figure();
ax = messages_count.add_axes([0,0,1,1])
ax.set_xticks(np.arange(0, 150, 10))
ax.plot(piv_count_messages_day['Anna'], piv_count_messages_day['day'])
ax.plot(piv_count_messages_day['Pavel'], piv_count_messages_day['day'])
ax.legend(['Anna','Pavel']);


df = piv_count_messages_day[['day','Anna','Pavel']].melt('day', var_name='cols',  value_name='vals')

import seaborn as sns
sns.set(style="whitegrid")
sns.set(font_scale=1.2)
sns.relplot(
    data=df,
    x="day", y="vals", hue= 'cols',
    kind="line",
    palette=["#ffb1c6",'#45c6ef'],
    height=5, aspect=2, linewidth = 3
).set(
    title="Stock Prices",
    ylabel="count",
    xlabel='date'
)




#Count
# fig = plt.figure()
# fig1 = plt.figure()
# ax = fig.add_axes([1,1,0.3,1])
# ax_mean = fig1.add_axes([1,1,0.3,1])
# langs = ['Anna', 'Pavel']
# students = [anna_count,pavel_count]
# ax.bar('Anna',anna_count,color = 'red')
# ax.bar('Pavel',pavel_count,color = 'blue')
# ax_mean.bar('Anna',anna_mean,color = 'red')
# ax_mean.bar('Pavel',pavel_mean,color = 'blue')
# plt.show()



# tags
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

skuch_tags = tags[tags.name == 'skuka']['tags'].reset_index(drop=True)[0]
skuch_tags = skuch_tags.split(';')
skuch_tags = '|'.join(skuch_tags)
skuch_compile = re.compile(skuch_tags)


mat_words = pd.read_excel(r'mat_tags.xlsx')
mat_words = mat_words['tag'].to_list()
mat_words.append('ебан')

mat_tags = '|'.join(mat_words)
mat_tags = re.compile(mat_tags)





messages['text'] = messages['text'].apply(str)
messages['text'] = messages['text'].apply(str.lower)

messages['love'] = messages['text'].apply(lambda x: bool(love_compile.search(x)))
messages['ans'] = messages['text'].apply(lambda x: bool(ans_compile.search(x)))
messages['skuka'] = messages['text'].apply(lambda x: bool(skuch_compile.search(x)))
messages['serd'] = messages['text'].apply(lambda x: bool(serd_compile.search(x)))
messages['anal'] = messages['text'].apply(lambda x: bool(anal_compile.search(x)))
messages['tnx'] = messages['text'].apply(lambda x: bool(tnx_compile.search(x)))
messages['smile'] = messages['text'].apply(lambda x: bool(smile_compile.search(x)))
messages['youtube'] = messages['text'].apply(lambda x: bool(youtube_compile.search(x)))
messages['mat'] = messages['text'].apply(lambda x: bool(mat_tags.search(x)))


#net = messages[messages.mat == False]
#est = messages[messages.mat]
messages[(messages.love) & (messages.sender == 'Pavel')].shape[0]
messages[(messages.love) & (messages.sender == 'Anna')].shape[0]
messages[(messages.ans) & (messages.sender == 'Pavel')].shape[0]
messages[(messages.ans) & (messages.sender == 'Anna')].shape[0]





piv_love = pd.pivot_table(messages, index = 'sender',values = ['love','ans',
                                                                'serd','anal','tnx',
                                                                'smile', 'youtube','mat','skuka'],aggfunc='sum')


piv_love = piv_love.transpose()
piv_love['Anna_proc'] = piv_love.apply(lambda x : round((x.Anna/(x.Anna + x.Pavel))*100,1),axis=1)
piv_love['Pavel_proc'] = piv_love.apply(lambda x : 100 - x.Anna_proc ,axis=1)

from tabulate import tabulate
print(tabulate(piv_love, headers='keys', tablefmt='psql'))




#--------#
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from PIL import Image

mask = np.array(Image.open('cloud1.png'))

russian_stopwords = stopwords.words("russian")
russian_stopwords.extend(['это','link\'','text\'','type\''])







messages_with_mat = messages[messages.mat]['text'].to_list()
s_messages_with_mat = ' '.join(messages_with_mat)
s_messages_with_mat = s_messages_with_mat.replace('\n', ' ')
s_messages_with_mat = s_messages_with_mat.translate(str.maketrans('', '', string.punctuation))
messages_with_mat_1= s_messages_with_mat.split(' ')
mat_words_clear = []
for word in messages_with_mat_1:
    if bool(mat_tags.search(word)):
        mat_words_clear.append(word)
mat_words_clear_gen = ' '.join(mat_words_clear)


wordcloud_mat = WordCloud(width = 1920, height = 1080,
                background_color ='white',
                max_words=300,
                min_font_size = 10).generate(mat_words_clear_gen)
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud_mat)
plt.axis("off")
plt.tight_layout(pad = 0)
#wordcloud_mat.to_file('wordcloud_mat.png')
plt.show()







# wordcloud
messages['text'] = messages['text'].apply(str)
all_words = messages['text'].to_list()

all_words = ' '.join(all_words)
all_words = all_words.replace('\n', ' ')
all_words = all_words.lower()
all_words = all_words.translate(str.maketrans('', '', string.punctuation))
all_words = ' '.join(set(all_words.split(' ')) - set(mat_words_clear))


wordcloud = WordCloud(width = 1920, height = 1080,
                background_color ='white',
                stopwords=russian_stopwords,
                max_words=500,
                mask = mask,
                min_font_size = 10).generate(all_words)

# plot the WordCloud image
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)

plt.show()
wordcloud.to_file('lol1.png')
