# coding: utf-8
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import numpy as np
import seaborn as sns
import string
import re


messages_orig = pd.read_json(r"result.json")
messages = pd.DataFrame(messages_orig.messages.to_list())
messages['sender'] = messages['from'].apply(lambda x: 'Anna' if x == 'Анюта' else 'Pavel')
##########################################################

# Message type analize
messages_df = messages
type_analize = pd.DataFrame()
index = 0
for sender in ['Anna', 'Pavel']:
    sender_mes = messages_df[messages_df.sender == sender]
    emoji = pd.pivot_table(sender_mes, columns='sticker_emoji',values ='id',aggfunc='count').transpose().reset_index()
    emoji = emoji.sort_values('id',ascending=False).head(5).sticker_emoji.to_list()
    data = {
        'sender' : sender,
        'photo_cnt' : sender_mes[sender_mes.photo == sender_mes.photo].shape[0],
        'video_cnt' : sender_mes[sender_mes.media_type == 'video_file'].shape[0],
        'video_dur_max' : sender_mes[sender_mes.media_type == 'video_file'].duration_seconds.mean(),
        'video_dur_mean' : sender_mes[sender_mes.media_type == 'video_file'].duration_seconds.max(),
        'voice_message' : sender_mes[sender_mes.media_type == 'voice_message'].shape[0],
        'voice_dur_mean' : sender_mes[sender_mes.media_type == 'voice_message'].duration_seconds.mean(),
        'voice_dur_max' : sender_mes[sender_mes.media_type == 'voice_message'].duration_seconds.max(),
        'video_message' : sender_mes[sender_mes.media_type == 'video_message'].shape[0],
        'video_message_dur_mean' : sender_mes[sender_mes.media_type == 'video_message'].duration_seconds.mean(),
        'sticker' : sender_mes[sender_mes.media_type == 'sticker'].shape[0],
        'animation' : sender_mes[sender_mes.media_type == 'animation'].shape[0] ,
        'audio_file' : sender_mes[sender_mes.media_type == 'audio_file'].shape[0] ,
        'edited' : sender_mes[sender_mes.edited == sender_mes.edited].shape[0],
        'emoji' : str(emoji),
    }
    d = pd.DataFrame(data,index = [index])
    index +=1
    type_analize = type_analize.append(d)

type_analize = type_analize.transpose()
type_analize.columns = type_analize.loc['sender']
type_analize['a_proc'] = type_analize.apply(lambda x : round((x.Anna/(x.Anna + x.Pavel) )*100,1) if type(x.Anna) != str else 0,axis=1)
type_analize['Pavel_proc'] = type_analize.apply(lambda x : 100 - x.a_proc ,axis=1)
# type_analize.to_excel('OUT/type_analize.xlsx')


w = pd.pivot_table(messages_df, columns='width',values ='id',aggfunc='count').transpose().reset_index()
h = pd.pivot_table(messages_df, columns='height',values ='id',aggfunc='count').transpose().reset_index()
print(f'Popular image size: {int(h[h.id == h.id.max()].height.values[0])}x{int(w[w.id == w.id.max()].width.values[0])}')




##########################################################
# Message date analize
messages = messages[['id', 'date', 'sender','text']]
messages['date'] = pd.to_datetime(messages['date'])

print(f'''
MIN DATE all message: {messages.date.min()}
MAX DATE all message: {messages.date.max()}''')

##########################################################
# Message count analize

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


##########################################################
# Message day_parts analize
piv_count_messages_date = pd.pivot_table(messages, index = 'date',columns = 'sender', values = 'id',aggfunc='count')
piv_count_messages_date = piv_count_messages_date.reset_index()
piv_count_messages_date['hour'] = piv_count_messages_date['date'].apply(lambda x : x.hour)

day_parts = {   'morning' : [7,11],
                'noon' : [12,16],
                'evening' : [17,22],
                'night' : [23,6]}


morning = piv_count_messages_date[(piv_count_messages_date['hour'] >= day_parts['morning'][0])
                            & (piv_count_messages_date['hour'] <= day_parts['morning'][1])]
noon = piv_count_messages_date[(piv_count_messages_date['hour'] >= day_parts['noon'][0])
                            & (piv_count_messages_date['hour'] <= day_parts['noon'][1])]
evening = piv_count_messages_date[(piv_count_messages_date['hour'] >= day_parts['evening'][0])
                            & (piv_count_messages_date['hour'] <= day_parts['evening'][1])]
night = piv_count_messages_date[(piv_count_messages_date['hour'] >= day_parts['night'][0])
                            | (piv_count_messages_date['hour'] <= day_parts['night'][1])]

df_per_day = piv_count_messages_date[['hour','Anna','Pavel']].melt('hour', var_name='cols',  value_name='vals')
df_per_day['time'] = df_per_day['hour'].apply(lambda x: 'morning' if x >= day_parts['morning'][0] and x <=day_parts['morning'][1]
                                                    else 'daytime' if x >= day_parts['noon'][0] and x <= day_parts['noon'][1]
                                                    else 'evening' if x >= day_parts['evening'][0] and x <= day_parts['evening'][1]
                                                    else 'night' if x >= day_parts['night'][0] or x <= day_parts['night'][1]
                                                    else 'lol')
df_per_day = df_per_day.dropna()
f, ax = plt.subplots(figsize=(20, 10))
sns.set(font_scale=1.3)
per_day = sns.countplot(data = df_per_day,
            x = 'time',hue='cols',
            palette=["#ffb1c6",'#45c6ef'], orient ='h',
            )
# per_day.get_figure().savefig('OUT/per_day.png')






##########################################################
# Message years analize
df = piv_count_messages_day[['day','Anna','Pavel']].melt('day', var_name='cols',  value_name='vals')
df['year'] = df['day'].apply(lambda x : str(x)[:4])

for year in df.year.unique():
    sns.set(style="whitegrid")
    sns.set(font_scale=1.2)
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    year_stats = sns.relplot(
        data=df[df.year == year],
        x="day", y="vals", hue= 'cols',
        kind="line",
        palette=["#ffb1c6",'#45c6ef'],
        height=5, aspect=2, linewidth = 3
    ).set(
        title="Stock Prices",
        ylabel="count",
        xlabel='date'
    )
    #year_stats.fig.savefig('OUT/2019.png')




##########################################################
# tags
# TODO: Add TagSearch class
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
##########################################################

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
# piv_love.to_excel(r'OUT/tags_search.xlsx')



##########################################################
# WordCloud

import nltk
from PIL import Image
from nltk.corpus import stopwords

mask = np.array(Image.open('cloud1.png'))
nltk.download("stopwords")

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
                max_words=500,
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


#custom
all_words = all_words.replace('nn','')
all_words = all_words.replace('n','')

wordcloud = WordCloud(width = 1920, height = 1080,
                background_color ='white',
                stopwords=russian_stopwords,
                max_words=500,
                #mask = mask,
                min_font_size = 10).generate(all_words)

# plot the WordCloud image
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)

plt.show()
wordcloud.to_file('OUT/wordcloud_orig.png')
wordcloud_mat.to_file(r'OUT/wordcloud_mat.png',)
