import requests
import re
import numpy as np
import pandas as pd
import json


def spider(url, comments):
    # url = 'https://weibo.com/ajax/statuses/buildComments?flow=0&is_reload=1&id=4471033010394297&is_show_bulletin=2&is_mix=0&max_id=0&count=20&uid=6486930715'
    headers = {
        'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0'
    }
    res = requests.get(url,headers=headers)
    res1 = res.json()
    count = 1
    max_id = 0
    while True:
        if len(res1['data'])==0:
            print("以获取全部评论！")
            break
        print("--------爬取第{}页数据--------".format(count))
        max_id = res1['max_id']
        count += 1
        # print(len(res1['data']))
        for i in range(len(res1['data'])):
            comment = res1['data'][i]['text']
            comments.append(comment)
        url = url.split('=')
        url1 = ''
        i = 0
        for i in range(len(url)):
            if 'max_id' in url[i]:
                url1 += url[i]
                url1 += '='
                break
            else:
                url1 += url[i]
                url1 += '='
        url1 = url1 + str(max_id) + '&count='
        for j in range(i+2,len(url)):
            url1 += url[j]
        url = url1
        res = requests.get(url,headers=headers)
        res1 = res.json()

    return comments

if __name__ == '__main__':
    comments1 = []
    comments2 = []
    comments3 = []
    comments4 = []
    url1 = 'https://weibo.com/ajax/statuses/buildComments?flow=0&is_reload=1&id=5005259468181302&is_show_bulletin=2&is_mix=0&max_id=0&count=20&uid=5539703320&fetch_level=0&locale=zh-CN'
    url2 = 'https://weibo.com/ajax/statuses/buildComments?flow=0&is_reload=1&id=5005353962441013&is_show_bulletin=3&is_mix=0&max_id=0&count=20&uid=2140522467&fetch_level=0&locale=zh-CN'
    url3 = 'https://weibo.com/ajax/statuses/buildComments?flow=0&is_reload=1&id=5005644109451631&is_show_bulletin=2&is_mix=0&max_id=0&count=20&uid=3197845214&fetch_level=0&locale=zh-CN'
    url4 = 'https://weibo.com/ajax/statuses/buildComments?flow=0&is_reload=1&id=4970454142943452&is_show_bulletin=2&is_mix=0&max_id=0&count=20&uid=5851185687&fetch_level=0&locale=zh-CN'
    spider(url1, comments1)
    spider(url2, comments2)
    spider(url3, comments3)
    spider(url4, comments4)
    comments1.extend(comments2)
    comments1.extend(comments3)
    comments1.extend(comments4)
    data = pd.DataFrame(comments1, columns=['评论'])
    print(data.shape)
    data.to_csv('data.csv', sep=',', index=False, encoding='utf-8-sig')