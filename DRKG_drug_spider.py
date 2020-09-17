#coding=utf-8
import sqlite3
import requests
from bs4 import BeautifulSoup as bs
import math
import pandas as pd
import time

def download(basic_url,url_id,num_retries=150):
    url=basic_url+str(url_id)
    try:
        html=requests.get(url).content
        soup=bs(html,'lxml')
    except:
        if num_retries>0:
            print (num_retries)
            return download(url,num_retries-1)
        else:
            time.sleep(30)
            return download(url,150)

    return soup

def get_id(smd_page,bd_page):
    url_id=[]
    for i in range(smd_page):
        print (i+1)
        #basic_url='https://www.drugbank.ca/drugs?approved=1&c=name&d=up&page='
        #drug 分成两种，smd——small molecule drug bd——biotech drug，网址不同
        basic_url='https://www.drugbank.ca/drugs?approved=1&c=name&ca=0&d=up&eu=0&experimental=1&illicit=1&investigational=1&nutraceutical=1&us=0&withdrawn=1&page='
        soup=download(basic_url,i+1,num_retries=150)
        for j in soup.select('.name-value strong a'):
            url_id.append(j.attrs['href'].split('/')[-1])
    for i in range(bd_page):
        print (i+1)
        #basic_url='https://www.drugbank.ca/biotech_drugs?approved=1&Protein+Based+Therapies=0&page='
        basic_url='https://www.drugbank.ca/biotech_drugs?utf8=%E2%9C%93&approved=0&nutraceutical=0&illicit=0&investigational=0&withdrawn=0&experimental=0&us=0&ca=0&eu=0&Protein+Based+Therapies=0&Nucleic+Acid+Based+Therapies=0&Gene+Therapies=0&Vaccines=0&Allergenics=0&Cell+transplant+therapies=0&commit=Apply+Filter&page='
        soup=download(basic_url,i+1,num_retries=150)
        for j in soup.select('.name-value strong a'):
            url_id.append(j.attrs['href'].split('/')[-1])
    return url_id

def identification(soup):
    iden_list=[]
    attr_list=[]
    d={}
    for i in soup.find('dl').findAll('dt'):
        iden_list.append(i.text)
    for i in soup.find('dl').findAll('dd'):
        attr_list.append(i.text)
    for i in range(len(attr_list)):
        d[iden_list[i]]=attr_list[i]
    return d


def interactions(url_id,name):
    interaction=''
    url='https://www.drugbank.ca/drugs/'+url_id+'/drug_interactions.json?group=approved&'
    try:
        length = requests.get(url).json()['recordsTotal']
        for j in range(math.floor(length/100)+1):
            #这里是100个、100个反应事件来爬虫，因为貌似给出的json上限是100个，也有可能我用的和李景皓学长不是同一个响应头
            new_url=url+'start='+str(100*j)+'&length=100'
            for i in requests.get(new_url).json()['data']:
                interaction_id=(bs(i[0],'lxml').find('a').attrs['href'].split('/')[-1]+'|')
                name2=i[0].split('<')[1]
                name2=name2.split('>')[-1]
                interaction+=interaction_id
                event.append(i[1])
                #这里要预先创建好event表
                cur.execute("insert into event(id1,name1,id2,name2,interaction)values(?,?,?,?,?)",(url_id,name,interaction_id[:-1],name2,i[1]))
                #爬完后再筛掉不在反映列表里的drug
            interaction=interaction[:-1]
    except:
        pass
    return interaction,event

def head_attr(soup):
    d={}
    try:
        for i in soup.select('.bond-list-container'):
            attr=''
            for j in i.select('.bond-list strong a'):
                attr+=(j.attrs['href'].split('/')[-1]+'|')
            d[i.h3.text]=attr[:-1]
    except:
        pass
    return d

conn=sqlite3.connect("Drug_original_addDrug.db")
cur=conn.cursor()
#cur.execute('''create table drug512(id,name,interaction,smile,target,enzyme,carrier,transporter);''')
#所有drug的话457 87
#approved drug 106 56
#url_id=get_id(106,56)
basic_url='https://www.drugbank.ca/drugs/'
event=[]
drug=pd.read_excel("drug_list.xlsx",header=None)
url_id=['DB00048','DB02427','DB05697','DB06517','DB15351','DB15599']
for i in url_id:
    soup=download(basic_url,i,num_retries=150)
    try:
        d_iden=identification(soup)
    except:
        continue
    try:
        name=d_iden['Name']
    except:
        name=''
    try:
        smile=d_iden['SMILES']
        if smile=='Not Available':
            smile=''
    except:
        smile=''
    interaction,event=interactions(i,name)

    d_attr=head_attr(soup)
    try:
        target=d_attr['Targets']
    except:
        target=''
    try:
        enzyme=d_attr['Enzymes']
    except:
        enzyme=''
    try:
        carrier=d_attr['Carriers']
    except:
        carrier=''
    try:
        transporter=d_attr['Transporters']
    except:
        transporter=''
    #这里预先要创建好drug表
    cur.execute("insert into drug(id,name,interaction,smile,target,enzyme,carrier,transporter)values(?,?,?,?,?,?,?,?)",(i,name,interaction,smile,target,enzyme,carrier,transporter))
    print (i)
conn.commit()
conn.close()
# =============================================================================
# data=xlrd.open_workbook('all_drug.xls')
# table=data.sheets()[0]
# l= table.col_values(0)
# for _id in l:
#     html=download(_id)
#     soup=downloads(html)
#     conn=sqlite3.connect("D:\Drug.db")
#     conn.execute("insert into drugbank(id,name,smiles,interactions,targets,enzymes,carrier,transporters)values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(_id,name(html),smiles(html),interactions(html),targets(html),enzymes(html),carrier(html),transporters(html)))
#     conn.commit()
# =============================================================================

