from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)
import csv
import sqlite3
import time
import numpy as np
import pandas as pd
from pandas import DataFrame
import xlsxwriter
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Activation, BatchNormalization, Convolution1D, MaxPool1D
from keras.utils.vis_utils import plot_model
from keras import regularizers
from keras.callbacks import EarlyStopping

event_num = 65
droprate = 0.3
vector_size = 572


def DNN():
    train_input = Input(shape=(vector_size * 2,), name='Inputlayer')
    # train_in = Dense(2048, activation='relu',name='FullyConnectLayer1')(train_input)
    # train_in = BatchNormalization()(train_in)
    # train_in = Dropout(droprate)(train_in)
    # train_in = Dense(1024, activation='relu',name="FullyConnectLayer2")(train_in)
    # train_in = BatchNormalization()(train_in)
    # train_in = Dropout(droprate)(train_in)
    train_in = Dense(512, activation='relu', name="FullyConnectLayer3")(train_input)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)
    train_in = Dense(256, activation='relu', name="FullyConnectLayer4")(train_in)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)
    train_in = Dense(event_num, name="SoftmaxLayer")(train_in)
    out = Activation('softmax', name="OutputLayer")(train_in)
    model = Model(input=train_input, output=out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


"""
model = Sequential()
model.add(Dense(2048,activation='relu',input_shape=(vector_size*2,)))
model.add(Dropout(droprate))
model.add(Dense(1024,activation='relu'))
model.add(Dropout(droprate))
model.add(Dense(512,activation='relu'))
model.add(Dropout(droprate))
model.add(Dense(256,activation='relu'))
model.add(Dense(event_num,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
"""


def prepare(df_drug, df_event, df_interaction, feature_list, vector_size, interaction_num):
    # df_event was read from event_number
    # df_interaction was read from mirrow1
    for index, row in df_event.iterrows():
        if eval(row['number']) >= interaction_num:
            event_num = (index + 1)
    d_label = {}  # 把反应名称和数字一一对应，数字是1-65
    d_feature = {}  # 把药物ID和特征向量一一对应，向量是200维度
    # 把反应类型转化为数字
    label_value = 0
    for i in np.array(df_event['event']).tolist():
        d_label[i] = label_value
        label_value += 1

    # 把需要用到的特征连接到一起
    vector = np.zeros((len(np.array(df_drug['id']).tolist()), 0), dtype=float)
    map={}
    for index,row in df_drug.iterrows():
        map[row['id']]=index
    for i in feature_list:
        vector = np.hstack((vector, feature_vector(i, df_drug, vector_size)))
    # 把药物ID转化为特征向量
    for i in range(len(np.array(df_drug['id']).tolist())):
        d_feature[np.array(df_drug['id']).tolist()[i]] = vector[i]
    # 利用上面两个字典获得特征向量和标签
    new_feature = []
    new_label = []
    record_label = []
    for index, row in df_interaction.iterrows():
        if d_label[row['interaction']] < event_num:
            new_feature.append(np.hstack((d_feature[row['id1']], d_feature[row['id2']])))
            record_label.append([map[row['id1']],map[row['id2']]])
            new_label.append(d_label[row['interaction']])
    new_feature = np.array(new_feature)
    new_label = np.array(new_label)
    return (new_feature, new_label, event_num,record_label)


def feature_vector(feature_name, df, vector_size):
    # df are the 572 kinds of drugs
    # 相似度计算函数
    def Jaccard(matrix):
        matrix = np.mat(matrix)
        numerator = matrix * matrix.T
        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
        return numerator / denominator

    all_feature = []
    drug_list = np.array(df[feature_name]).tolist()
    # 每个药物的特征 for example, when feature_name is target, drug_list=["P30556|P05412","P28223|P46098|……"]
    for i in drug_list:
        for each_feature in i.split('|'):
            if each_feature not in all_feature:
                all_feature.append(each_feature)  # 得到所有的特征
    print("length of all feature is", len(all_feature))
    feature_matrix = np.zeros((len(drug_list), len(all_feature)), dtype=float)
    df_feature = DataFrame(feature_matrix, columns=all_feature)  # 利用dataframe的键构建特征矩阵
    for i in range(len(drug_list)):
        for each_feature in df[feature_name].iloc[i].split('|'):
            df_feature[each_feature].iloc[i] = 1
    # sim_matrix=np.array(df_feature)
    # df_feature=np.array(df_feature)
    sim_matrix = Jaccard(np.array(df_feature))
    #worksheet = workBook.add_worksheet(feature_name)
    sim_matrix1 = np.array(sim_matrix)
    count = 0

    print(sim_matrix)
    # sim_matrix=np.array(df_feature)
    pca = PCA(n_components=vector_size)  # 指定降维维度
    pca.fit(sim_matrix)
    sim_matrix = pca.transform(sim_matrix)
    print("After PCA its shape is", sim_matrix.shape)
    return sim_matrix
    # return df_feature


def get_index(label_matrix, event_num, seed, CV):
    index_all_class = np.zeros(len(label_matrix))
    for j in range(event_num):
        index = np.where(label_matrix == j)
        kf = KFold(n_splits=CV, shuffle=True, random_state=seed)
        k_num = 0
        for train_index, test_index in kf.split(range(len(index[0]))):
            print("train_index,test_index", train_index, test_index)
            index_all_class[index[0][test_index]] = k_num
            k_num += 1
        # for (train_index,test_index) in kf.split(range(len(index[0]))):

    return index_all_class


def cross_validation(feature_matrix, label_matrix, clf_type, event_num, seed, CV, set_name,train_drug,test_drug,record_label):
    all_eval_type = 11
    result_all1 = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve1 = np.zeros((event_num, each_eval_type), dtype=float)
    result_all2 = np.zeros((all_eval_type, 1), dtype=float)
    result_eve2 = np.zeros((event_num, each_eval_type), dtype=float)
    y_true1 = np.array([])
    y_pred1 = np.array([])
    y_score1 = np.zeros((0, event_num), dtype=float)
    y_true2 = np.array([])
    y_pred2 = np.array([])
    y_score2 = np.zeros((0, event_num), dtype=float)
    #index_all_class = get_index(label_matrix, event_num, seed, CV)
    matrix = []
    if type(feature_matrix) != list:
        matrix.append(feature_matrix)
        # =============================================================================
        #     elif len(np.shape(feature_matrix))==3:
        #         for i in range((np.shape(feature_matrix)[-1])):
        #             matrix.append(feature_matrix[:,:,i])
        # =============================================================================
        feature_matrix = matrix
    is_train_drug=[False]*572
    for i in train_drug:
        is_train_drug[i]=True;
    train_index=[]
    test1_index=[]
    test2_index=[]
    count=0
    for i in record_label:
        if (is_train_drug[i[0]] and is_train_drug[i[1]]):
            train_index.append(count)
        elif(is_train_drug[i[0]]) or (is_train_drug[i[1]]):
            test1_index.append(count)
        else:
            test2_index.append(count)
        count=count+1
    train_index=np.array(train_index)

    test1_index=np.array(test1_index)
    test2_index=np.array(test2_index)


    pred1 = np.zeros((test1_index.shape[0], event_num), dtype=float)
    pred2 = np.zeros((test2_index.shape[0], event_num), dtype=float)
    for i in range(len(feature_matrix)):
        x_train = feature_matrix[i][train_index]
        print(x_train.shape)
        x_test1 = feature_matrix[i][test1_index]
        x_test2 = feature_matrix[i][test2_index]
        y_train = label_matrix[train_index]
        # one-hot encoding
        y_train_one_hot = np.array(y_train)
        y_train_one_hot = (np.arange(65) == y_train[:, None]).astype(dtype='float32')
        y_test1 = label_matrix[test1_index]
        y_test2 = label_matrix[test2_index]
        # one-hot encoding
        y_test1_one_hot = np.array(y_test1)
        y_test1_one_hot = (np.arange(65) == y_test1[:, None]).astype(dtype='float32')
        y_test2_one_hot = np.array(y_test2)
        y_test2_one_hot = (np.arange(65) == y_test2[:, None]).astype(dtype='float32')
        if clf_type == 'DNN':
            dnn = DNN()
            # dnn.summary()
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

            dnn.fit(x_train, y_train_one_hot, batch_size=128, epochs=100, validation_data=(x_test1, y_test1_one_hot),
                    callbacks=[early_stopping])
            #dnn.fit(x_train, y_train_one_hot, batch_size=128, epochs=100, validation_data=(x_test1, y_test1_one_hot),)
            pred1 += dnn.predict(x_test1)
            pred2 += dnn.predict(x_test2)
            continue
        elif clf_type == 'RF':
            clf = RandomForestClassifier(n_estimators=100)
        elif clf_type == 'GBDT':
            clf = GradientBoostingClassifier()
        elif clf_type == 'SVM':
            clf = SVC(probability=True)
        elif clf_type == 'FM':
            clf = GradientBoostingClassifier()
        elif clf_type == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=4)
        else:
            clf = LogisticRegression()
        clf.fit(x_train, y_train)
        pred1 += clf.predict_proba(x_test1)
        pred2 += clf.predict_proba(x_test2)
    pred_score1 = pred1 / len(feature_matrix)
    pred_type1 = np.argmax(pred_score1, axis=1)
    y_true1 = np.hstack((y_true1, y_test1))
    y_pred1 = np.hstack((y_pred1, pred_type1))
    y_score1 = np.row_stack((y_score1, pred_score1))
    pred_score2 = pred2 / len(feature_matrix)
    pred_type2 = np.argmax(pred_score2, axis=1)
    y_true2 = np.hstack((y_true2, y_test2))
    y_pred2 = np.hstack((y_pred2, pred_type2))
    y_score2 = np.row_stack((y_score2, pred_score2))
    result_all1, result_eve1 = evaluate(y_pred1, y_score1, y_true1, event_num, set_name)
    result_all2, result_eve2 = evaluate(y_pred2, y_score2, y_true2, event_num, set_name)
    # =============================================================================
    #         a,b=evaluate(pred_type,pred_score,y_test,event_num)
    #         for i in range(all_eval_type):
    #             result_all[i]+=a[i]
    #         for i in range(each_eval_type):
    #             result_eve[:,i]+=b[:,i]
    #     result_all=result_all/5
    #     result_eve=result_eve/5
    # =============================================================================
    return result_all1, result_eve1,result_all2,result_eve2


def evaluate(pred_type, pred_score, y_test, event_num, set_name):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    #y_one_hot = label_binarize(y_test, np.arange(event_num))
    y_one_hot = np.array(y_test)
    y_one_hot = (np.arange(65) == y_one_hot[:, None]).astype(dtype='float32')
    #pred_one_hot = label_binarize(pred_type, np.arange(event_num))
    pred_one_hot = np.array(pred_type)
    pred_one_hot = (np.arange(65) == pred_one_hot[:, None]).astype(dtype='float32')
    # df1=pd.DataFrame({'y_ture':y_test})
    # df2=pd.DataFrame({'y_pred':pred_score})
    # ALL=[df1,df2]
    # writer=pd.ExcelWriter("true-pred.xlsx")
    # df1.to_excel(writer,sheet_name='SMILES',index=False)
    # df2.to_excel(writer,sheet_name='SMILES',index=False)

    precision, recall, th = multiclass_precision_recall_curve(y_one_hot, pred_score)
    count = 0
    new_precision = []
    new_recall = []
    new_th = []
    for i in range(len(precision)):
        count = count + 1
        if (count == 10000):
            count = 0
            new_precision.append(precision[count])
            new_recall.append(recall[count])
            new_th.append(th[count])
    new_precision = np.array(new_precision)
    new_recall = np.array(new_recall)
    new_th = np.array(new_th)
    save_precision_recall(new_precision, new_recall, new_th, clf)
    plt.plot(recall, precision, label=set_name)

    # with open('37class.csv', "w",newline='') as csvfile:
    #     writer=csv.writer(csvfile)
    #     for i in range(len(y_test)):
    #         if (y_test[i]==36):
    #             writer.writerow(pred_score[i])

    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[2] = roc_aupr_score(y_one_hot, pred_score, average='macro')
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    #result_all[4] = roc_auc_score(y_one_hot, pred_score, average='macro')
    result_all[5] = f1_score(y_test, pred_type, average='micro')
    result_all[6] = f1_score(y_test, pred_type, average='macro')
    result_all[7] = precision_score(y_test, pred_type, average='micro')
    # result_all[7],result_all[9]=self_metric_calculate(y_one_hot,pred_one_hot)
    result_all[8] = precision_score(y_test, pred_type, average='macro')
    result_all[9] = recall_score(y_test, pred_type, average='micro')
    result_all[10] = recall_score(y_test, pred_type, average='macro')
    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
        result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                          average=None)
        #result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
        #                                 average=None)
        result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                    average='binary')
        result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                           average='binary')
        result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                        average='binary')
    return [result_all, result_eve]


def save_precision_recall(precision, recall, th, set_name):
    workBook = xlsxwriter.Workbook(set_name + ".xlsx")
    workSheet = workBook.add_worksheet()
    workSheet.write("A1", "Precision")
    workSheet.write("B1", "Recall")
    workSheet.write("C1", "Threshold")
    workSheet.write_column("A2", precision)
    workSheet.write_column("B2", recall)
    workSheet.write_column("C2", th)
    workBook.close()


def self_metric_calculate(y_true, pred_type):
    for i in range(len(y_true)):
        print("y_true=", y_true[i])
        print("pred_type=", pred_type[i])
    y_true = y_true.ravel()
    y_pred = pred_type.ravel()
    print("\n")
    print("After ravel")
    for i in range(len(y_true)):
        print(y_true[i], '')
    print("\n")
    for i in range(len(y_pred)):
        print(y_pred[i], '')
    print("\n")
    print("y_true shape and y_pred shape are", y_true.shape, y_pred.shape)
    print("\n")
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))
    y_true_c = y_true.take([0], axis=1).ravel()
    y_pred_c = y_pred.take([0], axis=1).ravel()
    for i in range(len(y_true_c)):
        print("true", y_true_c[i], "pred", y_pred_c[i])
    print("\n")
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for i in range(len(y_true_c)):
        if (y_true_c[i] == 1) and (y_pred_c[i] == 1):
            TP += 1
        if (y_true_c[i] == 1) and (y_pred_c[i] == 0):
            FN += 1
        if (y_true_c[i] == 0) and (y_pred_c[i] == 1):
            FP += 1
        if (y_true_c[i] == 0) and (y_pred_c[i] == 0):
            TN += 1
    print("TP=", TP, "FN=", FN, "FP=", FP, "TN=", TN)
    return (TP / (TP + FP), TP / (TP + FN))


def multiclass_precision_recall_curve(y_true, y_score):
    y_true = y_true.ravel()
    y_score = y_score.ravel()
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_score.ndim == 1:
        y_score = y_score.reshape((-1, 1))
    y_true_c = y_true.take([0], axis=1).ravel()
    y_score_c = y_score.take([0], axis=1).ravel()
    precision, recall, pr_thresholds = precision_recall_curve(y_true_c, y_score_c)
    return (precision, recall, pr_thresholds)


def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(precision, recall, reorder=True)

    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)


def drawing(d_result, contrast_list, info_list):
    column = []
    for i in contrast_list:
        column.append(i)
    df = pd.DataFrame(columns=column)
    if info_list[-1] == 'aupr':
        for i in contrast_list:
            df[i] = d_result[i][:, 1]
    else:
        for i in contrast_list:
            df[i] = d_result[i][:, 2]
    df = df.astype('float')
    color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
    df.plot.box(ylim=[0, 1.0], grid=True, color=color)
    plt.title(info_list[0])
    plt.xlabel(info_list[1])
    plt.ylabel(info_list[2])
    plt.show()
    return 0


def save_result(feature_name, result_type, clf_type, result, droprate):
    with open(feature_name + '_' + result_type + clf_type + str(droprate) + '.csv', "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0


dnn = DNN()
dnn.summary()
seed = 0
CV = 5
# event_num=65
interaction_num = 10
conn = sqlite3.connect("event.db")
df_drug = pd.read_sql('select * from drug;', conn)  # 药物特征数据库
df_event = pd.read_sql('select * from event_number;', conn)  # 反应类型名称数据库
df_interaction = pd.read_sql('select * from mirrow1;', conn)  # 反应类型数据库
# feature_matrix,label_matrix,event_num=prepare(df_drug,df_event,df_interaction,feature_list,vector_size,interaction_num)

label_matrix = np.loadtxt('label_matrix')
feature_list = ['smile','target','enzyme']
# feature_list=['target']
for feature in feature_list:
    set_name = feature + '+'
set_name = set_name[:-1]
# clf_list=['GBDT','RF','KNN','LR']
clf_list = ['DNN']
result_all = {}
result_eve = {}
all_matrix = []
# =============================================================================
# for i in feature_list:
#     print (i)
#     matrix=np.loadtxt(i+'_matrix')
#     all_matrix.append(matrix)
# =============================================================================
# feature_matrix=np.loadtxt('feature_matrix')
#workBook = xlsxwriter.Workbook("feature_matrix.xlsx")
kf = KFold(n_splits=5,shuffle=True,random_state=seed)

for feature in feature_list:
    print(feature)
    new_feature, new_label, event_num,record_label = prepare(df_drug, df_event, df_interaction, [feature], vector_size,
                                                interaction_num)
    all_matrix.append(new_feature)

#workBook.close()


plt.xlabel("Recall")
plt.ylabel("Precision")

# plot_model(DNN(),to_file="model.png",show_shapes=True)


start = time.clock()

clf='DNN'
count=0
all_1=np.zeros((11,11))
each_1=np.zeros((65,6))
all_2=np.zeros((11,11))
each_2=np.zeros((65,6))
for train_drug, test_drug in kf.split(range(572)):
    count=count+1
    print(count)
    all_result1, each_result1,all_result2,each_result2 = cross_validation(all_matrix, new_label,clf,event_num, seed, CV,
                                            set_name,train_drug,test_drug,record_label)  # 为了反应数把label_matrix改成了new_label
    all_1+=(all_result1)/5
    all_2+=(all_result2)/5
    each_1+=(each_result1)/5
    each_2+=(each_result2)/5
        # =============================================================================
        #     save_result('all_nosim','all',clf,all_result)
        #     save_result('all_nosim','eve',clf,each_result)
        # =============================================================================

save_result("task1","allearly",clf,all_1,droprate)
save_result("task1","eachearly",clf,each_1,droprate)
save_result("task2","allearly",clf,all_2,droprate)
save_result("task2","eachearly",clf,each_2,droprate)

print("time used:", time.clock() - start)

# plt.title("Precision-recall curve")
plt.show()

# new_feature,new_label,event_num=prepare(df_drug,df_event,df_interaction,feature_list,vector_size,i)

# =============================================================================
# info_list=['Integrate_contrast','Integrate_type','aupr']
# drawing(result_eve,['smile','target','enzyme','pathway','feature','all'],info_list)
# info_list=['Integrate_contrast','Integrate_type','auc']
# drawing(result_eve,['smile','target','enzyme','pathway','feature','all'],info_list)
#
# =============================================================================
