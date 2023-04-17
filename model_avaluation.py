'''
Model Evaluation (Model Değerlendirmesi) 
mAP(mean average precision)
AP(average precision)
Intersection Over Union(iou)
Recall
Precision
F1 Score

'''

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np
import os



detection_List = []
thresh_iou = 0.5
confidences_tresh = 0.25
mAPList = []
classCount = 6
labels = ["person","car","bicycle","laptop", "camera","phone"]
histList = [] 
   

def addMapList(label_id, precision, recall, F1_score):
    '''

    Parameters : label_id, precision, recall, F1_score
    
    Returns
    -------
    mAPList : global dizi
        
    addMapList fonksiyonu, mean_average_precision fonksiyonundan çağırılır.  
    showMAP fonksiyonuna global mAPList dizisini gönderir. Bu dizi
    her image detectionından sonra elde edilen 
    label_id, precision, recall, F1_score değerlerinin toplam ve net sonucunu
    hesaplatmak mAPList dizine kaydeder.
    '''
    mAPList.append([label_id, precision,recall,F1_score]) 
    return mAPList   

def addHistList(recall, precision,f1_score):
    '''
    Parameters : recall, precision,f1_score
    
    Returns
    -------
    histList : global dizi
        
    addHistList fonksiyonu, showMAP fonksiyonundan çağırılır.  
    showHistogram fonksiyonuna global histList dizisini gönderir. 
    Her image detectionından sonra elde edilen 
    recall, precision,f1_score değerleri grafiksel gösterim için histList dizisine kaydeder.
    
    '''
    
    histList.append([recall, precision,f1_score]) 
    return mAPList 
    
def mean_average_precision(detection_List = []):
# =============================================================================
# Eger bir detection icin IoU>=0.5 ise True Positive (TP)
# Eger bir detecion icin IoU<0.5 ise False Positive (FP)
# Eger bir ground truth var detection yok ise False Negative (FN)
# Eger bir detection var ground truth yok ise yine True Negative (TN)
# =============================================================================   


    TP=[0,0,0,0,0,0]
    FP=[0,0,0,0,0,0]
    TN=[0,0,0,0,0,0]
    FN=[0,0,0,0,0,0]
 
        
                

    '''
    -> mAP i hesaplamak için ön işlemler için mean_average_precision fonksiyonu yazıldı. 
    -> Parametre olarak detection_List alındı, output olarak da 
    addMapList fonksiyonuna i, precision, recall, F1_score verileri gönderildi.
    
    -> mean_average_precision fonksiyonun çalışması:
    detection_List den gelen iou değeri, predict(detect image) labelid(class id) ve ground truth labelid değerleri kıyaslanarak
    TP, FP, FN hesaplandı. Bunun için class(label) -> (labels = ["person","car","bicycle","laptop", "camera","phone"]) 
    sayısı kadar indise sahip TP, FP, TN, FN dizileri oluşturuldu.
    Bu örneğin TP dizisinde her bir indis class id yi temsil ediyor. 
    
    1. koşul 
    predict label ve ground truth label eşit ve de iou değeri 0 ve 1 aralığında olma koşulunda;
    eğer iou değeri thresh_iou değerine büyük veya eşit ise truth label indisine eşit TP dizisinde de 
    aynı indis değeri 1 artırıldı,
    eğer iou değeri thresh_iou değerinden küçükse truth label indisine eşit FP dizisinde de 
    aynı indis değeri 1 artırıldı,
    2. koşul 
    predict label ve ground truth label eşit değil veya iou değeri 1'den büyük ya da eşitise  veya iou değeri 0'dan küçük ya da eşit olma koşulunda;
    ground truth labelid var predict labelid yok ise FN 1 artırıldı
    ground truth labelid yok predict labelid var ise TN 1 artırıldı
    ground truth labelid var predict labelid var ve eşit değilse ise FN 1 artırıldı
    3. koşul
    bunların dışında birşey yapma
    
    '''
   
    #detection_List([image_file_path,iou,tr_label_id, pr_label_id]) 
   
        
    for i in range(len(detection_List)):
        tr_class_name = labels[detection_List[i][2]]
        pr_class_name = labels[detection_List[i][3]]
        #print("tr_class_name : {}".format(tr_class_name))
        #print("pr_class_name : {}".format(pr_class_name))
        #print("truth label id: {}".format(detection_List[i][2]))
        #print("predict label id: {}".format(detection_List[i][3]))
        
        if detection_List[i][2] == detection_List[i][3] and (detection_List[i][1] >0 and detection_List[i][1] <=1):
            if detection_List[i][1] >= thresh_iou:
                TP[detection_List[i][2]] += 1
            elif detection_List[i][1] < thresh_iou:
                 FP[detection_List[i][2]] += 1
                
            else:
                 pass
        
        elif detection_List[i][2] != detection_List[i][3] and detection_List[i][1] >= 0 or detection_List[i][1] <=0:
            if detection_List[i][2] >= 0 and detection_List[i][3] <= -1:  
                FN[detection_List[i][2]] += 1
            elif detection_List[i][2] >= 0 and detection_List[i][3] >=0 :
                FN[detection_List[i][2]] += 1
            elif detection_List[i][2] <= -1 and detection_List[i][3] >=0:
                TN[detection_List[i][3]] += 1
            else:
                pass
        else:
          pass
      
    print("TP : {}".format(TP))
    print("FP : {}".format(FP))
    print("FN : {}".format(FN))
    print("TN : {}".format(TN))
    
    
    '''
     labels dizisinin boyutu kadar (TP, FP, FN aynı indis değerinde) döngü içerisinde;
     her bir indis için (classId => labelId ye göre) precision, recall, F1_score değerleri hesaplatıldı,
     döngü içerinde hesaplanan bu değerler addMapList(i, precision, recall,F1_score) fonksiyonuna gönderildi
     
     '''
    for i in range(len(labels)):
        
        try:
            precision = float((TP[i])/(TP[i] + FP[i]))
            
            #print("precision : {}".format(precision))
        except ZeroDivisionError:
            precision =  0
        
        try:
            recall = float((TP[i])/(TP[i] + FN[i]))
            #print("recall : {}".format(recall))
        except ZeroDivisionError:
            recall =  0
         
        try:
            F1_score = float((precision * recall)/((precision + recall)/2))
            #print("F1_score : {}".format(F1_score))
        except ZeroDivisionError:
            F1_score =  0
        
        addMapList(i, precision, recall,F1_score) 
     
   
   
   
   
#%%

def showMAP(mAPList=[]):
    '''
    -> showMAP fonksiyonu bütün classlara ait precision, recall,F1_score değerlerinin toplatılıp class sayısına bölünerek 
    kesin sonuçların hesaplandığı fonksiyondur.
    -> showMAP fonksiyonu parametre olarak mAPList dizisini alır. Output olarak RECALL, PRECISION, F1_SCORE için son değeri verir.
    Aynı zamanda detection boyunca gelen verileri recList, precList, f1_scoreList dizilerine eklyerek bu listedeki veriler için recall
    precision, f1 score değerlerinin grafiksel gösterimini yapar. 
    
    -> showMAP fonksiyonunun çalışması: addMapList fonksiyonu mAPList dizini döndürüyor. showMAP fonksiyonu, mAPList dizisini parametre 
    alarak çağırdiğimizda çalışan bir fonksiyom.
    
    addMapList fonksiyonu mAPList dizisini (i, precision, recall,F1_score => i= classId ) değerleri her bir image detection ından sonra diziye kaydediyor.
    showMAP fonksiyonu bütüm imageler detection edildikten sonra çağırılıyor. Bütün her bir image e ait classid, precision, recall, F1_score değerleri artık elimizde.
    
    sumRecall, sumPrecis, sum_F1_Score dizileri yine classların indexi boyutunda oluşturuldu.
    mAPList dizisinin boyutunda döngü oluşturuldu.
    Bu döngüde class idsine göre recall, precision ve 
    f1 scorelar sumRecall, sumPrecis, sum_F1_Score dizilerine kendi sırasına göre eklendi.
    RECALL, PRECISION,F1_SCORE değerleinin grafiksel gösterimi için addHistList fonksiyonuna kaydedildi.
    
    '''
    # mAPList -> label_id, precision,recall,F1_score
    sumRecall = [0,0,0,0,0,0]
    sumPrecis = [0,0,0,0,0,0]
    sum_F1_Score = [0,0,0,0,0,0]
    RECALL = 0
    PRECISION = 0
    F1_SCORE = 0
    
    #print("label_id, precision, recall, F1_score {}".format(mAPList))
    for i in range(len(mAPList)):
        sumRecall[mAPList[i][0]] += mAPList[i][2] #recall
        #print("class : {} -> sumRecall : {}".format(labels[mAPList[i][0]], sumRecall[mAPList[i][0]]))  
        
        sumPrecis[mAPList[i][0]] += mAPList[i][1] #precision
        #print("class : {} -> sumPrecis : {}".format(labels[mAPList[i][0]], sumPrecis[mAPList[i][0]]))  
        
        sum_F1_Score[mAPList[i][0]] += mAPList[i][3] #F1_score
        #print("class : {} -> sum_F1_Score : {}".format(labels[mAPList[i][0]], sum_F1_Score[mAPList[i][0]]))  
    
    '''
    
    Dizi olarak eklenen sumRecall, sumPrecis, sum_F1_Score yine classIdlerin boyutunda döngüde 
    gerçek RECALL, PRECISION, F1_SCORE değerlerini bulmak için toplandı.
    
    '''      
    # AP değerleri   
    for i in range(len(labels)):
        RECALL += sumRecall[i]
        PRECISION += sumPrecis[i]
        F1_SCORE += sum_F1_Score[i]
      
    '''
    mAP değerleri
    RECALL, PRECISION, F1_SCORE değerlerinin net değerini bulmak için de 
    class dizisinin boyutuna bölündü
    '''  
    RECALL = RECALL / len(labels)
    PRECISION = PRECISION / len(labels)
    F1_SCORE = F1_SCORE / len(labels)
    
    print("RECALL : {:.3f}%".format(RECALL*10))
    print("PRECISION : {:.3f}%".format(PRECISION*10))
    print("F1_SCORE : {:.3f}%".format(F1_SCORE*10))    
   
    addHistList(RECALL, PRECISION,F1_SCORE)
    
#%%

def showHistogram(histList = []):
    
    ''' 
   
    Grafiksel Gösterimi Matplotlib Kütüphanesi 
    Grafik oluşturmak için recList, precList, f1_scoreList dizilerine kaydedildi.
   '''
    recList = []
    precList = []
    f1_scoreList = []
    
    for i in range(len(histList)):
        recList.append(histList[i][0]) # recall
        precList.append(histList[i][1]) # precision
        f1_scoreList.append(histList[i][2]) # f1_score
        
    # print("recList : {}".format(recList))   
    # print("precList : {}".format(recList))   
    # print("f1_scoreList : {}".format(recList))  
    
    y = recList
    y.sort()
    x1 = precList
    x1.sort
    z = f1_scoreList
    z.sort()
    #print(z)
     
    # linear
    plt.scatter(x1,y)
    plt.plot(x1,y, 'b')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Histogram of mAP')
    plt.grid(True)
    plt.show()
    
    # linear
    plt.plot(x1,y, 'b')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Histogram of mAP')
    plt.grid(True)
    plt.show()
    
    # linear
    plt.plot(z,'b')
    plt.title('F1 Score')
    plt.grid(True)
    plt.show()
     
    # symmetric log
    plt.plot(x1, y)
    plt.yscale('log')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('mAP')
    plt.grid(True)
    
   
       
    # grafikten bölüm alma
    # set_xlim([0,2]) x ekseninde
    # set_ylim([0,2]) y ekseninde
    f = plt.figure()
    axes = f.add_axes([0.1,0.1,0.7,0.7])
    axes.plot(x1,y,color='purple',linewidth=2, linestyle='-')
    axes.set_xlabel('precision')
    axes.set_ylabel('recall')
    axes.set_title('Histogram of mAP')
    axes.set_xlim([0,1])
    axes.set_ylim([0,1])
    plt.show()
     
    ''' 
    
    Grafiksel Gösterimi Seaborn Kütüphanesi 
    
    ''' 
     
    sns.set(style="white", palette="Blues", font="sans-serif", 
    font_scale=1.1)
    sns.distplot(x1,color="g",hist=False, kde_kws={'shade':True},axlabel="precision") 
    plt.tight_layout()
    plt.show()
    
    sns.set(style="white", palette="Blues", font="sans-serif", 
    font_scale=1.1)
    sns.distplot(y,color="b",hist=False, kde_kws={'shade':True},axlabel="recall")
    plt.tight_layout()
    plt.show()
    
    sns.set(style="white", palette="Blues", font="sans-serif", 
    font_scale=1.1)
    sns.distplot(z,color="b",hist=False, kde_kws={'shade':True},axlabel="f1 score")
    plt.tight_layout()
    plt.show()
    
    sns.set(style="whitegrid", palette="Blues", font="sans-serif", 
    font_scale=1.5)
    sns.distplot(x1,color="m",axlabel="precision")
    plt.tight_layout()
    plt.show()
    
    sns.set(style="whitegrid", palette="Blues", font="sans-serif", 
    font_scale=1.5)
    sns.distplot(y,color="m",axlabel="recall")
    plt.tight_layout()
    plt.show()
    
    sns.set(style="whitegrid", palette="Blues", font="sans-serif", 
    font_scale=1.5)
    sns.distplot(z,color="m",axlabel="f1 score")
    plt.tight_layout()
    plt.show()
    
    
#%%
def intersection_over_union(box1, box2): 
    '''
    -> intersection_over_union fonksiyonu, nesne dedektörünün doğruluğunu ölçmek için kullanılan 
    bir değerlendirme metriğidir. ground-truth bounding boxes (nesnemizin görüntüde nerede olduğunu 
    belirten test kümesinden hazır olarak bulunan (burada herbir image için verileri txt dosyasında bulunan) 
    sınırlayıcı kutular) ve predicted bounding boxes; objec detectiondan tahmin edilen sınırlayıcı 
    kutular(predicted bounding boxes). Bu iki sınırlayıcı kutuya sahip olduğumuz sürece, intersection over union(iou) uygulayabiliriz.
    
    -> intersection_over_union(box1, box2),  imageLabelling fonksiyonu içerisinden çağırıliyor.Parametre olarak
    ground-truth bounding boxes için box1 ve predicted bounding boxes için box2 iki diziyi alır. 
    Output olarak iou sonucunu döndürür.
    
    boxA = (tr_start_x,tr_start_y,tr_end_x,tr_end_y) iki diziyi çalıştırır.
    detection sonucudan elde edilen  boxB = (start_x, start_y, end_x, end_y)
    box1 ve box2 dizisi değişkenlere atandı.
    sırasıyla iki değişkenlerin 1 ve 2. sinin maximum olanı ve 3. ve 4. nün minumum değerleri alındı.
    geniş dikdörtgenin genişliği ve yüksekiğini hesaplamak için mutlak değeri alındı
    negatif sonuçları önledik
    box1 ve box2 için iki dikdörtgenin alanı hesaplandı
    Geniş dikdörgenden, iki dikdörgenden çıkarıldı kesişimi hesaplamak için
    iou değeri blundu
    
    '''
    #print("box1 : {}".format(box1))
    #print("box2 : {}".format(box2))
    
    x1,y1,x2,y2 = box1
    x3,y3,x4,y4 = box2
    
    x_area1 = max(x1,x3)
    y_area1 = max(y1,y3) 
    x_area2 = min(x2,x4)
    y_area2 = min(y2,y4)
    
    '''
    
    '''
    
    width_area = abs(x_area2 - x_area1)
    height_area = abs(y_area2 - y_area1)
    
    #dikdörtenin alanı  hesaplandı.
    area_size = width_area * height_area
   
    width_box1 = abs(x2 - x1)
    height_box1 = abs(y2 - y1)
    width_box2 = abs(x4 - x3)
    height_box2 = abs(y4 - y3)
    
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2
    
    area_union = area_box1 + area_box2 - area_size
    iou = area_size / area_union

    return iou
#%%
def addedIouList(image_file_path, iou, tr_label_id, pr_label_id):
    '''
    addedIouList fonksiyonu mAP i hesapalmak için mean_average_precision fonksiyonuna parametre olarak
    gönderilmek üzere detection_Listesini kaydeden fonksiyondur.
    imageLabelling fonksiyonundan çağırılır ve image_file_path, iou, tr_label_id, pr_label_id verileri gönderilir.
    Çıktı olarak detection_List dizisini üretir ve gelen verileri aynen iletir.
    '''
    ### bütün resim dosyalarına ait iou değerlerini, mAP değerini bulmak için detection_List e ekliyoruz.
    return detection_List.append([image_file_path,iou,tr_label_id, pr_label_id]) 
   
#%%
def detectionFile(img):
    '''
    -> detectionFile fonksiyonu imageLabelling fonksiyonundan çağırılan ve fonksiyondan 
    gelen imageleri detection dan geçiren bir fonksiyondur. Parametre olarak imageLabelling de 
    okunan image i alır.
    Output olarak bounding dizisi döndürür. Bu dizi, detect edilen bounding box ın verilerini
    ve tespit edilen class id ye ait verileri içerir.
    imageLabelling fonksiyonundan çağırılan ve çalıştırılan fonksiyondur. Imageler detection edilir.
    '''
    
    ### İmageleri object_detection ediyoruz. 
    
    img_width = img.shape[1]
    img_height = img.shape[0]
    
    img_blob = cv2.dnn.blobFromImage(img, 1/255, (608,608),(0, 0, 0),swapRB=True, crop=False)
    
    model = cv2.dnn.readNetFromDarknet("obj.cfg","obj.weights")
    
    layers = model.getLayerNames()
    output_layer = [layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]
    
    model.setInput(img_blob)
    detection_layers = model.forward(output_layer)
    
    
    ids_list = []
    boxes_list = []
    confidences_list = []
    bounding_box = []
    bounding = []
    
  
    
    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            
            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]
            
            if confidence > confidences_tresh:
                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array([img_width,img_height,img_width,img_height])
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")
                    
                     
                start_x = int(box_center_x - (box_width/2))
                start_y = int(box_center_y - (box_height/2))
                
                              
                ############## NON-MAXIMUM SUPPRESSION ###################
                
                ids_list.append(predicted_id)
                confidences_list.append(float(confidence))
                boxes_list.append([start_x, start_y, int(box_width), int(box_height)])
                
                
    
    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, confidences_tresh, thresh_iou) 
    for max_id in max_ids:
        
        max_class_id = max_id[0]
        box = boxes_list[max_class_id]
        
        start_x = box[0] 
        start_y = box[1] 
        box_width = box[2] 
        box_height = box[3] 
        predicted_id = ids_list[max_class_id]
        label = labels[predicted_id]
        confidence = confidences_list[max_class_id]
      
    ################################################
                
        end_x = start_x + box_width
        end_y = start_y + box_height
              
        txt_label = "pr: {}/{:.2f}%".format(label, confidence*100)
        #print("predicted object {}".format(label))
        bounding=[start_x, start_y, end_x, end_y,txt_label,predicted_id] 
        print(bounding)
       
    return bounding

#%%
def getGroundTruthVal(filename):
    '''
    

    -> getGroundTruthVal fonksiyonu, imageLabelling fonksiyonun içerisinden çağırılan 
    ve filename parametresi alan bir fonksiyondur. Output olarak truthBox dizisini üretir
    ground-truth bounding boxes verilerini txt dosyasından alır ve truthBox dizisine ekler.
    ----------
    -> getGroundTruthVal fonksiyonu dosyayı okuyup her satır için diziyi truthBoxa kaydeder. 
    -------


    '''
    try:
       truthBox = []
       with open(filename) as infile:
            for line in infile:
                #line.strip()
                inner_list = (elt.strip() for elt in line.split(','))
                truthBox.append(line)
            #print("filename : {} inner_list : {}".format(filename,truthBox))
            infile.close()
       
       return truthBox 
    except IOError:
         print("File not found")
         return "File not found"

#%%
def imageLabelling(image_file_path):
    '''
    

    imageLabelling fonksiyonu işlemleri gerçekleştiren ana fonksiyondur.
    Her bir image e ait dosyayı alır ve çalıştırır
    imageLabelling fonksiyonu içerisinden detectionFile,getGroundTruthVal,intersection_over_union,
    addedIouList fonksiyonları çalıştırılır ve gelen verilerle işlemler yapılır. 
    

    '''
    boxA = []
    boxB = []
    boxes_list = []  
    truthBox =[]
    img = cv2.imread(image_file_path)
    img = cv2.resize(img,(608,608),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
   
    # predicted bounding boxes değerlerini almak için detectionFile fonksiyonu çağrıliyor
    # boxes_list -> start_x, start_y, end_x, end_y,txt_label,label
    boxes_list = detectionFile(img)
    # print("boxes_list: {}".format(boxes_list))
       
    
    iou = 0
   
    # Ayni image e ait annotation(ground-truth bounding boxes) bilgilerinin olduğu 
    # txt dosyasını açıyoruz getGroundTruthVal fonksiyou ile ve ground-truth bounding boxes bilgilerini alıyoruz.
            
    content = image_file_path.split(".PNG")
    fileName = content[0]+".txt"
    truth = getGroundTruthVal(fileName)
    
    count = 0
    nBox = []
    label = 0
    cx = 0 
    cy = 0 
    w =0 
    h=0
    
    # truthBox da aynı image için birden çok ground-truth bounding boxes oluşabilir
    # Birden çok ground-truth bounding boxes varsa döngüyle sonuçları alıyoruz
    # Gelen string formatında veriyi boşluklardan ayrırarak gerekli değişkenlere atıyoruz
    # Eğer hiç veri yoksa hata çıkmasını önlemek için ground-truth bounding boxes için 
    # önemsiz değerler veriyoruz mesela if i == '' or i == ' ': koşuluyla
                  
    
    for truthBox in truth:
        inner_list = (elt for elt in truthBox.split(" "))
        for i in inner_list:
            print("i :::{}".format(i))
            if count == 0:
                if i == '' or i == ' ':
                  label = -1
                else:
                  label = int(i)
            elif count == 1 :
                if i == '' or i == ' ':
                  cx = 0
                else:
                  cx = float(i)
            elif count == 2:
                if i == '' or i == ' ':
                  cy = 0
                else:
                 cy = float(i)
            elif count == 3 :
                if i == '' or i == ' ':
                  w = 0
                else:
                 w = float(i)
            elif count == 4:
                if i == '' or i == ' ':
                  h = 0
                else:
                  h = float(i)
            else:
                pass
            count += 1
           
        count = 0  
        nBox = [label,cx,cy,w,h]  
            
        print("nBox :: {}".format(nBox))
        #print("f : {}".format(f))
        
        #txt dosyasından alınan ground-truth bounding boxes değerleri floata, integere cast
        # box_center_x, box_center_y, box_width, box_height 
         
        f = np.array(nBox)
        truth_label_index = np.array(nBox[:1],int)
        #print("truth_label_index: {}".format(truth_label_index))
        truth_rectangle_point_list = np.array(nBox[1:],float)
        #print("Truth f : {}".format(truth_rectangle_point_list))
        f_tr = [i * 608 for i in truth_rectangle_point_list[0:]] ## ## txt dosyasından alınan ground truth değerleri detection ekran ölçüsüne çevriliyor. 
        
        ## txt dosyasından alınan ground-truth bounding boxes değerleri rectanglın başlangıç ve bitiş moktaları için işleniyor
        tr_label = labels[int(truth_label_index)]
        #print("Truth label: {}".format(tr_label))
        tr_x = int(f_tr[0])
        tr_y = int(f_tr[1])
        tr_width = int(f_tr[2])
        tr_height = int(f_tr[3])
        tr_start_x = int(tr_x - tr_width/2)
        tr_start_y = int(tr_y - tr_height/2)
        tr_end_x = int(tr_start_x + tr_width) 
        tr_end_y = int(tr_start_y + tr_height) 
        
        print("tr_start_x : {}".format(tr_start_x))
        print("tr_start_y : {}".format(tr_start_y))
        print("tr_end_x : {}".format(tr_end_x))
        print("tr_end_y : {}".format(tr_end_y))
     
        # ground-truth bounding boxes- > boxA
        
        
        # Bir sonraki image oluşmuş ve değerleri alınmışsa 
        # yani detectiondan veri gelmişse->predicted bounding boxes oluşmuşsa-> len(boxes_list )>0:   
        print("boxes_list: {}".format(boxes_list))
        print("nBox[0] : {}".format(nBox[0]))
        
        if len(boxes_list )>0: 
            # ground-truth bounding boxes- > boxA
            boxA = (tr_start_x,tr_start_y,tr_end_x,tr_end_y)
            print("boxA : {}".format(len(boxA)))
            cv2.rectangle(img, (boxA[:2]), (boxA[2:]),	(0,191,255),1)
            gr_label = "gr: {}".format(tr_label)
            print("ground_label: {}".format(tr_label))
            cv2.putText(img,gr_label,(boxA[0]-50,boxA[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,191,255), 1)
            
           
            # predicted bounding box -> boxB
            start_x, start_y, end_x, end_y, txt_label, pr_label_id = boxes_list  
            pr_label = labels[pr_label_id]
            boxB = (start_x, start_y, end_x, end_y)
            print("boxB: {}".format(boxB))
            cv2.rectangle(img, (boxB[:2]), (boxB[2:]), (255, 0, 255), 1)
            cv2.putText(img,txt_label,(boxB[0]-50,boxB[1]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            print("precision_label: {}".format(pr_label))
            
            # intersection over union (iou) hesaplama ve değeri elde etmek için fonksiyon çağırılıyor
            iou = intersection_over_union(boxA, boxB)
            print("iou : {}".format(iou))
            cv2.putText(img, "IoU: {:.4f}".format(iou), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0),1)
            print("{} IoU: {:.4f} ".format(image_file_path, iou))
            
            # detectiondan geçen image dosyalarına ait değerleri, 
            # mAP değerini hesaplamak için addedIouList fonksiyonuyla detection_List e ekliyoruz.   
            addedIouList(image_file_path, iou, int(truth_label_index), pr_label_id)
            
        # nesne tespiti detectionda ve  groundtruth da bulunmuyorsa    
        elif len(boxes_list )<=0 and nBox[0]<0: 
            print("İmage de nesne algılanmadı. Ground truth box ve predict box değeri bulunmuyor")
            
        # object detectionda predict bbox oluşmadı fakat ground truth box varsa 
        elif len(boxes_list )<=0 and nBox[0]>=0 :
            
            # ground-truth bounding boxes- > boxA
            boxA = (tr_start_x,tr_start_y,tr_end_x,tr_end_y)
            #print("boxA : {}".format(len(boxA)))
            cv2.rectangle(img, (boxA[:2]), (boxA[2:]),	(0,191,255),1)
            gr_label = "gr: {}".format(tr_label)
            #print("ground_label: {}".format(tr_label))
            cv2.putText(img,gr_label,(boxA[0]-50,boxA[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,191,255), 1)
            
            # bütün resim dosyalarına ait değerleri, 
            # mAP değerini hesaplamak için addedIouList fonksiyonuyla detection_List e ekliyoruz.   
            addedIouList(image_file_path, 0, int(truth_label_index), -1)
            
        # object detectionda predict bbox oluştu fakat ground truth box yoksa
        elif len(boxes_list )>0 and nBox[0]<=-1: 
             # predicted bounding box -> boxB
            start_x, start_y, end_x, end_y, txt_label, pr_label_id = boxes_list  
            pr_label = labels[pr_label_id]
            boxB = (start_x, start_y, end_x, end_y)
            #print("boxB: {}".format(boxB))
            cv2.rectangle(img, (boxB[:2]), (boxB[2:]), (255, 0, 255), 1)
            cv2.putText(img,txt_label,(boxB[0]-50,boxB[1]-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            print("precision_label: {}".format(pr_label))
             # bütün resim dosyalarına ait değerleri, 
            # mAP değerini hesaplamak için addedIouList fonksiyonuyla detection_List e ekliyoruz.   
            addedIouList(image_file_path, 0,-1, pr_label_id)
        else:
            pass
            
           
            
    cv2.imshow("Detection Window", img)  
    cv2.waitKey(0)
    
#½½
    
'''
   Model evaluation işlemlerinin başlangıç bölümü
   For Döngüsüyle annotas klasöründe bulununan bütün image leri imageLabelling fonksiyonuyla 
   tek tek detect ediyoruz. mean_average_precision fonksiyonuyla mAP(mean average precision) hesaplanıyor.
   Her image çalıştıktıkan sonra döngü içerisinde showMAP fonksiyonu çalışıyor ve fonksiyon 
   mAP(mean average precision) e ait 
   Recall, Precision ve F1 Score değerlerini çıktı olarak veriyor
   Bütün resimler çalıştıktan sonra ve döngü bittikten sonra Recall, Precision ve F1 Score değerlerini grafiksel gösterimi için
   showHistogram(histList) fonksiyonu çalışıyor.
   
    '''

for detectFile in os.listdir("annotas/"):
    if detectFile.endswith(".PNG"):
        print("annotas/"+detectFile)
        imageLabelling("annotas/"+detectFile) 
        #print("detection_List : {}".formfat(detection_List))
        mean_average_precision(detection_List)
        showMAP(mAPList)  
showHistogram(histList)

#½½         

        
