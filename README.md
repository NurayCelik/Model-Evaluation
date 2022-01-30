### Model Evaluation (Model Değerlendirmesi) 

mAP(mean average precision)

AP(average precision)

IoU(Intersection Over Union)

Recall

Precision

F1 Score

mAP Histogram


We write our own yolo weight and cfg files to the project. We organize the Labels array according to their class names.
We add 0 to TP, FP, TN, FN, sumRecall , sumPrecis, sum_F1_Score arrays as their class names.

Sample outputs after running the project:

ground_label: bukulme

precision_label: ezilme

annotas/frame_000005.PNG IoU: 0.9483

TP : [0, 0, 4, 0, 6, 0]

FP : [0, 0, 1, 0, 0, 0]

FN : [1, 2, 4, 7, 2, 0]

TN : [0, 0, 0, 0, 0, 0]

RECALL : 49.905%

PRECISION : 55.361%

F1_SCORE : 50.270%


![Figure 2022-01-05 113921](https://user-images.githubusercontent.com/29830733/148188313-5cb9bdc3-b0ea-45cb-a83e-3efe128fe3c5.png)
![Figure 2022-01-05 113935](https://user-images.githubusercontent.com/29830733/148188448-9c81b016-7bdd-4619-b5f1-069cfe386198.png)
![Figure 2022-01-05 113927](https://user-images.githubusercontent.com/29830733/148188565-53a50a3b-1ce4-46bf-96b7-ee549db3e045.png)

![Figure 2022-01-05 114002](https://user-images.githubusercontent.com/29830733/148190882-2cd040d7-93b7-47cd-811b-c61cfe3cbd14.png)
![Figure 2022-01-05 113956](https://user-images.githubusercontent.com/29830733/148190938-08f69366-1e4b-43ae-b138-0f51b2fa1d11.png)
![Figure 2022-01-05 114013](https://user-images.githubusercontent.com/29830733/148190667-1240ac52-e868-49f2-a7f0-54b0a213b29d.png)

![Figure 2022-01-05 114021](https://user-images.githubusercontent.com/29830733/148190452-55264106-89b0-49ed-811f-504b6dfd1376.png)
![Figure 2022-01-05 114017](https://user-images.githubusercontent.com/29830733/148190512-b49c2e27-adbb-4e44-a424-b049b32b1bae.png)
![Figure 2022-01-05 114025](https://user-images.githubusercontent.com/29830733/148190607-814fba06-d696-4e39-8dd3-4a0b0d0fdaa7.png)

