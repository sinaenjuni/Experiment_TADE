from sklearn.metrics import confusion_matrix
import numpy as np


labels = [0, 1, 2, 4, 5]


class ClassificationMetric:
    def __init__(self, target_label):
        self.target_label = target_label

        self.best_epoch = 0
        self.best_acc = 0
        self.best_acc_per_class = [0] * len(target_label)
        self.best_cm = 0

        self.best_acc_per_class_at_epoch = {'epoch': [0] * len(target_label), 'value': [0] * len(target_label)}


    # def getConfusionMatrix(self, true, pred):
    #     return confusion_matrix(true, pred, labels=self.target_label)
    # def getAccuracy(self, true, pred):
    #     return accuracy_score(true, pred)
    def calcMetric(self, epoch, true, pred):
        cm = confusion_matrix(true, pred, labels=self.target_label)
        sum = cm.sum()
        sum_per_class = cm.sum(1)
        diag = cm.diagonal()
        diag_sum = diag.sum()
        acc = diag_sum/sum
        acc_per_class = diag/sum

        if acc > self.best_acc:
            self.best_epoch = epoch
            self.best_acc = acc
            self.best_cm = cm
            for i in range(len(self.target_label)):
                self.best_acc_per_class[i] = acc_per_class[i]

        for i in range(len(self.target_label)):
            if self.best_acc_per_class_at_epoch['value'][i] < acc_per_class[i]:
                self.best_acc_per_class_at_epoch['value'][i] = acc_per_class[i]
                self.best_acc_per_class_at_epoch['epoch'][i] = epoch
        print('epoch:', epoch)
        print('acc:', acc)
        print('acc per class', acc_per_class)
        print('cm')
        print(cm)

        print('best epoch:', self.best_epoch)
        print('best acc:', self.best_acc)
        print('best acc per class:', self.best_acc_per_class)
        print('best acc per class at epoch:', self.best_acc_per_class_at_epoch)
        print('best cm:')
        print(self.best_cm)
        print("====================================")
        # print('cm', cm)
        # print('acc', acc)
        # print('acc_per_class', acc_per_classes)
        # return {'cm':cm, 'acc': acc, 'acc_per_class':acc_per_classes}

    def getBestMetric(self):
        print(self.best_epoch)
        print(self.best_cm)
        print(self.best_acc)
        print(self.best_acc_per_class)
        print(self.best_acc_per_class_at_epoch)


metric = ClassificationMetric(labels)

for i in range(3):
    y_true = (np.random.rand(10)*5).astype(int)
    y_pred = (np.random.rand(10)*5).astype(int)
    result = metric.calcMetric(i+1, y_true, y_pred)
    # print(result)
    # metric.getBestMetric()

# cm = confusion_matrix(y_true, y_pred, labels=labels)
# plot_confusion_matrix()
#
# print(cm)
# sum = cm.sum()
# sum_per_class = cm.sum(1)
# print('sum:', sum)
# print('sum per classes', sum_per_class)
# diag = cm.diagonal()
# diag_sum = diag.sum()
# acc = diag_sum/sum
# acc_per_classes = diag/sum
#
# print('acc:', acc)
# print('diag:', diag)
# print('acc per classes:',diag/sum_per_class)
#
# # cm_diag = cm.diagonal().sum()
# #
# # cm_acc = cm_diag/cm_sum
# #
# # print(cm)
# # print(cm_acc)