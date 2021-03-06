from sklearn.metrics import confusion_matrix
import numpy as np
from utiles.pretty_confusion_matrix import pp_matrix
import pandas as pd


class ClassificationMetric:
    def __init__(self, target_label):
        self.target_label = target_label

        self.best_epoch = 0
        self.best_acc = 0
        self.best_acc_per_class = [0] * len(target_label)
        self.best_cm = 0

        self.best_acc_per_class_and_epoch = {'epoch': [0] * len(target_label), 'value': [0] * len(target_label)}

    def calcMetric(self, epoch, true, pred):
        cm = confusion_matrix(true, pred, labels=self.target_label)
        sum = cm.sum()
        sum_per_class = cm.sum(1)
        diag = cm.diagonal()
        diag_sum = diag.sum()
        acc = diag_sum/sum
        acc_per_class = diag/sum_per_class

        if acc > self.best_acc:
            self.best_epoch = epoch
            self.best_acc = acc
            self.best_cm = cm
            for i in range(len(self.target_label)):
                self.best_acc_per_class[i] = acc_per_class[i]

        for i in range(len(self.target_label)):
            if self.best_acc_per_class_and_epoch['value'][i] < acc_per_class[i]:
                self.best_acc_per_class_and_epoch['value'][i] = acc_per_class[i]
                self.best_acc_per_class_and_epoch['epoch'][i] = epoch

        ret = ""
        ret += f'epoch: {epoch}\n'
        ret += f"acc: {acc:.4f} ({epoch}), best: {self.best_acc:.4f} ({self.best_epoch})\n"
        for i in range(len(self.target_label)):
            ret += (f'class {i}: '
                  f'{acc_per_class[i]:.4f} ({epoch}), '
                  f'{self.best_acc_per_class[i]:.4f} ({self.best_epoch}), '
                  f'{self.best_acc_per_class_and_epoch["value"][i]:.4f} ({self.best_acc_per_class_and_epoch["epoch"][i]})\n')
        ret += "cm\n"
        ret += f"{cm}\n"
        ret += "Best cm\n"
        ret += f"{self.best_cm}\n"
        ret += "===================================="

        print(ret)

        return {"epoch": epoch,
                "acc": acc,
                "best_epoch": self.best_epoch,
                "best_acc": self.best_acc,
                "best_acc_per_class": self.best_acc_per_class,
                "best_acc_per_class_and_epoch": self.best_acc_per_class_and_epoch,
                "cm": cm,
                "best_cm": self.best_cm,
                "text": ret}



if __name__ == "__main__":
    labels = [0, 1, 2, 4, 5]
    metric = ClassificationMetric(labels)

    for i in range(10):
        y_true = (np.random.rand(10)*5).astype(int)
        y_pred = (np.random.rand(10)*5).astype(int)
        result = metric.calcMetric(i+1, y_true, y_pred)
        print(result['cm'])

    df_cm = pd.DataFrame(result['cm'])
    pp_matrix(df_cm)


