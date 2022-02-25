import numpy as np

class AccPerCls:
    def __init__(self):
        self.label = np.array([])
        self.pred = np.array([])

    def appendLableANDPred(self, label, pred):
        self.label = np.append(self.label, label.numpy())
        self.pred = np.append(self.pred, pred.numpy())

    def getAccPerCle(self):
        # print(label, pred)
        self.label = self.label.astype(np.int)
        self.pred = self.pred.astype(np.int)

        result = []
        unique, counts = np.unique(self.label, return_counts=True)
        match = (self.label == self.pred)

        print("ID", unique, "개수", counts)
        print("정답", self.label)
        print("예측", self.pred)
        print('일치', match)
        for unique_, counts_ in zip(unique, counts):
            print(f'label: {unique_} '
                  f'match: {match[self.label==unique_].sum()} '
                  f'count: {counts_} '
                  f'accuracy per class: {match[self.label==unique_].sum()/counts_}')

            result.append({"label": unique_,
                           "match": match[self.label==unique_].sum(),
                           "count": counts_,
                           "accuracy_per_class": match[self.label==unique_].sum()/counts_})
        print(f'accuracy: {match.mean()}')

        return {"per_class": result, "accuracy": match.mean()}

    def flush(self):
        self.label = np.array([])
        self.pred = np.array([])
