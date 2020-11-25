from sklearn.metrics import classification_report,confusion_matrix, f1_score, auc, roc_curve

def get_metrics(y_pred,y_true):
  cm = confusion_matrix(y_pred=y_pred, y_true=y_true)
  total = sum(sum(cm))
  accuracy = (cm[0,0] + cm[1,1])/total
  sensitivity = cm[0,0]/(cm[0,0] + cm[0,1])
  specificity = cm[1,1]/(cm[1,0] + cm[1,1])
  f1score = f1_score(y_pred,y_true)

  print("Accuracy:",accuracy)
  print("Sensitivity:",sensitivity)
  print("Specificity:",specificity)
  print("F1-Score:",f1score)
