from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Etape 1: Creation et affichage de DataSet
X = np.array([[1], [2], [3], [4], [5], [5.1], [5.5], [6], [7], [8], [9]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
plt.grid()
plt.scatter(X, y)
plt.show()
# Etape 3 Création de modèle
model = LogisticRegression()
model.fit(X, y)
# Etape 4. afficher le coeffecient slope w1 et le bias w0
w1 = model.coef_[0]  # est une liste qui contient les coeffients
w0 = model.intercept_
print(w1, w0)
# Etape 5. Prédiction
y_pred = model.predict(X)
# Afficher la probalité associé au modèle
proba = model.predict_proba(X)
# Afficher les classes qu'on a
classes = model.classes_
print(classes)
# On affiche la fontion de décision c--à-d la fonction logit
# f(z)=sigmoid(z)=1/(1+e**-(f(z))
DF = model.decision_function(X)
DF_ = X*w1+w0
# Evaluation
print(confusion_matrix(y, y_pred))
