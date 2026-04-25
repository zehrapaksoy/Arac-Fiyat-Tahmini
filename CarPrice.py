# Gerekli kütüphaneleri yükledik, çünkü veri işleme, modelleme ve görselleştirme için ihtiyaç duyacağım araçlar bunlar.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Veri setini içe aktardık. Bu, araç fiyatlarıyla ilgili bir veri seti.
car_prices = pd.read_csv('CarPrice_Assignment.csv')

# Veri setindeki kategorik ve sayısal değişkenleri ayırdık. Hedef değişken olan 'price' sayısal değişkenlerden çıkarttık.
categorical_features = car_prices.select_dtypes(include=['object']).columns.tolist()
numerical_features = car_prices.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features.remove('price')  # Tahmin etmeye çalıştığımız 'price' sütununu ayrı tutuyorum.

# Ön işleme için bir pipeline oluşturduk. Sayısal verileri ölçeklendirmek ve kategorik verileri OneHotEncoder ile dönüştürmek için gerekli adımları tanımladık.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Eğitim ve test setlerini ayırdık. Veri setinin %80'ini eğitim, %20'sini test için kullanmayı tercih ettik.
X = car_prices.drop('price', axis=1)
y = car_prices['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modellerin her biri için bir pipeline oluşturduk. Hyperparametre araması yapmam gereken modeller için RandomizedSearchCV kullandık.
model_pipelines = {
    'SVM': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomizedSearchCV(SVR(),
                                   param_distributions={'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]},
                                   scoring='r2', cv=5, n_iter=4, random_state=42))
    ]),
    'k-NN': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomizedSearchCV(KNeighborsRegressor(),
                                   param_distributions={'n_neighbors': [3, 5, 7, 9]},
                                   scoring='r2', cv=5, n_iter=4, random_state=42))
    ]),
    'Linear Regression': Pipeline([
        ('preprocessor', preprocessor),
        ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
        ('regressor', LinearRegression())
    ]),
    'Random Forest': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomizedSearchCV(RandomForestRegressor(random_state=42),
                                   param_distributions={'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30]},
                                   scoring='r2', cv=5, n_iter=9, random_state=42))
    ]),
    'Lasso': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomizedSearchCV(Lasso(max_iter=1000000, random_state=42),
                                   param_distributions={'alpha': [0.01, 0.1, 1, 10]},
                                   scoring='r2', cv=5, n_iter=4, random_state=42))
    ]),
    'Ridge': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomizedSearchCV(Ridge(max_iter=1000000, random_state=42),
                                   param_distributions={'alpha': [0.01, 0.1, 1, 10]},
                                   scoring='r2', cv=5, n_iter=4, random_state=42))
    ]),
    'Elastic Net': Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomizedSearchCV(ElasticNet(max_iter=1000000, random_state=42),
                                   param_distributions={'alpha': [0.01, 0.1, 1, 10], 'l1_ratio': [0.2, 0.5, 0.8]},
                                   scoring='r2', cv=5, n_iter=9, random_state=42))
    ])
}

# Her model için eğitimi gerçekleştirdik ve test verileriyle tahminlerde bulunduk.
results = {}
for name, pipeline in model_pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Performans metriklerini hesapladık (MSE, R², F1, Precision, Recall).
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    y_test_class = pd.cut(y_test, bins=[-np.inf, y_test.median(), np.inf], labels=["Low", "High"])  # Tahminler için hedef sınıfları oluşturuyoruz.
    y_pred_class = pd.cut(y_pred, bins=[-np.inf, y_test.median(), np.inf], labels=["Low", "High"])

    f1 = f1_score(y_test_class, y_pred_class, labels=["Low", "High"], average='weighted')
    precision = precision_score(y_test_class, y_pred_class, labels=["Low", "High"], average='weighted')
    recall = recall_score(y_test_class, y_pred_class, labels=["Low", "High"], average='weighted')

    # Sonuçları kaydediyorum.
    results[name] = {'MSE': mse, 'R²': r2, 'F1': f1, 'Precision': precision, 'Recall': recall}

    # Gerçek ve tahmin edilen değerlerin görsel karşılaştırmasını yaptık.
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    plt.xlabel('Gerçek Fiyatlar')
    plt.ylabel('Tahmin Edilen Fiyatlar')
    plt.title(f'{name} Sonuçları')
    plt.show()

# Confusion matrix ile tahmin sonuçlarının sınıflandırma başarısını görselleştirdik.
labels = np.union1d(y_test_class, y_pred_class)  # Benzersiz etiketler
cm = confusion_matrix(y_test_class, y_pred_class, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Predictions")
plt.show()

# Sonuçları bir DataFrame'de topladık ve performansı karşılaştırmak için grafikler oluşturduk.
results_df = pd.DataFrame(results).T
print(results_df)

plt.figure(figsize=(10, 5))
plt.bar(results_df.index, results_df['MSE'], color=['orange', 'purple', 'blue', 'green', 'red', 'brown', 'cyan'])
plt.xlabel('Model')
plt.ylabel('MSE')
plt.title('Model MSE Karşılaştırması')
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(results_df.index, results_df['R²'], color=['orange', 'purple', 'blue', 'green', 'red', 'brown', 'cyan'])
plt.xlabel('Model')
plt.ylabel('R²')
plt.title('Model R² Karşılaştırması')
plt.show()
