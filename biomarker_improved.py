#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAPSAMLI ENDOMETRİOZİS BİYOMARKER ANALİZİ (Python Pipeline)
Overfitting Önleme + Feature Selection + Regularization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LassoCV, RidgeCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Try to import imblearn, but don't fail if not available
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
    print("imblearn kütüphanesi bulundu - SMOTE ve undersampling kullanılabilir")
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("imblearn kütüphanesi bulunamadı - Basit dengeleme yöntemleri kullanılacak")

# GEO dataset işleme için ek importlar
try:
    import GEOparse
    GEOPARSE_AVAILABLE = True
    print("GEOparse kütüphanesi bulundu - GEO datasetleri direkt yüklenebilir")
except ImportError:
    GEOPARSE_AVAILABLE = False
    print("GEOparse kütüphanesi bulunamadı - CSV/Excel dosyaları kullanılacak")

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EndometriosisBiomarkerAnalysis:
    def __init__(self, output_dir="results"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.analysis_name = f"endometriosis_analysis_{self.timestamp}"
        self.output_dir = Path(output_dir) / self.analysis_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Çıktı dizini: {self.output_dir}")
        
        # Feature selection parametreleri
        self.max_features = 100  # Maksimum feature sayısı
        self.min_features = 10   # Minimum feature sayısı
        self.cv_folds = 5        # Cross-validation fold sayısı
        self.cv_repeats = 3      # Cross-validation tekrar sayısı

    def load_excel_sheets(self, file_path):
        """Excel dosyasından tüm sheet'leri yükleme"""
        print(f"Yükleniyor: {file_path}")
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            print(f"Bulunan sheet'ler: {', '.join(sheet_names)}")
            
            sheets_data = {}
            for sheet_name in sheet_names:
                try:
                    data = pd.read_excel(file_path, sheet_name=sheet_name)
                    sheets_data[sheet_name] = data
                    print(f"Yüklendi: {sheet_name} - {data.shape[0]} satır, {data.shape[1]} sütun")
                except Exception as e:
                    print(f"Hata: {sheet_name} - {str(e)}")
            return sheets_data
        except Exception as e:
            print(f"Dosya yükleme hatası: {str(e)}")
            return {}

    def detect_column_names(self, data, dataset_name):
        """Sütun isimlerini otomatik tespit etme"""
        print(f"Sütun tespiti: {dataset_name}")
        print(f"Mevcut sütunlar: {list(data.columns)}")
        
        # Sütun isimlerini küçük harfe çevirme
        data.columns = data.columns.str.lower()
        
        gene_col = None
        possible_gene_cols = ['genesymbol', 'genes', 'gene_symbol', 'symbol', 'gene_name', 'genename']

        for col in data.columns:
            if col.lower() in possible_gene_cols:
                gene_col = col
                break

        if gene_col is None:
            gene_col = data.columns[0]
            print(f"Uyarı: Gen ismi sütunu bulunamadı, ilk sütun kullanılıyor: {gene_col}")
        
        # logFC sütununu tespit etme
        logfc_col = None
        possible_logfc_cols = ['logfc', 'log2fc', 'log2foldchange', 'foldchange', 'fc', 'log2']
        
        for col in possible_logfc_cols:
            if col in data.columns:
                logfc_col = col
                print(f"logFC sütunu bulundu: {logfc_col}")
                break
        
        # P-value sütununu tespit etme
        pval_col = None
        possible_pval_cols = ['pvalue', 'p.value', 'p_val', 'pval', 'p.value.adj', 'padj']
        
        for col in possible_pval_cols:
            if col in data.columns:
                pval_col = col
                print(f"P-value sütunu bulundu: {pval_col}")
                break
        
        # adj.P.Val sütununu tespit etme
        adj_pval_col = None
        possible_adj_pval_cols = ['adj.p.val', 'adj.pvalue', 'fdr', 'padj', 'qvalue', 'p.adj']
        
        for col in possible_adj_pval_cols:
            if col in data.columns:
                adj_pval_col = col
                print(f"Adjusted P-value sütunu bulundu: {adj_pval_col}")
                break
        
        return gene_col, logfc_col, pval_col, adj_pval_col

    def standardize_dgea_data(self, data, dataset_name):
        """DGEA sonuçlarını standartlaştırma"""
        print(f"Standartlaştırılıyor: {dataset_name}")
        
        # Sütun isimlerini tespit etme
        gene_col, logfc_col, pval_col, adj_pval_col = self.detect_column_names(data, dataset_name)
        
        # Standartlaştırılmış veri oluşturma
        standardized_data = pd.DataFrame({
            'Gene': data[gene_col].astype(str),
            'Dataset': dataset_name
        })
        
        if logfc_col:
            standardized_data['logFC'] = pd.to_numeric(data[logfc_col], errors='coerce')
        else:
            standardized_data['logFC'] = np.nan
            print("Uyarı: logFC sütunu bulunamadı")
        
        if pval_col:
            standardized_data['P.Value'] = pd.to_numeric(data[pval_col], errors='coerce')
        else:
            standardized_data['P.Value'] = np.nan
            print("Uyarı: P-value sütunu bulunamadı")
        
        if adj_pval_col:
            standardized_data['adj.P.Val'] = pd.to_numeric(data[adj_pval_col], errors='coerce')
        else:
            # Eğer adj.P.Val yoksa P.Value kullan
            standardized_data['adj.P.Val'] = standardized_data['P.Value']
            print("Uyarı: Adjusted P-value sütunu bulunamadı, P-value kullanılıyor")
        
        # NA değerleri temizleme
        standardized_data = standardized_data.dropna(subset=['Gene', 'logFC'])
        
        # Gen isimlerini temizleme
        standardized_data = standardized_data[
            ~standardized_data['Gene'].str.match(r'^\d+\.?\d*$', na=False)
        ]
        
        print(f"Temizlenmiş veri: {standardized_data.shape[0]} satır")
        return standardized_data

    def advanced_feature_selection(self, X, y, method='combined', n_features=None):
        """Gelişmiş feature selection"""
        print(f"Feature selection uygulanıyor: {method}")
        
        if n_features is None:
            n_features = min(self.max_features, X.shape[1] // 10)  # Feature sayısının %10'u
        
        n_features = max(self.min_features, min(n_features, X.shape[1]))
        print(f"Seçilecek feature sayısı: {n_features}")
        
        if method == 'f_test':
            # F-test ile feature selection
            selector = SelectKBest(score_func=f_classif, k=n_features)
            X_selected = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
            selected_scores = selector.scores_[selected_indices]
            
        elif method == 'mutual_info':
            # Mutual information ile feature selection
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
            X_selected = selector.fit_transform(X, y)
            selected_indices = selector.get_support(indices=True)
            selected_scores = selector.scores_[selected_indices]
            
        elif method == 'rf_importance':
            # Random Forest importance ile feature selection
            rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
            rf.fit(X, y)
            importance_scores = rf.feature_importances_
            selected_indices = np.argsort(importance_scores)[-n_features:]
            X_selected = X[:, selected_indices]
            selected_scores = importance_scores[selected_indices]
            
        elif method == 'lasso':
            # LASSO ile feature selection
            lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
            lasso.fit(X, y)
            selected_indices = np.where(lasso.coef_[0] != 0)[0]
            if len(selected_indices) > n_features:
                coef_abs = np.abs(lasso.coef_[0])
                selected_indices = np.argsort(coef_abs)[-n_features:]
            X_selected = X[:, selected_indices]
            selected_scores = np.abs(lasso.coef_[0][selected_indices])
            
        elif method == 'combined':
            f_selector = SelectKBest(score_func=f_classif, k=min(n_features*2, X.shape[1]))
            X_f_selected = f_selector.fit_transform(X, y)
            f_indices = f_selector.get_support(indices=True)
            
            # RF ile final seçim
            rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
            rf.fit(X_f_selected, y)
            rf_importance = rf.feature_importances_
            final_indices = np.argsort(rf_importance)[-n_features:]
            
            # Orijinal indekslere çevirme
            selected_indices = f_indices[final_indices]
            X_selected = X[:, selected_indices]
            selected_scores = rf_importance[final_indices]
            
        else:
            raise ValueError(f"Bilinmeyen feature selection metodu: {method}")
        
        print(f"Seçilen feature sayısı: {X_selected.shape[1]}")
        return X_selected, selected_indices, selected_scores

    def create_balanced_dataset(self, X, y, method='simple'):
        """Dengeli veri seti oluşturma"""
        print(f"Veri dengeleme uygulanıyor: {method}")
        
        if method == 'smote' and IMBLEARN_AVAILABLE:
            # SMOTE ile dengeleme
            n_minority = sum(y == 1)
            if n_minority >= 5:
                k_neighbors = min(5, n_minority - 1)
                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                try:
                    X_balanced, y_balanced = smote.fit_resample(X, y)
                except Exception as e:
                    print(f"SMOTE hatası: {e}. Orijinal veri kullanılıyor.")
                    X_balanced, y_balanced = X, y
            else:
                print("Uyarı: SMOTE için yeterli azınlık sınıf örneği yok. Orijinal veri kullanılıyor.")
                X_balanced, y_balanced = X, y
                
        elif method == 'undersample' and IMBLEARN_AVAILABLE:
            # Undersampling ile dengeleme
            rus = RandomUnderSampler(random_state=42)
            try:
                X_balanced, y_balanced = rus.fit_resample(X, y)
            except Exception as e:
                print(f"Undersampling hatası: {e}. Orijinal veri kullanılıyor.")
                X_balanced, y_balanced = X, y
            
        elif method == 'simple':
            # Basit dengeleme: Class weights kullanma
            print("Basit dengeleme uygulanıyor (class weights)")
            X_balanced, y_balanced = X, y
            
        else:
            print("Uyarı: İstenen dengeleme metodu mevcut değil. Orijinal veri kullanılıyor.")
            X_balanced, y_balanced = X, y
            
        print(f"Orijinal dağılım: {np.bincount(y)}")
        print(f"Dengeli dağılım: {np.bincount(y_balanced)}")
        
        return X_balanced, y_balanced

    def robust_cross_validation(self, X, y, model, cv_folds=5, cv_repeats=3):
        """Güçlü cross-validation"""
        print("Cross-validation uygulanıyor...")
        
        # Repeated stratified k-fold CV
        cv = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats, random_state=42)
        
        # Multiple metrics
        cv_scores = {
            'accuracy': cross_val_score(model, X, y, cv=cv, scoring='accuracy'),
            'roc_auc': cross_val_score(model, X, y, cv=cv, scoring='roc_auc'),
            'precision': cross_val_score(model, X, y, cv=cv, scoring='precision'),
            'recall': cross_val_score(model, X, y, cv=cv, scoring='recall'),
            'f1': cross_val_score(model, X, y, cv=cv, scoring='f1')
        }
        
        # Baseline model ile karşılaştırma
        baseline = DummyClassifier(strategy='stratified', random_state=42)
        baseline_scores = cross_val_score(baseline, X, y, cv=cv, scoring='roc_auc')
        
        results = {
            'cv_scores': cv_scores,
            'mean_scores': {metric: scores.mean() for metric, scores in cv_scores.items()},
            'std_scores': {metric: scores.std() for metric, scores in cv_scores.items()},
            'baseline_auc': baseline_scores.mean(),
            'improvement': cv_scores['roc_auc'].mean() - baseline_scores.mean()
        }
        
        print(f"CV Sonuçları:")
        for metric, scores in cv_scores.items():
            print(f"  {metric}: {scores.mean():.3f} ± {scores.std():.3f}")
        print(f"Baseline AUC: {baseline_scores.mean():.3f}")
        print(f"İyileştirme: {results['improvement']:.3f}")
        
        return results

    def train_regularized_models(self, X, y, feature_names):
        """Regularization ile model eğitimi"""
        print("Regularized modeller eğitiliyor...")
        
        # Veriyi standardize etme
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        models = {}
        
        # 1. Regularized Random Forest
        print("Random Forest (Regularized)...")
        rf_params = {
            'n_estimators': 50,
            'max_depth': 5,  # Overfitting önleme
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',  # Feature sayısını sınırlama
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'  # Class balancing
        }
        rf_model = RandomForestClassifier(**rf_params)
        rf_cv = self.robust_cross_validation(X_scaled, y, rf_model)
        
        # Final model
        rf_model.fit(X_scaled, y)
        rf_importance = rf_model.feature_importances_
        rf_genes = [feature_names[i] for i in np.argsort(rf_importance)[::-1]]
        
        models['random_forest'] = {
            'model': rf_model,
            'cv_results': rf_cv,
            'important_genes': rf_genes,
            'importance_scores': rf_importance
        }
        
        # 2. LASSO Logistic Regression
        print("LASSO Logistic Regression...")
        lasso_params = {
            'penalty': 'l1',
            'solver': 'liblinear',
            'C': 0.1,  # Güçlü regularization
            'random_state': 42,
            'max_iter': 50,
            'class_weight': 'balanced'  # Class balancing
        }
        lasso_model = LogisticRegression(**lasso_params)
        lasso_cv = self.robust_cross_validation(X_scaled, y, lasso_model)
        
        # Final model
        lasso_model.fit(X_scaled, y)
        lasso_coef = np.abs(lasso_model.coef_[0])
        lasso_genes = [feature_names[i] for i in np.where(lasso_coef > 0)[0]]
        
        models['lasso'] = {
            'model': lasso_model,
            'cv_results': lasso_cv,
            'selected_genes': lasso_genes,
            'coefficients': lasso_coef
        }
        
        # 3. Ridge Logistic Regression
        print("Ridge Logistic Regression...")
        ridge_params = {
            'penalty': 'l2',
            'solver': 'liblinear',
            'C': 1.0,
            'random_state': 42,
            'max_iter': 50,
            'class_weight': 'balanced'  # Class balancing
        }
        ridge_model = LogisticRegression(**ridge_params)
        ridge_cv = self.robust_cross_validation(X_scaled, y, ridge_model)
        
        # Final model
        ridge_model.fit(X_scaled, y)
        ridge_coef = np.abs(ridge_model.coef_[0])
        ridge_genes = [feature_names[i] for i in np.argsort(ridge_coef)[::-1]]
        
        models['ridge'] = {
            'model': ridge_model,
            'cv_results': ridge_cv,
            'important_genes': ridge_genes,
            'coefficients': ridge_coef
        }
        
        # 4. Regularized SVM
        print("Regularized SVM...")
        svm_params = {
            'kernel': 'linear',
            'C': 0.1,  # Güçlü regularization
            'random_state': 42,
            'probability': True,
            'class_weight': 'balanced'  # Class balancing
        }
        svm_model = SVC(**svm_params)
        svm_cv = self.robust_cross_validation(X_scaled, y, svm_model)
        
        # Final model
        svm_model.fit(X_scaled, y)
        svm_coef = np.abs(svm_model.coef_[0])
        svm_genes = [feature_names[i] for i in np.argsort(svm_coef)[::-1]]
        
        models['svm'] = {
            'model': svm_model,
            'cv_results': svm_cv,
            'important_genes': svm_genes,
            'coefficients': svm_coef
        }
        
        # 5. Regularized XGBoost
        print("Regularized XGBoost...")
        xgb_params = {
            'n_estimators': 50,
            'max_depth': 3,  # Overfitting önleme
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'scale_pos_weight': sum(y == 0) / sum(y == 1) if sum(y == 1) > 0 else 1  # Class balancing
        }
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_cv = self.robust_cross_validation(X_scaled, y, xgb_model)
        
        # Final model
        xgb_model.fit(X_scaled, y)
        xgb_importance = xgb_model.feature_importances_
        xgb_genes = [feature_names[i] for i in np.argsort(xgb_importance)[::-1]]
        
        models['xgboost'] = {
            'model': xgb_model,
            'cv_results': xgb_cv,
            'important_genes': xgb_genes,
            'importance_scores': xgb_importance
        }
        
        return models

    def load_data(self, expression_df, group_labels):
        """Expression data ve group labels yükleme"""
        self.expression_df = expression_df
        self.group_labels = group_labels
        print(f"Veri yüklendi: {expression_df.shape[0]} örnek, {expression_df.shape[1]} gen")
        print(f"Grup dağılımı: {dict(pd.Series(group_labels).value_counts())}")

    def select_important_genes(self, top_n=50, method='rf_importance'):
        """En önemli genleri seçme"""
        print(f"En önemli {top_n} gen seçiliyor...")
        
        # Binary labels oluşturma
        y = (self.group_labels == 'Endometriosis').astype(int)
        X = self.expression_df.values
        
        # Feature selection uygulama
        X_selected, selected_indices, selection_scores = self.advanced_feature_selection(
            X, y, method=method, n_features=top_n
        )
        
        if selected_indices is not None:
            selected_genes = [self.expression_df.columns[i] for i in selected_indices]
            print(f"Seçilen gen sayısı: {len(selected_genes)}")
            return selected_genes
        else:
            print("Hiç gen seçilemedi.")
            return []

    def cross_validate_model(self, selected_genes, model_type='rf', cv_folds=5, cv_repeats=3):
        """Model cross-validation"""
        print(f"{model_type.upper()} cross-validation başlatılıyor...")
        
        # Seçili genlerle veri hazırlama
        X = self.expression_df[selected_genes].values
        y = (self.group_labels == 'Endometriosis').astype(int)
        
        # Model seçimi
        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=50, max_depth=5, random_state=42, 
                class_weight='balanced', n_jobs=-1
            )
        elif model_type == 'svm':
            model = SVC(
                kernel='linear', C=0.1, random_state=42, 
                probability=True, class_weight='balanced'
            )
        elif model_type == 'logreg':
            model = LogisticRegression(
                penalty='l2', C=1.0, random_state=42, 
                max_iter=50, class_weight='balanced'
            )
        elif model_type == 'xgb':
            model = xgb.XGBClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                random_state=42, eval_metric='logloss', use_label_encoder=False,
                scale_pos_weight=sum(y == 0) / sum(y == 1) if sum(y == 1) > 0 else 1
            )
        else:
            raise ValueError(f"Bilinmeyen model tipi: {model_type}")
        
        # Cross-validation
        cv_results = self.robust_cross_validation(X, y, model, cv_folds, cv_repeats)
        
        print(f"{model_type.upper()} CV Sonuçları:")
        print(f"  AUC: {cv_results['mean_scores']['roc_auc']:.3f} ± {cv_results['std_scores']['roc_auc']:.3f}")
        print(f"  Accuracy: {cv_results['mean_scores']['accuracy']:.3f} ± {cv_results['std_scores']['accuracy']:.3f}")
        print(f"  F1: {cv_results['mean_scores']['f1']:.3f} ± {cv_results['std_scores']['f1']:.3f}")
        
        return cv_results

    def xgboost_training_curves(self, selected_genes):
        """XGBoost eğitim eğrileri"""
        print("XGBoost eğitim eğrileri çiziliyor...")
        
        # Veri hazırlama
        X = self.expression_df[selected_genes].values
        y = (self.group_labels == 'Endometriosis').astype(int)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # XGBoost model
        xgb_model = xgb.XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            random_state=42, eval_metric='logloss', use_label_encoder=False,
            scale_pos_weight=sum(y == 0) / sum(y == 1) if sum(y == 1) > 0 else 1
        )
        
        # Eğitim
        eval_set = [(X_train, y_train), (X_test, y_test)]
        xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        
        # Eğitim eğrileri
        results = xgb_model.evals_result()
        
        # Plot çizme
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(results['validation_0']['logloss'], label='Train')
        plt.plot(results['validation_1']['logloss'], label='Test')
        plt.xlabel('Iteration')
        plt.ylabel('Log Loss')
        plt.title('XGBoost Training Curves - Log Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        train_auc = [1 - x for x in results['validation_0']['logloss']]
        test_auc = [1 - x for x in results['validation_1']['logloss']]
        plt.plot(train_auc, label='Train')
        plt.plot(test_auc, label='Test')
        plt.xlabel('Iteration')
        plt.ylabel('AUC (1 - Log Loss)')
        plt.title('XGBoost Training Curves - AUC')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "xgboost_training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("XGBoost eğitim eğrileri kaydedildi.")

    def compare_all_models(self, selected_genes):
        """Tüm modellerin karşılaştırmalı değerlendirmesi"""
        print("Tüm modellerin karşılaştırmalı değerlendirmesi...")
        
        # Veri hazırlama
        X = self.expression_df[selected_genes].values
        y = (self.group_labels == 'Endometriosis').astype(int)
        
        # Modeller
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42, 
                class_weight='balanced', n_jobs=-1
            ),
            'SVM': SVC(
                kernel='linear', C=0.1, random_state=42, 
                probability=True, class_weight='balanced'
            ),
            'Logistic Regression': LogisticRegression(
                penalty='l2', C=1.0, random_state=42, 
                max_iter=1000, class_weight='balanced'
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                random_state=42, eval_metric='logloss', use_label_encoder=False,
                scale_pos_weight=sum(y == 0) / sum(y == 1) if sum(y == 1) > 0 else 1
            )
        }
        
        # Cross-validation sonuçları
        results = {}
        for name, model in models.items():
            print(f"\n{name} değerlendiriliyor...")
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
            
            auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
            f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
            
            results[name] = {
                'AUC': (auc_scores.mean(), auc_scores.std()),
                'Accuracy': (acc_scores.mean(), acc_scores.std()),
                'F1': (f1_scores.mean(), f1_scores.std())
            }
            
            print(f"  AUC: {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
            print(f"  Accuracy: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")
            print(f"  F1: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
        
        # Görselleştirme
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics = ['AUC', 'Accuracy', 'F1']
        
        for i, metric in enumerate(metrics):
            model_names = list(results.keys())
            means = [results[name][metric][0] for name in model_names]
            stds = [results[name][metric][1] for name in model_names]
            
            bars = axes[i].bar(model_names, means, yerr=stds, capsize=5, alpha=0.8)
            axes[i].set_title(f'{metric} Scores')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
            
            # Bar renkleri
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nModel karşılaştırma grafiği kaydedildi: model_comparison.png")
        
        return results

    def plot_training_curves(self, X, y, selected_genes, models):
        """Tüm modeller için train/validation eğrileri"""
        print("Train/Validation eğrileri çiziliyor...")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Her model için eğitim eğrileri
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        model_names = ['random_forest', 'lasso', 'ridge', 'svm', 'xgboost']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, (model_name, color) in enumerate(zip(model_names, colors)):
            if i >= len(axes):
                break
                
            model = models[model_name]['model']
            
            # Model eğitimi ve tahmin
            model.fit(X_train, y_train)
            train_pred = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_train)
            test_pred = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
            
            # Train/Test accuracy
            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))
            
            # Train/Test loss (log loss for probabilistic models)
            if hasattr(model, 'predict_proba'):
                from sklearn.metrics import log_loss
                train_loss = log_loss(y_train, train_pred)
                test_loss = log_loss(y_test, test_pred)
            else:
                train_loss = 1 - train_acc
                test_loss = 1 - test_acc
            
            # Plot
            axes[i].plot([0, 1], [train_acc, test_acc], 'o-', color=color, label=f'{model_name.upper()}')
            axes[i].set_xlabel('Train/Test')
            axes[i].set_ylabel('Accuracy')
            axes[i].set_title(f'{model_name.upper()} - Accuracy')
            axes[i].set_ylim(0, 1)
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
            
            # Loss plot
            if i < len(axes) - 1:
                axes[i+1].plot([0, 1], [train_loss, test_loss], 's-', color=color, label=f'{model_name.upper()}')
                axes[i+1].set_xlabel('Train/Test')
                axes[i+1].set_ylabel('Loss')
                axes[i+1].set_title(f'{model_name.upper()} - Loss')
                axes[i+1].grid(True, alpha=0.3)
                axes[i+1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves_all_models.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Train/Validation eğrileri kaydedildi: training_curves_all_models.png")

    def plot_roc_curves(self, X, y, models):
        """Tüm modeller için ROC eğrileri"""
        print("ROC eğrileri çiziliyor...")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        plt.figure(figsize=(10, 8))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, (model_name, color) in enumerate(zip(models.keys(), colors)):
            model = models[model_name]['model']
            
            # Model eğitimi
            model.fit(X_train, y_train)
            
            # ROC curve
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = model.decision_function(X_test)
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, color=color, lw=2, 
                    label=f'{model_name.upper()} (AUC = {auc_score:.3f})')
        
        # Random classifier
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - All Models')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "roc_curves_all_models.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("ROC eğrileri kaydedildi: roc_curves_all_models.png")

    def plot_feature_importance_comparison(self, models, selected_genes):
        """Feature importance karşılaştırması"""
        print("Feature importance karşılaştırması çiziliyor...")
        
        # Her modelden top 10 geni alma
        top_genes_per_model = {}
        
        for model_name, model_results in models.items():
            if 'important_genes' in model_results:
                top_genes_per_model[model_name] = model_results['important_genes'][:10]
            elif 'selected_genes' in model_results:
                top_genes_per_model[model_name] = model_results['selected_genes'][:10]
        
        # Tüm unique genleri toplama
        all_top_genes = set()
        for genes in top_genes_per_model.values():
            all_top_genes.update(genes)
        
        # Heatmap için matrix oluşturma
        gene_list = list(all_top_genes)
        model_names = list(top_genes_per_model.keys())
        
        importance_matrix = np.zeros((len(gene_list), len(model_names)))
        
        for i, gene in enumerate(gene_list):
            for j, model_name in enumerate(model_names):
                if gene in top_genes_per_model[model_name]:
                    rank = top_genes_per_model[model_name].index(gene)
                    importance_matrix[i, j] = 10 - rank
                else:
                    importance_matrix[i, j] = 0
        
        # Heatmap
        plt.figure(figsize=(12, max(8, len(gene_list) * 0.3)))
        
        # Model isimlerini düzenleme
        model_labels = [name.replace('_', ' ').title() for name in model_names]
        
        sns.heatmap(importance_matrix, 
                   xticklabels=model_labels,
                   yticklabels=gene_list,
                   cmap='YlOrRd',
                   annot=True,
                   fmt='.0f',
                   cbar_kws={'label': 'Importance Rank (10=highest)'})
        
        plt.title('Feature Importance Comparison Across Models')
        plt.xlabel('Models')
        plt.ylabel('Genes')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_importance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Feature importance karşılaştırması kaydedildi: feature_importance_comparison.png")

    def advanced_model_analysis_with_gridsearch(self, X, y, feature_names):
        """GridSearchCV ile en iyi modeli bulma ve kapsamlı analiz"""
        from sklearn.model_selection import GridSearchCV, train_test_split
        from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, classification_report
        from sklearn.ensemble import RandomForestClassifier
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        
        print("=== GRIDSEARCHCV İLE GELİŞMİŞ MODEL ANALİZİ ===")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Random Forest için GridSearchCV parametreleri
        rf_param_grid = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 5, 10],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Base Random Forest model
        base_rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        
        # GridSearchCV
        print("GridSearchCV başlatılıyor...")
        grid_search = GridSearchCV(
            estimator=base_rf,
            param_grid=rf_param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # GridSearchCV eğitimi
        grid_search.fit(X_train, y_train)
        
        # En iyi model
        best_rf = grid_search.best_estimator_
        print(f"En iyi parametreler: {grid_search.best_params_}")
        print(f"En iyi CV skoru: {grid_search.best_score_:.3f}")
        
        # Test seti üzerinde değerlendirme
        test_score = best_rf.score(X_test, y_test)
        print(f"Test seti accuracy: {test_score:.3f}")
        
        # 1. ROC Eğrisi ve AUC Skoru
        print("ROC eğrisi çiziliyor...")
        rf_probs = best_rf.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, rf_probs, pos_label=best_rf.classes_[1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Best Random Forest (GridSearchCV)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(self.output_dir / 'rf_roc_curve_gridsearch.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion Matrix
        print("Confusion matrix çiziliyor...")
        y_pred = best_rf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=best_rf.classes_)
        
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_rf.classes_)
        disp.plot(cmap='Blues', values_format='d')
        plt.title('Confusion Matrix - Best Random Forest (GridSearchCV)')
        plt.savefig(self.output_dir / 'rf_confusion_matrix_gridsearch.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        # 3. Classification Report
        print("Classification report oluşturuluyor...")
        report = classification_report(y_test, y_pred, target_names=best_rf.classes_.astype(str))
        print("Classification Report:\n", report)
        
        # Report'u dosyaya kaydet
        with open(self.output_dir / 'rf_classification_report_gridsearch.txt', 'w') as f:
            f.write("GridSearchCV Random Forest Classification Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Best Parameters: {grid_search.best_params_}\n")
            f.write(f"Best CV Score: {grid_search.best_score_:.3f}\n")
            f.write(f"Test Accuracy: {test_score:.3f}\n")
            f.write(f"ROC AUC: {roc_auc:.3f}\n\n")
            f.write(report)
        
        # 4. GridSearchCV Sonuçlarını CSV'ye Kaydetme
        print("GridSearchCV sonuçları kaydediliyor...")
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv(self.output_dir / 'rf_gridsearch_results.csv', index=False)
        
        # En iyi 10 parametre kombinasyonunu gösterme
        top_results = results_df.nlargest(10, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
        print("\nEn iyi 10 parametre kombinasyonu:")
        for i, row in top_results.iterrows():
            print(f"{i+1}. Score: {row['mean_test_score']:.3f} ± {row['std_test_score']:.3f}")
            print(f"   Params: {row['params']}")
        
        # 5. Özellik Önem Skoru (Feature Importance)
        print("Feature importance çiziliyor...")
        importances = best_rf.feature_importances_
        
        # En önemli 20 özelliği görselleştirme
        top_indices = importances.argsort()[::-1][:20]
        plt.figure(figsize=(12, 8))
        plt.barh(range(20), importances[top_indices][::-1], align='center')
        plt.yticks(range(20), [feature_names[i] for i in top_indices][::-1])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importances - Best Random Forest (GridSearchCV)')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'rf_feature_importance_gridsearch.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        # Feature importance'ları CSV'ye kaydetme
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        importance_df.to_csv(self.output_dir / 'rf_feature_importance_gridsearch.csv', index=False)
        
        # 6. Model Performans Özeti
        performance_summary = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'test_accuracy': test_score,
            'roc_auc': roc_auc,
            'top_features': [feature_names[i] for i in top_indices[:10]]
        }
        
        # Özeti JSON olarak kaydetme
        import json
        with open(self.output_dir / 'rf_performance_summary_gridsearch.json', 'w') as f:
            json.dump(performance_summary, f, indent=2)
        
        print(f"\n=== GRIDSEARCHCV ANALİZİ TAMAMLANDI ===")
        print(f"Oluşturulan dosyalar:")
        print(f"✓ rf_roc_curve_gridsearch.png")
        print(f"✓ rf_confusion_matrix_gridsearch.png")
        print(f"✓ rf_classification_report_gridsearch.txt")
        print(f"✓ rf_gridsearch_results.csv")
        print(f"✓ rf_feature_importance_gridsearch.png")
        print(f"✓ rf_feature_importance_gridsearch.csv")
        print(f"✓ rf_performance_summary_gridsearch.json")
        
        return {
            'best_model': best_rf,
            'grid_search': grid_search,
            'performance_summary': performance_summary,
            'feature_importance': importance_df
        }

    def comprehensive_biomarker_analysis(self, endo_file, auto_file):
        """Kapsamlı biyomarker analizi"""
        print("=== KAPSAMLI ENDOMETRİOZİS BİYOMARKER ANALİZİ BAŞLATILIYOR ===")
        
        # ===========================================================================
        # VERİ YÜKLEME VE ÖN İŞLEME
        # ===========================================================================
        print("\n" + "="*50)
        print("VERİ YÜKLEME VE ÖN İŞLEME")
        print("="*50)
        
        # Excel dosyalarını yükleme
        endo_sheets = self.load_excel_sheets(endo_file)
        auto_sheets = self.load_excel_sheets(auto_file)
        
        # Verileri standartlaştırma
        endo_data_list = []
        for sheet_name, data in endo_sheets.items():
            endo_data_list.append(self.standardize_dgea_data(data, sheet_name))
        
        auto_data_list = []
        for sheet_name, data in auto_sheets.items():
            auto_data_list.append(self.standardize_dgea_data(data, sheet_name))
        
        # Verileri birleştirme
        all_endo_data = pd.concat(endo_data_list, ignore_index=True)
        all_auto_data = pd.concat(auto_data_list, ignore_index=True)
        
        # Grup bilgisi ekleme
        all_endo_data['Group'] = 'Endometriosis'
        all_auto_data['Group'] = 'Autoimmune'
        
        # ===========================================================================
        # FEATURE SELECTION VE EXPRESSION MATRIX OLUŞTURMA
        # ===========================================================================
        print("\n" + "="*50)
        print("FEATURE SELECTION VE EXPRESSION MATRIX")
        print("="*50)
        
        # Tüm genleri alma
        all_genes = set(all_endo_data['Gene'].unique()) | set(all_auto_data['Gene'].unique())
        print(f"Toplam gen sayısı: {len(all_genes)}")
        
        # Expression matrix oluşturma
        all_data = pd.concat([all_endo_data, all_auto_data], ignore_index=True)
        expression_data = all_data.groupby(['Gene', 'Dataset', 'Group'])['logFC'].mean().reset_index()
        expression_matrix = expression_data.pivot_table(
            index=['Dataset', 'Group'], 
            columns='Gene', 
            values='logFC', 
            fill_value=0
        )
        
        # Expression matrix hazırlama
        expr_matrix = expression_matrix.values
        groups = expression_matrix.index.get_level_values('Group').values
        gene_names = expression_matrix.columns.tolist()
        
        print(f"Expression matrix boyutu: {expr_matrix.shape}")
        print(f"Grup dağılımı: {dict(pd.Series(groups).value_counts())}")
        
        # ===========================================================================
        # FEATURE SELECTION
        # ===========================================================================
        print("\n" + "="*50)
        print("FEATURE SELECTION")
        print("="*50)
        
        # Binary labels
        y = (groups == 'Endometriosis').astype(int)
        
        # Feature selection uygulama
        X_selected, selected_indices, selection_scores = self.advanced_feature_selection(
            expr_matrix, y, method='combined', n_features=self.max_features
        )
        
        if selected_indices is not None:
            selected_genes = [gene_names[i] for i in selected_indices]
            print(f"Seçilen gen sayısı: {len(selected_genes)}")
        else:
            selected_genes = []
            print("Hiç gen seçilemedi.")
        
        # ===========================================================================
        # VERİ DENGELEME
        # ===========================================================================
        print("\n" + "="*50)
        print("VERİ DENGELEME")
        print("="*50)
        
        X_balanced, y_balanced = self.create_balanced_dataset(X_selected, y, method='simple')
        
        # ===========================================================================
        # MODEL EĞİTİMİ
        # ===========================================================================
        print("\n" + "="*50)
        print("MODEL EĞİTİMİ")
        print("="*50)
        
        models = self.train_regularized_models(X_balanced, y_balanced, selected_genes)
        
        # ===========================================================================
        # PERFORMANS KARŞILAŞTIRMASI
        # ===========================================================================
        print("\n" + "="*50)
        print("PERFORMANS KARŞILAŞTIRMASI")
        print("="*50)
        
        # Model performanslarını karşılaştırma
        performance_summary = []
        for model_name, model_results in models.items():
            cv_results = model_results['cv_results']
            performance_summary.append({
                'Model': model_name,
                'CV_AUC_Mean': cv_results['mean_scores']['roc_auc'],
                'CV_AUC_Std': cv_results['std_scores']['roc_auc'],
                'CV_Accuracy_Mean': cv_results['mean_scores']['accuracy'],
                'CV_F1_Mean': cv_results['mean_scores']['f1'],
                'Improvement_over_Baseline': cv_results['improvement']
            })
        
        performance_df = pd.DataFrame(performance_summary)
        print("\nModel Performans Karşılaştırması:")
        print(performance_df.round(3))
        
        # En iyi modeli seçme
        best_model_name = performance_df.loc[performance_df['CV_AUC_Mean'].idxmax(), 'Model']
        best_model = models[best_model_name]
        print(f"\nEn iyi model: {best_model_name}")
        
        # ===========================================================================
        # BİYOMARKER SEÇİMİ
        # ===========================================================================
        print("\n" + "="*50)
        print("BİYOMARKER SEÇİMİ")
        print("="*50)
        
        # Her modelden önemli genleri alma
        all_important_genes = set()
        gene_importance = {}
        
        for model_name, model_results in models.items():
            if 'important_genes' in model_results:
                genes = model_results['important_genes'][:20]  # Top 20
                all_important_genes.update(genes)
                for gene in genes:
                    if gene not in gene_importance:
                        gene_importance[gene] = {}
                    gene_importance[gene][f'{model_name}_rank'] = genes.index(gene) + 1
            elif 'selected_genes' in model_results:
                genes = model_results['selected_genes']
                all_important_genes.update(genes)
                for gene in genes:
                    if gene not in gene_importance:
                        gene_importance[gene] = {}
                    gene_importance[gene][f'{model_name}_selected'] = True
        
        # Biyomarker özeti oluşturma
        biomarker_summary = []
        for gene in all_important_genes:
            gene_info = {'Gene': gene}
            if gene in gene_importance:
                gene_info.update(gene_importance[gene])
            biomarker_summary.append(gene_info)
        
        biomarker_df = pd.DataFrame(biomarker_summary)
        
        # ===========================================================================
        # SONUÇLARI KAYDETME
        # ===========================================================================
        print("\n" + "="*50)
        print("SONUÇLARI KAYDETME")
        print("="*50)
        
        # Performans sonuçları
        performance_df.to_csv(self.output_dir / "model_performance.csv", index=False)
        
        # Biyomarker sonuçları
        biomarker_df.to_csv(self.output_dir / "biomarkers.csv", index=False)
        
        # Detaylı model sonuçları
        for model_name, model_results in models.items():
            if 'important_genes' in model_results and 'importance_scores' in model_results:
                # Random Forest, XGBoost gibi modeller için
                genes_df = pd.DataFrame({
                    'Gene': model_results['important_genes'],
                    'Importance': model_results['importance_scores'][np.argsort(model_results['importance_scores'])[::-1]]
                })
                genes_df.to_csv(self.output_dir / f"{model_name}_genes.csv", index=False)
            elif 'selected_genes' in model_results and 'coefficients' in model_results:
                # LASSO gibi modeller için
                genes_df = pd.DataFrame({
                    'Gene': model_results['selected_genes'],
                    'Coefficient': model_results['coefficients'][model_results['coefficients'] > 0]
                })
                genes_df.to_csv(self.output_dir / f"{model_name}_genes.csv", index=False)
            elif 'important_genes' in model_results and 'coefficients' in model_results:
                # Ridge, SVM gibi modeller için
                genes_df = pd.DataFrame({
                    'Gene': model_results['important_genes'],
                    'Coefficient': model_results['coefficients'][np.argsort(model_results['coefficients'])[::-1]]
                })
                genes_df.to_csv(self.output_dir / f"{model_name}_genes.csv", index=False)
        
        # ===========================================================================
        # GÖRSELLEŞTİRME
        # ===========================================================================
        print("\n" + "="*50)
        print("GÖRSELLEŞTİRME")
        print("="*50)
        
        # Model performans karşılaştırması
        plt.figure(figsize=(12, 8))
        x = np.arange(len(performance_df))
        width = 0.35
        
        plt.bar(x - width/2, performance_df['CV_AUC_Mean'], width, label='CV AUC', alpha=0.8)
        plt.bar(x + width/2, performance_df['CV_Accuracy_Mean'], width, label='CV Accuracy', alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, performance_df['Model'].tolist(), rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Top biyomarkerlar
        if len(biomarker_df) > 0:
            plt.figure(figsize=(12, 8))
            top_genes = biomarker_df.head(20)
            plt.barh(range(len(top_genes)), [1]*len(top_genes))
            plt.yticks(range(len(top_genes)), top_genes['Gene'].tolist())
            plt.xlabel('Importance')
            plt.title('Top 20 Biomarkers')
            plt.tight_layout()
            plt.savefig(self.output_dir / "top_biomarkers.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Train/Validation curves for all models
        print("Train/Validation eğrileri çiziliyor...")
        self.plot_training_curves(X_balanced, y_balanced, selected_genes, models)
        
        # ROC curves for all models
        print("ROC eğrileri çiziliyor...")
        self.plot_roc_curves(X_balanced, y_balanced, models)
        
        # Feature importance comparison
        print("Feature importance karşılaştırması çiziliyor...")
        self.plot_feature_importance_comparison(models, selected_genes)
        
        # Her model için ayrı ayrı accuracy/loss grafiği çizme
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced)
        for model_name, model_results in models.items():
            model = model_results['model']
            # XGBoost için eval_set ile fit edilmişse tekrar fit etme
            if model_name == 'xgboost':
                model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
            else:
                model.fit(X_train, y_train)
            self.plot_model_accuracy_loss(model, model_name, X_train, X_test, y_train, y_test)

        self.plot_incremental_curves_for_all_models(X_balanced, y_balanced, self.output_dir)

        # ===========================================================================
        # GRIDSEARCHCV İLE GELİŞMİŞ RANDOM FOREST ANALİZİ
        # ===========================================================================
        print("\n" + "="*50)
        print("GRIDSEARCHCV İLE GELİŞMİŞ RANDOM FOREST ANALİZİ")
        print("="*50)
        
        # GridSearchCV ile gelişmiş analiz
        gridsearch_results = self.advanced_model_analysis_with_gridsearch(X_balanced, y_balanced, selected_genes)

        # ===========================================================================
        # ÖZET RAPOR
        # ===========================================================================
        print("\n=== ANALİZ TAMAMLANDI ===")
        print(f"Sonuçlar şu dizinde kaydedildi: {self.output_dir}")
        
        print("\n=== ÖZET RAPOR ===")
        print(f"Toplam gen sayısı: {len(all_genes)}")
        print(f"Seçilen gen sayısı: {len(selected_genes)}")
        print(f"Biyomarker sayısı: {len(all_important_genes)}")
        print(f"En iyi model: {best_model_name}")
        print(f"En iyi CV AUC: {performance_df['CV_AUC_Mean'].max():.3f}")
        print(f"Baseline üzerindeki iyileştirme: {performance_df['Improvement_over_Baseline'].max():.3f}")
        
        return {
            'models': models,
            'performance_df': performance_df,
            'biomarker_df': biomarker_df,
            'selected_genes': selected_genes,
            'best_model': best_model,
            'best_model_name': best_model_name,
            'gridsearch_results': gridsearch_results,
            'expression_matrix': expr_matrix,
            'group_labels': groups
        }

    def plot_incremental_curves_for_all_models(self, X, y, output_dir):
        """Her model için adım adım (incremental) train/val accuracy ve loss eğrisi çizer (K-Fold ortalaması ile)."""
        from sklearn.model_selection import StratifiedKFold
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        import xgboost as xgb
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import accuracy_score, log_loss
        import warnings
        warnings.filterwarnings('ignore')

        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Random Forest - Overfitting önleme parametreleri ile
        n_trees = 200       # Daha kısa sürede test edebilmek için azaltıldı
        step = 20           # Daha büyük adım aralıklarıyla test
        n_trees_list = range(step, n_trees + 1, step)
        train_accs, val_accs, train_losses, val_losses = [], [], [], []
        
        for n_trees_current in n_trees_list:
            fold_train_acc, fold_val_acc, fold_train_loss, fold_val_loss = [], [], [], []
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Overfitting önleme parametreleri ile Random Forest
                model = RandomForestClassifier(
                    n_estimators=n_trees_current,
                    max_depth=10,              # Ağaç derinliği sınırlı
                    min_samples_split=10,      # Dal bölme sınırı
                    min_samples_leaf=5,        # Her yaprakta min örnek
                    max_features='sqrt',       # Özellik seçimini sınırlama
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                fold_train_acc.append(accuracy_score(y_train, y_train_pred))
                fold_val_acc.append(accuracy_score(y_val, y_val_pred))
                fold_train_loss.append(1 - fold_train_acc[-1])
                fold_val_loss.append(1 - fold_val_acc[-1])
            train_accs.append(np.mean(fold_train_acc))
            val_accs.append(np.mean(fold_val_acc))
            train_losses.append(np.mean(fold_train_loss))
            val_losses.append(np.mean(fold_val_loss))
        
        plt.figure(figsize=(16,5))
        plt.subplot(1,2,1)
        plt.plot(n_trees_list, train_accs, label='Train', color='blue')
        plt.plot(n_trees_list, val_accs, label='Validation', color='red')
        plt.xlabel('n_estimators')
        plt.ylabel('Accuracy')
        plt.title('Random Forest - Accuracy Curves (Regularized)')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(n_trees_list, train_losses, label='Train', color='blue')
        plt.plot(n_trees_list, val_losses, label='Validation', color='red')
        plt.xlabel('n_estimators')
        plt.ylabel('Loss')
        plt.title('Random Forest - Loss Curves (Regularized)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'random_forest_regularized_curve.png', dpi=200, bbox_inches='tight')
        plt.close()

        # XGBoost - Düşük iterasyon ile
        n_trees_xgb = 200
        step_xgb = 20
        n_trees_list_xgb = range(step_xgb, n_trees_xgb + 1, step_xgb)
        train_accs, val_accs, train_losses, val_losses = [], [], [], []
        for n_trees in n_trees_list_xgb:
            fold_train_acc, fold_val_acc, fold_train_loss, fold_val_loss = [], [], [], []
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                model = xgb.XGBClassifier(n_estimators=n_trees, max_depth=3, learning_rate=0.1, random_state=42, eval_metric='logloss', use_label_encoder=False)
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                fold_train_acc.append(accuracy_score(y_train, y_train_pred))
                fold_val_acc.append(accuracy_score(y_val, y_val_pred))
                y_train_proba = model.predict_proba(X_train)[:,1]
                y_val_proba = model.predict_proba(X_val)[:,1]
                fold_train_loss.append(log_loss(y_train, y_train_proba))
                fold_val_loss.append(log_loss(y_val, y_val_proba))
            train_accs.append(np.mean(fold_train_acc))
            val_accs.append(np.mean(fold_val_acc))
            train_losses.append(np.mean(fold_train_loss))
            val_losses.append(np.mean(fold_val_loss))
        plt.figure(figsize=(16,5))
        plt.subplot(1,2,1)
        plt.plot(n_trees_list, train_accs, label='Train', color='blue')
        plt.plot(n_trees_list, val_accs, label='Validation', color='red')
        plt.xlabel('n_estimators')
        plt.ylabel('Accuracy')
        plt.title('XGBoost - Accuracy Curves (K-Fold)')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(n_trees_list, train_losses, label='Train', color='blue')
        plt.plot(n_trees_list, val_losses, label='Validation', color='red')
        plt.xlabel('n_estimators')
        plt.ylabel('Loss')
        plt.title('XGBoost - Loss Curves (K-Fold)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'xgboost_curve.png', dpi=200, bbox_inches='tight')
        plt.close()

        # SVM (C parametresi log-scale)
        C_list = np.logspace(-3, 2, 20)
        train_accs, val_accs, train_losses, val_losses = [], [], [], []
        for c in C_list:
            fold_train_acc, fold_val_acc, fold_train_loss, fold_val_loss = [], [], [], []
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                model = SVC(kernel='linear', C=c, probability=True, random_state=42)
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                fold_train_acc.append(accuracy_score(y_train, y_train_pred))
                fold_val_acc.append(accuracy_score(y_val, y_val_pred))
                y_train_proba = model.predict_proba(X_train)[:,1]
                y_val_proba = model.predict_proba(X_val)[:,1]
                fold_train_loss.append(log_loss(y_train, y_train_proba))
                fold_val_loss.append(log_loss(y_val, y_val_proba))
            train_accs.append(np.mean(fold_train_acc))
            val_accs.append(np.mean(fold_val_acc))
            train_losses.append(np.mean(fold_train_loss))
            val_losses.append(np.mean(fold_val_loss))
        plt.figure(figsize=(16,5))
        plt.subplot(1,2,1)
        plt.semilogx(C_list, train_accs, label='Train', color='blue')
        plt.semilogx(C_list, val_accs, label='Validation', color='red')
        plt.xlabel('C')
        plt.ylabel('Accuracy')
        plt.title('SVM - Accuracy Curves (K-Fold)')
        plt.legend()
        plt.subplot(1,2,2)
        plt.semilogx(C_list, train_losses, label='Train', color='blue')
        plt.semilogx(C_list, val_losses, label='Validation', color='red')
        plt.xlabel('C')
        plt.ylabel('Loss')
        plt.title('SVM - Loss Curves (K-Fold)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'svm_curve.png', dpi=200, bbox_inches='tight')
        plt.close()

        # LASSO (Logistic Regression L1)
        C_list = np.logspace(-3, 2, 20)
        train_accs, val_accs, train_losses, val_losses = [], [], [], []
        for c in C_list:
            fold_train_acc, fold_val_acc, fold_train_loss, fold_val_loss = [], [], [], []
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                model = LogisticRegression(penalty='l1', solver='liblinear', C=c, random_state=42, max_iter=50)
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                fold_train_acc.append(accuracy_score(y_train, y_train_pred))
                fold_val_acc.append(accuracy_score(y_val, y_val_pred))
                y_train_proba = model.predict_proba(X_train)[:,1]
                y_val_proba = model.predict_proba(X_val)[:,1]
                fold_train_loss.append(log_loss(y_train, y_train_proba))
                fold_val_loss.append(log_loss(y_val, y_val_proba))
            train_accs.append(np.mean(fold_train_acc))
            val_accs.append(np.mean(fold_val_acc))
            train_losses.append(np.mean(fold_train_loss))
            val_losses.append(np.mean(fold_val_loss))
        plt.figure(figsize=(16,5))
        plt.subplot(1,2,1)
        plt.semilogx(C_list, train_accs, label='Train', color='blue')
        plt.semilogx(C_list, val_accs, label='Validation', color='red')
        plt.xlabel('C')
        plt.ylabel('Accuracy')
        plt.title('LASSO - Accuracy Curves (K-Fold)')
        plt.legend()
        plt.subplot(1,2,2)
        plt.semilogx(C_list, train_losses, label='Train', color='blue')
        plt.semilogx(C_list, val_losses, label='Validation', color='red')
        plt.xlabel('C')
        plt.ylabel('Loss')
        plt.title('LASSO - Loss Curves (K-Fold)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'lasso_curve.png', dpi=200, bbox_inches='tight')
        plt.close()

        # Ridge (Logistic Regression L2)
        C_list = np.logspace(-3, 2, 20)
        train_accs, val_accs, train_losses, val_losses = [], [], [], []
        for c in C_list:
            fold_train_acc, fold_val_acc, fold_train_loss, fold_val_loss = [], [], [], []
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                model = LogisticRegression(penalty='l2', solver='liblinear', C=c, random_state=42, max_iter=50)
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                fold_train_acc.append(accuracy_score(y_train, y_train_pred))
                fold_val_acc.append(accuracy_score(y_val, y_val_pred))
                y_train_proba = model.predict_proba(X_train)[:,1]
                y_val_proba = model.predict_proba(X_val)[:,1]
                fold_train_loss.append(log_loss(y_train, y_train_proba))
                fold_val_loss.append(log_loss(y_val, y_val_proba))
            train_accs.append(np.mean(fold_train_acc))
            val_accs.append(np.mean(fold_val_acc))
            train_losses.append(np.mean(fold_train_loss))
            val_losses.append(np.mean(fold_val_loss))
        plt.figure(figsize=(16,5))
        plt.subplot(1,2,1)
        plt.semilogx(C_list, train_accs, label='Train', color='blue')
        plt.semilogx(C_list, val_accs, label='Validation', color='red')
        plt.xlabel('C')
        plt.ylabel('Accuracy')
        plt.title('Ridge - Accuracy Curves (K-Fold)')
        plt.legend()
        plt.subplot(1,2,2)
        plt.semilogx(C_list, train_losses, label='Train', color='blue')
        plt.semilogx(C_list, val_losses, label='Validation', color='red')
        plt.xlabel('C')
        plt.ylabel('Loss')
        plt.title('Ridge - Loss Curves (K-Fold)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'ridge_curve.png', dpi=200, bbox_inches='tight')
        plt.close()
        print('Tüm incremental loss/accuracy grafikleri (K-Fold) kaydedildi.')

    def plot_model_accuracy_loss(self, model, model_name, X_train, X_test, y_train, y_test):
        """Her model için accuracy ve loss eğrilerini ayrı ayrı kaydet"""
        import matplotlib.pyplot as plt
        from sklearn.metrics import accuracy_score, log_loss
        # XGBoost için özel log
        if model_name == 'xgboost' and hasattr(model, 'evals_result_'):
            results = model.evals_result()
            epochs = len(results['validation_0']['logloss'])
            x_axis = range(epochs)
            # Accuracy hesaplama
            train_acc = 1 - np.array(results['validation_0']['logloss'])
            val_acc = 1 - np.array(results['validation_1']['logloss'])
            train_loss = np.array(results['validation_0']['logloss'])
            val_loss = np.array(results['validation_1']['logloss'])
        else:
            # Diğer modeller için tek değer
            train_pred = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_train)
            test_pred = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
            train_acc = [accuracy_score(y_train, model.predict(X_train))]
            val_acc = [accuracy_score(y_test, model.predict(X_test))]
            if hasattr(model, 'predict_proba'):
                train_loss = [log_loss(y_train, train_pred)]
                val_loss = [log_loss(y_test, test_pred)]
            else:
                train_loss = [1 - train_acc[0]]
                val_loss = [1 - val_acc[0]]
            x_axis = [0]
        # Plot
        plt.figure(figsize=(16, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x_axis, train_acc, label='Train', color='blue')
        plt.plot(x_axis, val_acc, label='Validation', color='red')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title(f'{model_name.replace("_", " ").title()} - Accuracy Curves')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x_axis, train_loss, label='Train', color='blue')
        plt.plot(x_axis, val_loss, label='Validation', color='red')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'{model_name.replace("_", " ").title()} - Loss Curves')
        plt.legend()
        plt.tight_layout()
        outpath = self.output_dir / f"{model_name}_accuracy_loss.png"
        plt.savefig(outpath, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"{model_name} için accuracy/loss grafiği kaydedildi: {outpath}")

    def run_comprehensive_rf_benchmark(self, X, y, feature_names, output_dir=None):
        """
        Farklı feature selection yöntemleri, gen sayıları ve SMOTE ile/SMOTE olmadan
        Random Forest GridSearchCV benchmark'ı yapar. Sonuçları organize kaydeder.
        """
        import os
        import pandas as pd
        import numpy as np
        from pathlib import Path
        from imblearn.over_sampling import SMOTE
        from sklearn.model_selection import train_test_split
        import shutil
        import time
        
        if output_dir is None:
            output_dir = self.output_dir / "rf_benchmark"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        feature_methods = ['f_test', 'mutual_info', 'rf_importance']
        gene_counts = [10, 20, 30, 50, 100]
        smote_options = [False, True]
        summary_rows = []
        total_runs = len(feature_methods) * len(gene_counts) * len(smote_options)
        run_idx = 1
        start_time = time.time()

        for method in feature_methods:
            for n_genes in gene_counts:
                for use_smote in smote_options:
                    combo_name = f"{method}_top{n_genes}_{'smote' if use_smote else 'nosmote'}"
                    combo_dir = output_dir / combo_name
                    combo_dir.mkdir(parents=True, exist_ok=True)
                    print(f"\n=== [{run_idx}/{total_runs}] {combo_name} başlatılıyor ===")
                    run_idx += 1

                    # Feature selection
                    X_selected, selected_indices, _ = self.advanced_feature_selection(X, y, method=method, n_features=n_genes)
                    selected_genes = [feature_names[i] for i in selected_indices]

                    # SMOTE ile dengeleme
                    X_bal, y_bal = X_selected, y
                    if use_smote:
                        try:
                            smote = SMOTE(random_state=42, k_neighbors=min(5, sum(y==1)-1))
                            X_bal, y_bal = smote.fit_resample(X_selected, y)
                            print(f"SMOTE uygulandı: {np.bincount(y_bal)}")
                        except Exception as e:
                            print(f"SMOTE hatası: {e}. Orijinal veri kullanılıyor.")
                            X_bal, y_bal = X_selected, y

                    # GridSearchCV ve analizi gerçekleştirme
                    analyzer = self
                    analyzer.output_dir = combo_dir
                    results = analyzer.advanced_model_analysis_with_gridsearch(X_bal, y_bal, selected_genes)

                    # Sonuçları özetleme
                    perf = results['performance_summary']
                    perf_row = {
                        'feature_method': method,
                        'n_genes': n_genes,
                        'smote': use_smote,
                        'best_params': perf['best_params'],
                        'best_cv_score': perf['best_cv_score'],
                        'test_accuracy': perf['test_accuracy'],
                        'roc_auc': perf['roc_auc'],
                        'top_features': perf['top_features']
                    }
                    summary_rows.append(perf_row)

        # Tüm sonuçları CSV'ye kaydetme
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / 'rf_benchmark_summary.csv', index=False)
        print(f"\nTüm kombinasyonlar tamamlandı. Sonuçlar: {output_dir / 'rf_benchmark_summary.csv'}")
        print(f"Toplam süre: {time.time() - start_time:.1f} sn")
        return summary_df

if __name__ == "__main__":
    # Excel dosya yolları
    endo_file = "C:/Users/bilge/Desktop/bilge/bilge/endo_dgea_all.xlsx"
    auto_file = "C:/Users/bilge/Desktop/bilge/bilge/autoimmune_dgea_all.xlsx"
    
    # Analizi başlatma
    analyzer = EndometriosisBiomarkerAnalysis()
    
    # Yeni workflow örneği
    print("=== YENİ WORKFLOW ÖRNEĞİ ===")
    print("Bu örnek, kullanıcının istediği workflow'u gösterir.")
    print("Gerçek veri ile kullanmak için aşağıdaki adımları takip edin:")
    print()
    print("# 1. Model nesnesini başlat")
    print("model = EndometriosisBiomarkerAnalysis()")
    print()
    print("# 2. Veriyi yükle")
    print("expression_df = df.drop(columns=['class'])")
    print("group_labels = df['class']")
    print("model.load_data(expression_df, group_labels)")
    print()
    print("# 3. Feature selection: En önemli 50 gen")
    print("important_genes = model.select_important_genes(top_n=50)")
    print()
    print("# 4. Random Forest cross-validation")
    print("rf_scores = model.cross_validate_model(important_genes, model_type='rf')")
    print()
    print("# 5. SVM cross-validation")
    print("svm_scores = model.cross_validate_model(important_genes, model_type='svm')")
    print()
    print("# 6. Logistic Regression cross-validation")
    print("logreg_scores = model.cross_validate_model(important_genes, model_type='logreg')")
    print()
    print("# 7. XGBoost cross-validation")
    print("xgb_scores = model.cross_validate_model(important_genes, model_type='xgb')")
    print()
    print("# 8. XGBoost için eğitim & doğrulama eğrileri")
    print("model.xgboost_training_curves(important_genes)")
    print()
    print("# 9. Karşılaştırmalı değerlendirme")
    print("model.compare_all_models(important_genes)")
    print()
    
    # Kapsamlı analiz
    try:
        if os.path.exists(endo_file) and os.path.exists(auto_file):
            print("Excel dosyaları bulundu. Kapsamlı analiz başlatılıyor...")
            results = analyzer.comprehensive_biomarker_analysis(endo_file, auto_file)
            
            # Sonuçları gösterme
            print("\n=== EN İYİ BİYOMARKERLAR ===")
            top_biomarkers = results['biomarker_df'].head(20)
            for i, (_, row) in enumerate(top_biomarkers.iterrows(), 1):
                print(f"{i:2d}. {row['Gene']}")
            
            print("\n=== MODEL PERFORMANSI ===")
            print(results['performance_df'])
        else:
            print("Excel dosyaları bulunamadı. Kapsamlı analiz atlanıyor.")
    except Exception as e:
        print(f"Kapsamlı analiz hatası: {e}")
    
    print("\nKapsamlı analiz tamamlandı! Overfitting önleme ve feature selection uygulandı.")
    print("Yeni workflow metodları eklendi ve kullanıma hazır.") 
