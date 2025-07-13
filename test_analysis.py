#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for biomarker analysis
"""

import os
import sys
from biomarker_improved import EndometriosisBiomarkerAnalysis

def main():
    print("Test analizi başlatılıyor...")
    
    # Excel dosya yolları
    endo_file = "C:/Users/bilge/Desktop/bilge/bilge/endo_dgea_all.xlsx"
    auto_file = "C:/Users/bilge/Desktop/bilge/bilge/autoimmune_dgea_all.xlsx"
    
    # Analizi başlatma
    analyzer = EndometriosisBiomarkerAnalysis()
    
    try:
        if os.path.exists(endo_file) and os.path.exists(auto_file):
            print("Excel dosyaları bulundu. Kapsamlı analiz başlatılıyor...")
            results = analyzer.comprehensive_biomarker_analysis(endo_file, auto_file)
            
            # Incremental curves çizimi
            print("\nIncremental curves çiziliyor...")
            X = results['expression_matrix']
            y = results['group_labels']
            analyzer.plot_incremental_curves_for_all_models(X, y, analyzer.output_dir)
            
            print(f"\nSonuçlar kaydedildi: {analyzer.output_dir}")
            
            # Dosyaların var olup olmadığını kontrol et
            files_to_check = [
                "model_performance.csv",
                "biomarkers.csv",
                "random_forest_genes.csv",
                "ridge_genes.csv",
                "svm_genes.csv",
                "xgboost_genes.csv",
                "lasso_genes.csv",
                "model_performance_comparison.png",
                "top_biomarkers.png",
                "training_curves_all_models.png",
                "roc_curves_all_models.png",
                "feature_importance_comparison.png",
                "random_forest_regularized_curve.png",
                "incremental_curves_svm.png",
                "incremental_curves_logistic_regression.png",
                "incremental_curves_xgboost.png",
                "rf_roc_curve_gridsearch.png",
                "rf_confusion_matrix_gridsearch.png",
                "rf_classification_report_gridsearch.txt",
                "rf_gridsearch_results.csv",
                "rf_feature_importance_gridsearch.png",
                "rf_feature_importance_gridsearch.csv",
                "rf_performance_summary_gridsearch.json"
            ]
            
            print("\nKaydedilen dosyalar:")
            for file_name in files_to_check:
                file_path = analyzer.output_dir / file_name
                if file_path.exists():
                    print(f"✓ {file_name}")
                else:
                    print(f"✗ {file_name} (bulunamadı)")
            
            print("\nAnaliz başarıyla tamamlandı!")
            
            # --- RF Benchmark ---
            print("\nRandom Forest benchmark başlatılıyor...")
            X_bench = results['expression_matrix']
            y_bench = (results['group_labels'] == 'Endometriosis').astype(int)
            feature_names_bench = results['selected_genes'] if 'selected_genes' in results else [str(i) for i in range(X_bench.shape[1])]
            summary_df = analyzer.run_comprehensive_rf_benchmark(X_bench, y_bench, feature_names_bench)
            print("\nRF benchmark özeti kaydedildi:", analyzer.output_dir / 'rf_benchmark' / 'rf_benchmark_summary.csv')
            
        else:
            print("Excel dosyaları bulunamadı.")
            
    except Exception as e:
        print(f"Hata oluştu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 