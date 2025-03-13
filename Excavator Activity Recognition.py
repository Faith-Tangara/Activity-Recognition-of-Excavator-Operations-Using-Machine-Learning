import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

# 1. Improved heuristic labeling with smoothing
def improved_heuristic_label(df, window_size=5):
    """Apply heuristic labeling with temporal smoothing to reduce noise"""
    print(f"Applying heuristic labeling to dataset with {len(df)} rows...")
    
    # Define thresholds for activities
    IDLE_THRESHOLD = 0.4
    SWING_THRESHOLD = 1.5
    DIG_BOOM_THRESHOLD = 0.8
    DIG_STICK_THRESHOLD = 0.8
    DIG_BUCKET_THRESHOLD = -0.5
    DUMP_BUCKET_THRESHOLD = 0.5
    RELOCATE_THRESHOLD = 0.8
    
    # Apply initial heuristic labeling
    def label_single_row(row):
        if (abs(row['Platform Speed']) < IDLE_THRESHOLD and 
            abs(row['Boom Speed']) < IDLE_THRESHOLD and 
            abs(row['Stick Speed']) < IDLE_THRESHOLD and 
            abs(row['Bucket Speed']) < IDLE_THRESHOLD):
            return 'idle'
        elif abs(row['Platform Speed']) > SWING_THRESHOLD:
            return 'swinging'
        elif ((abs(row['Boom Speed']) > DIG_BOOM_THRESHOLD or 
              abs(row['Stick Speed']) > DIG_STICK_THRESHOLD) and 
              row['Bucket Speed'] < DIG_BUCKET_THRESHOLD):
            return 'digging'
        elif row['Bucket Speed'] > DUMP_BUCKET_THRESHOLD:
            return 'dumping'
        elif (RELOCATE_THRESHOLD < abs(row['Platform Speed']) <= SWING_THRESHOLD):
            return 'relocating'
        else:
            return 'idle'
    
    # Apply initial labeling
    df['raw_label'] = df.apply(label_single_row, axis=1)
    
    # Apply smoothing using rolling window majority vote
    def smooth_labels(labels, window_size):
        smoothed = []
        labels_list = list(labels)
        for i in range(len(labels_list)):
            # Get window around current point
            start = max(0, i - window_size // 2)
            end = min(len(labels_list), i + window_size // 2 + 1)
            window = labels_list[start:end]
            # Count occurrences of each label in window
            counts = {}
            for label in window:
                counts[label] = counts.get(label, 0) + 1
            # Find majority label
            majority_label = max(counts.items(), key=lambda x: x[1])[0]
            smoothed.append(majority_label)
        return smoothed
    
    # Apply smoothing
    df['Label'] = smooth_labels(df['raw_label'], window_size)
    
    # Show example of smoothing effect
    if len(df) > 10:
        sample_idx = min(100, len(df) - 10)  # Get a sample index
        print("Example of smoothing effect:")
        print("Before smoothing:", df['raw_label'].iloc[sample_idx:sample_idx+10].tolist())
        print("After smoothing:", df['Label'].iloc[sample_idx:sample_idx+10].tolist())
    
    return df

# 2. Save labeled data to Excel
def save_labeled_data(df, filename):
    """Save the labeled data to an Excel file"""
    output_filename = f"Labeled_{filename}"
    print(f"Saving labeled data to {output_filename}...")
    df.to_excel(output_filename, index=False)
    return output_filename

# 3. Feature importance analysis
def analyze_feature_importance(model, feature_names):
    """Analyze and visualize feature importance from the model"""
    print("\nAnalyzing feature importance...")
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Print feature ranking
    print("Feature ranking:")
    for i in range(len(feature_names)):
        print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]:.4f})")
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    
    return indices, importances

# 4. Train Random Forest model
def train_rf_model(X_train, y_train, n_estimators=100, max_depth=10):
    """Train a Random Forest model with specified parameters"""
    print(f"Training Random Forest model with {n_estimators} trees and max_depth={max_depth}...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    rf.fit(X_train, y_train)
    return rf

# 5. Cross-testing performance analysis
def cross_testing_analysis(models, datasets, feature_cols):
    """Analyze performance across different training/testing combinations"""
    print("\nPerforming cross-testing analysis...")
    
    # Create results table
    results = []
    
    # Test each model on each dataset
    for train_name, model in models.items():
        for test_name, (X_test, y_test) in datasets.items():
            # Make predictions
            y_pred = model.predict(X_test[feature_cols])
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store results
            results.append({
                'Trained On': train_name,
                'Tested On': test_name,
                'Accuracy': accuracy,
                'y_true': y_test,
                'y_pred': y_pred
            })
    
    # Create DataFrame from results
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_true', 'y_pred']} 
                              for r in results])
    
    # Print results table
    print("\nCross-testing results:")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Save results to Excel
    results_df.to_excel("Cross_Testing_Results.xlsx", index=False)
    print("Cross-testing results saved to Cross_Testing_Results.xlsx")
    
    return results

# 6. Plot confusion matrices
def plot_confusion_matrices(results, labels):
    """Create confusion matrix visualization for cross-testing results"""
    print("\nCreating confusion matrix visualization...")
    
    # Determine grid size based on number of results
    n_results = len(results)
    n_cols = min(3, n_results)
    n_rows = (n_results + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    fig.suptitle('Confusion Matrices of Excavator Activity Classification', fontsize=16)
    
    # Flatten axes if needed
    if n_rows > 1 and n_cols > 1:
        axes = axes.flatten()
    elif n_rows == 1 and n_cols > 1:
        axes = axes  # Already a 1D array
    elif n_rows > 1 and n_cols == 1:
        axes = axes.flatten()
    else:
        axes = np.array([axes])  # Make it indexable
    
    # Plot each confusion matrix
    for i, result in enumerate(results):
        # Get axis
        ax = axes[i] if n_results > 1 else axes
        
        # Create confusion matrix
        cm = confusion_matrix(result['y_true'], result['y_pred'], labels=labels)
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        
        # Set title and labels
        ax.set_title(f"{result['Trained On']} Model on {result['Tested On']} Data\nAccuracy: {result['Accuracy']:.4f}")
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
    
    # Remove extra subplots
    for i in range(n_results, n_rows * n_cols):
        if i < len(axes):
            fig.delaxes(axes[i])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    plt.savefig("Confusion_Matrices.png", dpi=300, bbox_inches='tight')
    print("Confusion matrices saved to Confusion_Matrices.png")
    
    return fig

# 7. Main function to run the complete analysis
def main():
    print("Loading all 4 datasets...")
    
    try:
        # Load all datasets
        novice_df = pd.read_excel('Data based on Operators Skill Level - NOVICE.xlsx')
        expert_df = pd.read_excel('Data based on Operators Skill Level - EXPERT.xlsx')
        excavator_cx_df = pd.read_excel('Data based on the Geometry of the excavator - Excavator Case CX80C.xlsx')
        excavator_tc_df = pd.read_excel('Data based on the Geometry of the excavator -Excavator Terex TC75.xlsx')
        
        print(f"Loaded datasets: "
              f"Novice ({len(novice_df)} rows), "
              f"Expert ({len(expert_df)} rows), "
              f"Excavator CX80C ({len(excavator_cx_df)} rows), "
              f"Excavator TC75 ({len(excavator_tc_df)} rows)")
    
    except Exception as e:
        print(f"Error loading data: {e}")
        
        # Create sample data for demonstration
        print("Creating sample data for demonstration...")
        np.random.seed(42)
        
        # Create features
        features = ['Platform', 'Platform Speed', 'Boom', 'Boom Speed', 'Stick', 'Stick Speed', 'Bucket', 'Bucket Speed']
        
        # Create sample dataframes
        novice_df = pd.DataFrame(np.random.randn(1000, len(features)), columns=features)
        expert_df = pd.DataFrame(np.random.randn(1500, len(features)), columns=features)
        excavator_cx_df = pd.DataFrame(np.random.randn(1200, len(features)), columns=features)
        excavator_tc_df = pd.DataFrame(np.random.randn(1300, len(features)), columns=features)
    
    # Apply improved heuristic labeling with smoothing to all datasets
    print("\nApplying heuristic labeling to all datasets...")
    novice_df = improved_heuristic_label(novice_df)
    expert_df = improved_heuristic_label(expert_df)
    excavator_cx_df = improved_heuristic_label(excavator_cx_df)
    excavator_tc_df = improved_heuristic_label(excavator_tc_df)
    
    # Save labeled datasets to Excel
    save_labeled_data(novice_df, "Novice_Dataset.xlsx")
    save_labeled_data(expert_df, "Expert_Dataset.xlsx")
    save_labeled_data(excavator_cx_df, "Excavator_CX80C_Dataset.xlsx")
    save_labeled_data(excavator_tc_df, "Excavator_TC75_Dataset.xlsx")
    
    # Define features
    features = ['Platform', 'Platform Speed', 'Boom', 'Boom Speed', 'Stick', 'Stick Speed', 'Bucket', 'Bucket Speed']
    
    # Split data for all datasets
    print("\nSplitting all datasets into training and testing sets...")
    X_train_nov, X_test_nov, y_train_nov, y_test_nov = train_test_split(
        novice_df[features], novice_df['Label'], test_size=0.3, random_state=42)
    
    X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(
        expert_df[features], expert_df['Label'], test_size=0.3, random_state=42)
    
    X_train_cx, X_test_cx, y_train_cx, y_test_cx = train_test_split(
        excavator_cx_df[features], excavator_cx_df['Label'], test_size=0.3, random_state=42)
    
    X_train_tc, X_test_tc, y_train_tc, y_test_tc = train_test_split(
        excavator_tc_df[features], excavator_tc_df['Label'], test_size=0.3, random_state=42)
    
    # Train models for all datasets
    print("\nTraining models for all datasets...")
    novice_model = train_rf_model(X_train_nov, y_train_nov)
    expert_model = train_rf_model(X_train_exp, y_train_exp)
    cx_model = train_rf_model(X_train_cx, y_train_cx)
    tc_model = train_rf_model(X_train_tc, y_train_tc)
    
    # Analyze feature importance for one of the models
    plt.figure(figsize=(12, 6))
    analyze_feature_importance(novice_model, features)
    plt.savefig("Feature_Importance.png", dpi=300, bbox_inches='tight')
    print("Feature importance plot saved to Feature_Importance.png")
    
    # FIXED: Perform cross-testing analysis with individual models and datasets only
    # This prevents data leakage by not using the combined dataset
    models = {
        'Novice': novice_model,
        'Expert': expert_model,
        'Excavator_CX80C': cx_model,
        'Excavator_TC75': tc_model
    }
    
    datasets = {
        'Novice': (pd.DataFrame(X_test_nov), y_test_nov),
        'Expert': (pd.DataFrame(X_test_exp), y_test_exp),
        'Excavator_CX80C': (pd.DataFrame(X_test_cx), y_test_cx),
        'Excavator_TC75': (pd.DataFrame(X_test_tc), y_test_tc)
    }
    
    # Run cross-testing analysis
    results = cross_testing_analysis(models, datasets, features)
    
    # Plot confusion matrices
    labels = ['idle', 'swinging', 'digging', 'dumping', 'relocating']
    plot_confusion_matrices(results, labels)
    
    # Create a summary of activity distribution
    print("\nActivity distribution in labeled datasets:")
    activity_summary = pd.DataFrame({
        'Novice': novice_df['Label'].value_counts(),
        'Expert': expert_df['Label'].value_counts(),
        'Excavator_CX80C': excavator_cx_df['Label'].value_counts(),
        'Excavator_TC75': excavator_tc_df['Label'].value_counts()
    })
    
    print(activity_summary)
    
    # Save activity distribution to Excel
    activity_summary.to_excel("Activity_Distribution.xlsx")
    print("Activity distribution saved to Activity_Distribution.xlsx")
    
    # Plot activity distribution
    plt.figure(figsize=(12, 8))
    activity_summary.plot(kind='bar')
    plt.title('Activity Distribution Across Datasets')
    plt.xlabel('Activity')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("Activity_Distribution.png", dpi=300, bbox_inches='tight')
    print("Activity distribution plot saved to Activity_Distribution.png")
    
    print("\nAnalysis complete! All results have been saved to Excel files and plots.")
    
    print("\nSummary of files created:")
    print("1. Labeled_*.xlsx - Datasets with applied heuristic labels")
    print("2. Cross_Testing_Results.xlsx - Performance metrics for all model-dataset combinations")
    print("3. Confusion_Matrices.png - Visualization of confusion matrices")
    print("4. Feature_Importance.png - Feature importance analysis")
    print("5. Activity_Distribution.xlsx - Distribution of activities across datasets")
    print("6. Activity_Distribution.png - Visualization of activity distribution")

if __name__ == "__main__":
    main()