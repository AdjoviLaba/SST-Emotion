import os
import json
import matplotlib.pyplot as plt
import numpy as np

def plot_accuracies():
    if not os.path.exists('result/all_meta_result.txt'):
        print("No meta result file found.")
        return
        
    subjects = []
    accuracies = []
    with open('result/all_meta_result.txt', 'r') as f:
        for line in f:
            if ':' in line:
                parts = line.strip().split(':')
                subjects.append(parts[0].strip())
                accuracies.append(float(parts[1].strip()) * 100)
                
    if not accuracies:
        print("No data in all_meta_result.txt yet.")
        return

    # Modern aesthetic plot for accuracy
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 6))
    bars = plt.bar(subjects, accuracies, color='#4C72B0', edgecolor='black', alpha=0.8)
    
    avg_acc = np.mean(accuracies)
    plt.axhline(avg_acc, color='#C44E52', linestyle='--', linewidth=2, label=f'Average ({avg_acc:.1f}%)')
    
    plt.ylim(0, 100)
    plt.ylabel('LOSO Test Accuracy (%)', fontsize=12, fontweight='bold')
    plt.xlabel('Hold-out Subjects', fontsize=12, fontweight='bold')
    plt.title('Reptile Meta-Learning Cross-Subject Generalization', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('result/accuracy_bar_chart.png', dpi=300)
    print("Saved accuracy_bar_chart.png")

def plot_learning_curves():
    for i in range(15):
        hist_file = f'result/Sub_{i}_history.json'
        if os.path.exists(hist_file):
            with open(hist_file, 'r') as f:
                hist = json.load(f)
                
            plt.style.use('seaborn-darkgrid')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot Accuracy
            ax1.plot(hist['accuracy'], label='Train Accuracy', color='#55A868', linewidth=2.5)
            ax1.plot(hist['val_accuracy'], label='Validation Accuracy', color='#C44E52', linewidth=2.5)
            ax1.set_title(f'Subject {i} - Accuracy Curve (Fine-Tuning)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Epochs', fontsize=12)
            ax1.set_ylabel('Accuracy', fontsize=12)
            ax1.legend(fontsize=11)
            
            # Plot Loss
            ax2.plot(hist['loss'], label='Train Loss', color='#55A868', linewidth=2.5)
            ax2.plot(hist['val_loss'], label='Validation Loss', color='#C44E52', linewidth=2.5)
            ax2.set_title(f'Subject {i} - Loss Curve (Fine-Tuning)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epochs', fontsize=12)
            ax2.set_ylabel('Loss', fontsize=12)
            ax2.legend(fontsize=11)
            
            plt.tight_layout()
            plt.savefig(f'result/Sub_{i}_learning_curves.png', dpi=300)
            plt.close()
            print(f"Saved learning curves for Subject {i}")

if __name__ == '__main__':
    # Ensure matplotlib uses Agg backend to avoid GUI issues
    import matplotlib
    matplotlib.use('Agg')
    
    os.makedirs('result', exist_ok=True)
    plot_accuracies()
    plot_learning_curves()
