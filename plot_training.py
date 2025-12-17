"""
Generate Training Graphs for PPT
=================================
This script reads the training history and creates professional
accuracy/loss graphs to include in your presentation.

Author: Devansh
"""

import json
import matplotlib.pyplot as plt
import os

# Style settings for professional look
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

HISTORY_PATH = "training_history.json"
OUTPUT_DIR = "training_graphs"

def load_history():
    """Load training history from JSON file."""
    if not os.path.exists(HISTORY_PATH):
        print(f"Error: {HISTORY_PATH} not found!")
        print("Please run 'python train_model.py' first.")
        return None
    
    with open(HISTORY_PATH, 'r') as f:
        return json.load(f)

def plot_accuracy(history, save_path):
    """Generate accuracy graph."""
    epochs = range(1, len(history['accuracy']) + 1)
    
    fig, ax = plt.subplots()
    
    ax.plot(epochs, history['accuracy'], 'b-o', linewidth=2, markersize=6, 
            label='Training Accuracy', color='#2196F3')
    ax.plot(epochs, history['val_accuracy'], 'r-s', linewidth=2, markersize=6,
            label='Validation Accuracy', color='#4CAF50')
    
    ax.set_title('Model Accuracy Over Training Epochs', fontweight='bold', pad=15)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0, 1])
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add final accuracy annotation
    final_train = history['accuracy'][-1]
    final_val = history['val_accuracy'][-1]
    ax.annotate(f'Final: {final_train:.1%}', 
                xy=(len(epochs), final_train),
                xytext=(len(epochs)-2, final_train+0.05),
                fontsize=10, color='#2196F3')
    ax.annotate(f'Final: {final_val:.1%}', 
                xy=(len(epochs), final_val),
                xytext=(len(epochs)-2, final_val-0.08),
                fontsize=10, color='#4CAF50')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

def plot_loss(history, save_path):
    """Generate loss graph."""
    epochs = range(1, len(history['loss']) + 1)
    
    fig, ax = plt.subplots()
    
    ax.plot(epochs, history['loss'], 'b-o', linewidth=2, markersize=6,
            label='Training Loss', color='#F44336')
    ax.plot(epochs, history['val_loss'], 'r-s', linewidth=2, markersize=6,
            label='Validation Loss', color='#FF9800')
    
    ax.set_title('Model Loss Over Training Epochs', fontweight='bold', pad=15)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

def plot_combined(history, save_path):
    """Generate combined accuracy and loss graph."""
    epochs = range(1, len(history['accuracy']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(epochs, history['accuracy'], 'b-o', linewidth=2, markersize=5,
             label='Training', color='#2196F3')
    ax1.plot(epochs, history['val_accuracy'], 'g-s', linewidth=2, markersize=5,
             label='Validation', color='#4CAF50')
    ax1.set_title('Accuracy', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([0, 1])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(epochs, history['loss'], 'r-o', linewidth=2, markersize=5,
             label='Training', color='#F44336')
    ax2.plot(epochs, history['val_loss'], 'o-s', linewidth=2, markersize=5,
             label='Validation', color='#FF9800')
    ax2.set_title('Loss', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle('YAMNet Fine-Tuning: Training Progress', fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

def main():
    print("=" * 50)
    print("GENERATING TRAINING GRAPHS FOR PPT")
    print("=" * 50)
    
    # Load history
    history = load_history()
    if history is None:
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate graphs
    print("\nGenerating graphs...")
    plot_accuracy(history, os.path.join(OUTPUT_DIR, "training_accuracy.png"))
    plot_loss(history, os.path.join(OUTPUT_DIR, "training_loss.png"))
    plot_combined(history, os.path.join(OUTPUT_DIR, "training_combined.png"))
    
    # Summary
    print("\n" + "=" * 50)
    print("GRAPHS GENERATED SUCCESSFULLY!")
    print("=" * 50)
    print(f"\nFiles saved in '{OUTPUT_DIR}/' folder:")
    print("  - training_accuracy.png  (for PPT)")
    print("  - training_loss.png      (for PPT)")
    print("  - training_combined.png  (for PPT)")
    
    # Print training summary
    final_acc = history['val_accuracy'][-1]
    final_loss = history['val_loss'][-1]
    best_acc = max(history['val_accuracy'])
    
    print(f"\nTraining Summary:")
    print(f"  - Total Epochs: {len(history['accuracy'])}")
    print(f"  - Final Validation Accuracy: {final_acc:.2%}")
    print(f"  - Best Validation Accuracy: {best_acc:.2%}")
    print(f"  - Final Validation Loss: {final_loss:.4f}")

if __name__ == "__main__":
    main()
