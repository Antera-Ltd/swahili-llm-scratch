import matplotlib.pyplot as plt


epochs = list(range(1, 16))  # 1 to 15 epoch
loss_values = [
    10.0, 8.2, 7.1, 6.3, 5.6,
    5.0, 4.5, 4.1, 3.8, 3.5,
    3.2, 3.0, 2.8, 2.6, 2.5
]

# Create plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, loss_values, color='#2c3e50', marker='o', linewidth=2, markersize=4)

# Labels and title
plt.title('Training Loss Curve', fontsize=14, pad=15)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)

# Grid and limits
plt.grid(alpha=0.3)
plt.ylim(0, 11)
plt.xticks(range(0, 16, 2))

# Add start and end annotation
plt.annotate('Start: 10.0', xy=(1, 10.0), xytext=(2, 9),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('End: 2.5', xy=(15, 2.5), xytext=(12, 4),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Save as image
plt.tight_layout()
plt.savefig('training_loss_curve.png', dpi=300)
plt.show()