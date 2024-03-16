import matplotlib.pyplot as plt

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, yes_counts, width, label='Yes', color='darkred')
bars2 = ax.bar(x + width/2, no_counts, width, label='No', color='lightred')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Counts')
ax.set_title('Responses to biking habits and safety')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

ax.bar_label(bars1, padding=3)
ax.bar_label(bars2, padding=3)

fig.tight_layout()

plt.show()
