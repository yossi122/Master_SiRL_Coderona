import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

df = pd.read_csv('Holon.csv', index_col=0).drop_duplicates()
df_sorted = df.loc[df.index.sort_values()]
print(df.shape)
print(df.head())
print(df.columns)

def plot_population_stats(df, cols = [], peak_thresh = [], valley_thresh = [], save_path=''):
    ax = df[cols].plot()
    for line in ax.lines:
        line.set_linewidth(2.0)

    if peak_thresh is None or len(peak_thresh) == 0:
        peak_thresh = [0.1] * len(cols)

    plt.title(f'Plot of {cols}')
    plt.xlabel('Index')
    plt.xticks(rotation=90)

    previous_episode = None
    for i, index in enumerate(df.index):
        # Extract episode from index assuming the format is "episode_x_day_y"
        episode = index.split('_')[1]
        if episode != previous_episode:
            plt.axvline(x=i, color='r', linestyle='--', linewidth=0.7)
            previous_episode = episode

    for idx, c in enumerate(cols):
        print(idx, cols[idx], peak_thresh[idx])
        # Find local maxima and minima
        peaks, _ = find_peaks(df[c],  height=peak_thresh[idx])  # Adjust prominence as needed
        valleys, _ = find_peaks(-df[c],  height=-valley_thresh[idx])  # Adjust prominence as needed

        if len(peaks) > 0:
            # Plot vertical lines for each peak
            for peak in peaks:
                ax.axvline(x=peak, color='y',
                           linestyle='--',
                           #label=f'{c}:{df[c][peak]}'
                           )
                # Mark points with values
                plt.scatter(peak, df[c].iloc[peak], marker='o', color='r')
                plt.text(peak, df[c].iloc[peak], f'{df[c].iloc[peak]:.5f}',
                         verticalalignment='bottom',
                         horizontalalignment='left')
        if len(valleys) > 0:
            # Plot vertical lines for each peak
            for v in valleys:
                ax.axvline(x=v, color='m',
                           linestyle='-.',
                           # label=f'{c}:{df[c][peak]}'
                           )
                # Mark points with values
                plt.scatter(v, df[c].iloc[v], marker='o', color='b')
                plt.text(v, df[c].iloc[v], f'{df[c].iloc[v]:.5f}',
                         verticalalignment='bottom',
                         horizontalalignment='left')

            # Create custom legend entries with point values
        '''
        handles, labels = ax.get_legend_handles_labels()
        point_labels = [f'{label}\nValue: {df[cols].iloc[int(label.split()[-1])]}'
                        if 'Local Maxima' in label else label
                        for label in labels]
        ax.legend(handles, point_labels)
        '''
    plt.tight_layout()
    plt.ylabel('Values')
    plt.legend(loc='lower center')
    #ax.legend(             loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True,
    #          ncol=len(cols))
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()





'''
plot_population_stats(df_sorted,
                      cols=['infected_population', 'vaccinated_population'],
                      peak_thresh=[0.00005, 0.0005])
plot_population_stats(df_sorted, cols=['compliance', 'epsilon'])
'''
plot_population_stats(df_sorted, cols=['mean_reward', 'reward'],
                      peak_thresh=[39000000, 39000000], valley_thresh = [21000000, 300000],
                      save_path ='rewards.jpg')

