import matplotlib.pyplot as plt
import os


def plot_sequence_metrics(
    metrics_data, 
    benchmarks=None, 
    figsize=(15, 10),
    path=None
):

    # Create figure and axes
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Define plot styles
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'
    ]
    line_width = 2
    
    # Map of metrics to plot positions and titles
    plot_config = {
        'disorder': {'pos': 0, 'title': 'Disorder', 'color': colors[0]},
        'dipolar': {'pos': 1, 'title': 'Dipolar', 'color': colors[1]},
        'rotation': {'pos': 2, 'title': 'Rotation', 'color': colors[2]},
        'finite_disorder': {'pos': 3, 'title': 'Finite Disorder', 'color': colors[3]},
        'finite_dipolar': {'pos': 4, 'title': 'Finite Dipolar', 'color': colors[4]}
    }
    data_length = len(metrics_data['disorder'])
    x_axis = list(range(1, data_length+1))
    # Plot each metric
    for metric_name, config in plot_config.items():
        if metric_name in metrics_data:
            data = metrics_data[metric_name]
            ax = axes[config['pos']]
            
            # Plot data
            ax.plot(x_axis, data, color=config['color'], linewidth=line_width)

            # Plot benchmark metric
            if benchmarks:
                for i, benchmark in enumerate(benchmarks):
                    colors = ['black', 'yellow', 'purple']
                    ax.plot(
                        x_axis, 
                        benchmark[metric_name], 
                        color=colors[i], 
                        linewidth=line_width,
                        linestyle='--',
                        label=f'Benchmark: {benchmark["name"]}'
                    )
                max_benchmark = max(benchmark[metric_name])
            
            # Set title and labels
            ax.set_title(config['title'], fontsize=16, fontweight='bold')
            ax.set_xlabel('Cycle', fontsize=15)
            ax.set_ylabel('Score', fontsize=15)
            
            
            #Set y-axis limits
            max_val = max(data) if data and len(data) > 0 else 0
            max_val = max(max_val, max_benchmark)
            max_val = max(max_val, 1e-5)  # Ensure minimum visibility
            ax.set_ylim(0, max_val * 1.1)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(prop={'size': 18})  # Increased legend font size
            
            # Add spines
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)
    
    # Hide any unused subplots
    for i in range(len(plot_config), len(axes)):
        axes[i].set_visible(False)
    
    # Add a title for the entire figure
    fig.suptitle(
        f'AHT Scores for Sequence {metrics_data["name"]}', 
        fontsize=16, 
        fontweight='bold', 
        y=0.98
    )
    
    # Adjust layout and show
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
    if path:
        # Check if the directory exists, create it if it doesn't
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(path)
        plt.close()
    plt.show()
