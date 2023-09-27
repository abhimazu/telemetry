import os
import pandas as pd
import matplotlib.pyplot as plt

class statsPrompt:

    def __init__(self, csv_dir='.') -> None:
        self.csv_dir = csv_dir


    def process_csv(self, file_path):
        """
        Process a given CSV file: load, clean, and return summary statistics.
        """
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Convert percentage columns to numeric values
        df['cpu'] = df['cpu'].str.rstrip('%').astype('float') / 100.0
        df['gpu'] = df['gpu'].str.rstrip('%').astype('float') / 100.0
        df['gpu video'] = df['gpu video'].str.rstrip('%').astype('float') / 100.0
        df['gpu enhance'] = df['gpu enhance'].str.rstrip('%').astype('float') / 100.0

        # Convert memory usage to a numeric value
        df['mem'] = df['mem'].str.rstrip(' GB').astype('float')

        # Drop rows with NaN values resulting from non-numeric values
        df.dropna(inplace=True)
        
        # Calculate and return summary statistics
        # print(df.describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']])

        return df.describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]

    def plot_data(self, raw_data_dict):

        # Create a vertically stacked single plot for CPU, Memory, and GPU usages
        fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

        # CPU Usage
        for device, data in raw_data_dict.items():
            axes[0].plot(data['cpu'], label=device)
        axes[0].set_title('CPU Usage across Devices')
        axes[0].set_ylabel('CPU Usage')
        axes[0].legend()
        axes[0].grid(True)

        # Memory Usage
        for device, data in raw_data_dict.items():
            axes[1].plot(data['mem'], label=device)
        axes[1].set_title('Memory Usage across Devices')
        axes[1].set_ylabel('Memory Usage (GB)')
        axes[1].legend()
        axes[1].grid(True)

        # GPU Usage
        for device, data in raw_data_dict.items():
            axes[2].plot(data['gpu'], label=device)
        axes[2].set_title('GPU Usage across Devices')
        axes[2].set_ylabel('GPU Usage')
        axes[2].set_xlabel('Time Index')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()
        plt.show()

    def generate_prompt_for_llm(self, summary_dict):
        """
        Generate a prompt for LLM based on the summary statistics.
        """
        prompts = []

        for device, stats in summary_dict.items():
            prompt = f"For the device {device}, "
            for metric in stats.columns:
                metric_summary = ", ".join([f"{stat}: {value:.2f}" for stat, value in stats[metric].items()])
                prompt += f"the {metric} has {metric_summary}. "
            prompts.append(prompt)

        return "\n\n".join(prompts)


    def generate_prompt_data(self):

        # Dictionary to hold summary statistics for each CSV file
        summary_dict = {}

        csv_files = [file for file in os.listdir(self.csv_dir) if file.endswith('.csv')]

        # Process each CSV file again
        for file in csv_files:
            device_name = os.path.splitext(file)[0]  # Use the filename as the device name
            path = self.csv_dir + '/' + file
            summary_dict[device_name] = self.process_csv(f'{path}')

        # Display the processed devices and their respective data shapes to ensure data was loaded
        processed_devices = {device: summary.shape for device, summary in summary_dict.items()}

        print("Processed Files: ", processed_devices)

        # comparison_df = pd.concat(summary_dict).unstack(level=0)

        # print(comparison_df)

        # plot_data(summary_dict)

        # Generate the prompt
        prompt = self.generate_prompt_for_llm(summary_dict)

        # print(prompt)

        return prompt


if __name__=='__main__':

    gen_prompt = statsPrompt('files')
    prompt = gen_prompt.generate_prompt_data()
    print(prompt)
