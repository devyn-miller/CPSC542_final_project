import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataExplorer:
    def __init__(self, train_path, test_path, validation_path):
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.validation_df = pd.read_csv(validation_path)

    def display_head(self):
        print(self.train_df.head())

    def summary_statistics(self):
        print(self.train_df.describe())

    def check_missing_values(self):
        print(self.train_df.isnull().sum())

    def feature_distribution(self, feature_name):
        sns.histplot(self.train_df[feature_name], kde=True)
        plt.title(f'Distribution of {feature_name}')
        plt.show()

    def visualize_outliers(self, feature_name):
        sns.boxplot(x=self.train_df[feature_name])
        plt.title(f'Boxplot of {feature_name}')
        plt.show()

    def correlation_matrix(self):
        corr = self.train_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

# Example usage:
# explorer = DataExplorer('../data/train.csv', '../data/test.csv', '../data/validation.csv')
# explorer.display_head()
# explorer.summary_statistics()
# explorer.check_missing_values()
# explorer.feature_distribution('feature_name')  # Replace 'feature_name' with actual feature name
# explorer.visualize_outliers('feature_name')  # Replace 'feature_name' with actual feature name
# explorer.correlation_matrix()

