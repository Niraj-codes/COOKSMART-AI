import pandas as pd

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    
    # Remove any missing or NaN values
    df.dropna(subset=['ingredients'], inplace=True)

    # Clean ingredients column
    df['ingredients'] = df['ingredients'].str.replace('[^a-zA-Z, ]', '').str.lower()

    # Save the processed data
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_data('C:/Users/Niraj Prajapati/Desktop/ML PROJECT/data/raw/cuisines.csv', 'C:/Users/Niraj Prajapati/Desktop/ML PROJECT/data/processed/processed_dataset.csv')
