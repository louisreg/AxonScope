import os
import pandas as pd

def append_to_csv(df, filepath="benchmark.csv"):
    """
    Append or update a DataFrame in an existing CSV file, or create it if it doesn't exist.
    If the 'label' column in df already exists in the CSV, the old rows are replaced.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to append or update. Must contain a 'label' column.
    filepath : str, optional
        Name of the CSV file (default: 'benchmark.csv').

    Returns
    -------
    pandas.DataFrame
        The combined DataFrame saved to CSV.
    """
    # Resolve the path relative to the script's own directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, filepath)

    # If the file exists, load it
    if os.path.exists(full_path):
        existing_df = pd.read_csv(full_path)

        # Remove rows with the same labels as in the new df
        combined_df = pd.concat([
            existing_df[~existing_df['label'].isin(df['label'])],
            df
        ], ignore_index=True)
    else:
        combined_df = df.copy()

    # Write back to CSV
    combined_df.to_csv(full_path, index=False)

    return combined_df



def res_to_df(N_vec, T_vec, label) -> pd.DataFrame: 
    df = pd.DataFrame({
        "label": [label] * len(N_vec),
        "N": N_vec,
        "time": T_vec
    })
    
    return df