import pandas as pd


def complete_nan_meteo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Completes NaN values in columns containing 'meteo' in their names.
    - Numeric columns are filled with the median value grouped by 'piezo_station_department_code'.
    - Categorical columns are filled with the mode value grouped by 'piezo_station_department_code'.
    
    Args:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: The DataFrame with NaN values filled for 'meteo' columns.
    """
    # Identify columns related to "meteo"
    meteo_columns = [col for col in df.columns if "meteo" in col]
    
    # Separate numeric and categorical "meteo" columns
    numeric_meteo_columns = df[meteo_columns].select_dtypes(include=['float64', 'int64']).columns
    categorical_meteo_columns = df[meteo_columns] \
        .select_dtypes(include=['object', 'category']) \
        .columns    
    # Group by department name
    grouped = df.groupby('piezo_station_department_code')
    
    # Fill NaN in numeric columns with median
    for col in numeric_meteo_columns:
        df[col] = grouped[col].transform(lambda x: x.fillna(x.median()))
    
    # Fill NaN in categorical columns with mode
    for col in categorical_meteo_columns:
        df[col] = grouped[col].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else None)
        )
    
    return df


def complete_nan_prev(df: pd.DataFrame) -> pd.DataFrame:
    """
    Completes NaN values in columns containing 'prelev' in their names.
    - Numeric columns are filled with the median value grouped by 'piezo_station_department_code'.
    - Categorical columns are filled with the mode value grouped by 'piezo_station_department_code'.
    
    Args:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: The DataFrame with NaN values filled for 'prelev' columns.
    """
    # Identify columns related to "prelev"
    prelev_columns = [col for col in df.columns if "prelev" in col]
    print(prelev_columns)
    # Separate numeric and categorical "prelv" columns
    numeric_prelev_columns = df[prelev_columns].select_dtypes(include=['float64', 'int64']).columns
    categorical_prelev_columns = df[prelev_columns] \
        .select_dtypes(include=['object', 'category']) \
        .columns    
    # Group by department name
    grouped = df.groupby('piezo_station_department_code')
    
    # Fill NaN in numeric columns with median
    for col in numeric_prelev_columns:
        df[col] = grouped[col].transform(lambda x: x.fillna(x.median()))
    
    # Fill NaN in categorical columns with mode
    for col in categorical_prelev_columns:
        df[col] = grouped[col].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown")
        )
    
    return df


# Function to map department codes to regions
def map_department_to_region(department_code: str) -> str:

    department_to_region = {
        # Auvergne-Rhône-Alpes
        "01": "Auvergne-Rhône-Alpes", "03": "Auvergne-Rhône-Alpes", "07": "Auvergne-Rhône-Alpes",
        "15": "Auvergne-Rhône-Alpes", "26": "Auvergne-Rhône-Alpes", "38": "Auvergne-Rhône-Alpes",
        "42": "Auvergne-Rhône-Alpes", "43": "Auvergne-Rhône-Alpes", "63": "Auvergne-Rhône-Alpes", 
        "69": "Auvergne-Rhône-Alpes", "73": "Auvergne-Rhône-Alpes", "74": "Auvergne-Rhône-Alpes",
        
        # Bourgogne-Franche-Comté
        "21": "Bourgogne-Franche-Comté", "25": "Bourgogne-Franche-Comté", 
        "39": "Bourgogne-Franche-Comté",
        "58": "Bourgogne-Franche-Comté", "70": "Bourgogne-Franche-Comté", 
        "71": "Bourgogne-Franche-Comté",
        "89": "Bourgogne-Franche-Comté", "90": "Bourgogne-Franche-Comté",

        # Bretagne
        "22": "Bretagne", "29": "Bretagne", "35": "Bretagne", "56": "Bretagne",

        # Centre-Val de Loire
        "18": "Centre-Val de Loire", "28": "Centre-Val de Loire", "36": "Centre-Val de Loire",
        "37": "Centre-Val de Loire", "41": "Centre-Val de Loire", "45": "Centre-Val de Loire",

        # Corse
        "2A": "Corse", "2B": "Corse",

        # Grand Est
        "08": "Grand Est", "10": "Grand Est", "51": "Grand Est", "52": "Grand Est", 
        "54": "Grand Est", "55": "Grand Est", "57": "Grand Est", "67": "Grand Est",
        "68": "Grand Est", "88": "Grand Est",

        # Hauts-de-France
        "02": "Hauts-de-France", "59": "Hauts-de-France", "60": "Hauts-de-France", 
        "62": "Hauts-de-France", "80": "Hauts-de-France",

        # Île-de-France
        "75": "Île-de-France", "77": "Île-de-France", "78": "Île-de-France", "91": "Île-de-France", 
        "92": "Île-de-France", "93": "Île-de-France", "94": "Île-de-France", "95": "Île-de-France",

        # Normandie
        "14": "Normandie", "27": "Normandie", "50": "Normandie", "61": "Normandie", 
        "76": "Normandie",

        # Nouvelle-Aquitaine
        "16": "Nouvelle-Aquitaine", "17": "Nouvelle-Aquitaine", "19": "Nouvelle-Aquitaine",
        "23": "Nouvelle-Aquitaine", "24": "Nouvelle-Aquitaine", "33": "Nouvelle-Aquitaine",
        "40": "Nouvelle-Aquitaine", "47": "Nouvelle-Aquitaine", "64": "Nouvelle-Aquitaine",
        "79": "Nouvelle-Aquitaine", "86": "Nouvelle-Aquitaine", "87": "Nouvelle-Aquitaine",

        # Occitanie
        "09": "Occitanie", "11": "Occitanie", "12": "Occitanie", "30": "Occitanie", 
        "31": "Occitanie", "32": "Occitanie", "34": "Occitanie", "46": "Occitanie",
        "48": "Occitanie", "65": "Occitanie", "66": "Occitanie", "81": "Occitanie", 
        "82": "Occitanie",

        # Pays de la Loire
        "44": "Pays de la Loire", "49": "Pays de la Loire", "53": "Pays de la Loire", 
        "72": "Pays de la Loire", "85": "Pays de la Loire",

        # Provence-Alpes-Côte d'Azur
        "04": "Provence-Alpes-Côte d'Azur", "05": "Provence-Alpes-Côte d'Azur", 
        "06": "Provence-Alpes-Côte d'Azur", "13": "Provence-Alpes-Côte d'Azur",
        "83": "Provence-Alpes-Côte d'Azur", "84": "Provence-Alpes-Côte d'Azur",

        # French Overseas Regions
        "971": "Guadeloupe", "972": "Martinique", "973": "Guyane", "974": "La Réunion", 
        "976": "Mayotte",
    }

    return department_to_region.get(department_code, "Unknown")


def create_region(df: pd.DataFrame) -> pd.DataFrame:
    # Dictionary mapping department codes to regions
    
    df['region'] = df['piezo_station_department_code'].apply(map_department_to_region)

    return df


def complete_nan_meteo_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Completes NaN values in columns containing 'meteo' in their names.
    - Numeric columns are filled with the median value grouped by 'piezo_station_department_code'.
    - Categorical columns are filled with the mode value grouped by 'piezo_station_department_code'.
    
    Args:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: The DataFrame with NaN values filled for 'meteo' columns.
    """
    # Identify columns related to "meteo"
    meteo_columns = [col for col in df.columns if "meteo" in col]
    
    # Separate numeric and categorical "meteo" columns
    numeric_meteo_columns = df[meteo_columns].select_dtypes(include=['float64', 'int64']).columns
    categorical_meteo_columns = df[meteo_columns] \
        .select_dtypes(include=['object', 'category']) \
        .columns    
    # Group by department name
    grouped = df.groupby('region')
    
    # Fill NaN in numeric columns with median
    for col in numeric_meteo_columns:
        df[col] = grouped[col].transform(lambda x: x.fillna(x.median()))
    
    # Fill NaN in categorical columns with mode
    for col in categorical_meteo_columns:
        df[col] = grouped[col].transform(
            lambda x: x.fillna(x.mode()[0] if not x.mode().empty else None)
        )
    
    return df


def complete_nan_national(df: pd.DataFrame) -> pd.DataFrame:
    
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    # Fill NaN in numeric columns with median
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill NaN in categorical columns with mode
    for col in categorical_columns:
        mode_value = df[col].mode()[0] if not df[col].mode().empty else None
        df[col] = df[col].fillna(mode_value)
    
    return df